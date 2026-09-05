"""Main entry point for AuraMind: train, generate, evaluate, or interactive chat."""

import argparse
from pathlib import Path
import json
import torch

from auramind.config import ModelConfig, TrainingConfig, DataConfig, SamplingConfig
from auramind.model import DecoderTransformer
from auramind.tokenizer import AuraMindTokenizer
from auramind.generate import generate_response
from auramind.synthetic import (
    build_synthetic_dialogue,
    safety_filter,
    validate_structure,
    score_dialogue,
    SemanticDeduplicator,
)
from auramind.dataset import (
    download_and_extract_empathetic,
    load_real_empathetic_records,
    create_dataloaders,
)
from auramind.train import train_auramind, evaluate_model


def cmd_info(args):
    config = ModelConfig()
    model = DecoderTransformer(config)
    counts = model.parameter_count()
    print("=" * 60)
    print("AuraMind Architecture & Parameter Summary")
    print("=" * 60)
    print(f"Total Parameters:      {counts['total_parameters']:,}")
    print(f"Trainable Parameters:  {counts['trainable_parameters']:,}")
    print(f"Embedding / LM Head:   {counts['embedding_parameters']:,} (Weight-tied)")
    print(f"Transformer Blocks:    {counts['transformer_blocks_parameters']:,} (6 layers)")
    print(f"Context Window:        {config.block_size} tokens")
    print(f"Vocab Target:          {config.vocab_size} tokens")
    print("=" * 60)


def cmd_train(args):
    data_cfg = DataConfig(root_dir=Path(args.data_dir))
    data_cfg.create_dirs()

    train_cfg = TrainingConfig(
        total_steps=(25 if args.smoke else args.steps),
        batch_size=args.batch_size,
        mask_user_loss=(not args.no_mask_user_loss),
        smoke_test=args.smoke,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    print(f"Response-only loss masking: {train_cfg.mask_user_loss}")

    # 1. Prepare / Load data
    train_file = data_cfg.data_dir / "train.jsonl"
    val_file = data_cfg.data_dir / "validation.jsonl"
    test_file = data_cfg.data_dir / "test.jsonl"

    if not (train_file.exists() and val_file.exists()):
        print("Preparing dataset (real + synthetic)...")
        csv_path = download_and_extract_empathetic(data_cfg.raw_dir)
        real_records = load_real_empathetic_records(csv_path)

        # Synthetic generation with deduplication
        dedup = SemanticDeduplicator()
        synth_records = []
        synth_target = 100 if args.smoke else data_cfg.target_synthetic
        print(f"Generating {synth_target} synthetic dialogues...")

        for i in range(synth_target):
            dialogue = build_synthetic_dialogue(i)
            is_safe, _ = safety_filter(dialogue)
            if not is_safe or not validate_structure(dialogue):
                continue
            scores = score_dialogue(dialogue)
            if scores["total"] < data_cfg.quality_threshold:
                continue
            if not dedup.accept(dialogue):
                continue
            synth_records.append({
                "id": f"synth_{len(synth_records) + 1:06d}",
                "doc": dialogue,
                "source": "synthetic_v2",
            })

        all_records = real_records + synth_records
        import random
        random.seed(42)
        random.shuffle(all_records)

        n_train = int(len(all_records) * 0.8)
        n_val = int(len(all_records) * 0.1)

        train_records = all_records[:n_train]
        val_records = all_records[n_train : n_train + n_val]
        test_records = all_records[n_train + n_val :]

        def write_jsonl(path, recs):
            with open(path, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")

        write_jsonl(train_file, train_records)
        write_jsonl(val_file, val_records)
        write_jsonl(test_file, test_records)

        # Build corpus file for tokenizer
        corpus_path = data_cfg.data_dir / "train_corpus.txt"
        with open(corpus_path, "w", encoding="utf-8") as f:
            for r in train_records:
                f.write(r["doc"] + "\n")

        tokenizer = AuraMindTokenizer.train([corpus_path], vocab_size=4096)
        tokenizer.save(data_cfg.artifact_dir / "tokenizer.json")
    else:
        print("Loading cached dataset...")
        def read_jsonl(path):
            recs = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    recs.append(json.loads(line))
            return recs

        train_records = read_jsonl(train_file)
        val_records = read_jsonl(val_file)
        test_records = read_jsonl(test_file)
        tokenizer = AuraMindTokenizer.load(data_cfg.artifact_dir / "tokenizer.json")

    # 2. Setup model
    model_cfg = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = DecoderTransformer(model_cfg)

    # 3. Create loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        tokenizer=tokenizer,
        block_size=model_cfg.block_size,
        batch_size=train_cfg.batch_size if device.type == "cuda" else train_cfg.cpu_batch_size,
        mask_user_loss=train_cfg.mask_user_loss,
        device_type=device.type,
    )

    # 4. Train
    print("Starting training...")
    result = train_auramind(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        training_config=train_cfg,
        data_config=data_cfg,
        device=device,
    )
    print(f"Training complete! Best validation loss: {result['best_val_loss']}")
    print(f"Checkpoint saved to: {result['best_checkpoint']}")


def cmd_generate(args):
    data_cfg = DataConfig(root_dir=Path(args.data_dir))
    tok_path = data_cfg.artifact_dir / "tokenizer.json"
    ckpt_path = data_cfg.checkpoint_dir / "best_model.pt"

    if not tok_path.exists() or not ckpt_path.exists():
        print("Missing trained tokenizer or checkpoint. Please train the model first.")
        return

    tokenizer = AuraMindTokenizer.load(tok_path)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = ModelConfig.from_dict(ckpt["config"])
    model = DecoderTransformer(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    sampling_cfg = SamplingConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_tokens,
        use_cache=args.use_cache,
    )

    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        user_text=args.prompt,
        emotion=args.emotion,
        sampling_config=sampling_cfg,
        device=device,
    )
    print(f"\nUser: {args.prompt}")
    print(f"Counselor: {response}\n")


def cmd_chat(args):
    data_cfg = DataConfig(root_dir=Path(args.data_dir))
    tok_path = data_cfg.artifact_dir / "tokenizer.json"
    ckpt_path = data_cfg.checkpoint_dir / "best_model.pt"

    if not tok_path.exists() or not ckpt_path.exists():
        print("Missing trained tokenizer or checkpoint. Please train the model first.")
        return

    tokenizer = AuraMindTokenizer.load(tok_path)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = ModelConfig.from_dict(ckpt["config"])
    model = DecoderTransformer(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    sampling_cfg = SamplingConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        use_cache=True,
    )

    print("=" * 60)
    print("AuraMind Empathetic Counselor - Interactive Session")
    print("Type 'exit' or 'quit' to end.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Take care. Goodbye!")
                break

            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                user_text=user_input,
                sampling_config=sampling_cfg,
                device=device,
            )
            print(f"Counselor: {response}")
        except (KeyboardInterrupt, EOFError):
            print("\nTake care. Goodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="AuraMind V3 Enhanced")
    subparsers = parser.add_subparsers(dest="command")

    # Info
    subparsers.add_parser("info", help="Display model architecture and parameters")

    # Train
    train_parser = subparsers.add_parser("train", help="Train AuraMind model")
    train_parser.add_argument("--data-dir", default="auramind_data")
    train_parser.add_argument("--steps", type=int, default=3000)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--smoke", action="store_true", help="Run quick 25-step smoke test")
    train_parser.add_argument("--no-mask-user-loss", action="store_true", help="Disable response-only loss masking")
    train_parser.add_argument("--cpu", action="store_true", help="Force CPU mode")

    # Generate
    gen_parser = subparsers.add_parser("generate", help="Generate response for a single prompt")
    gen_parser.add_argument("prompt", type=str, help="User prompt to respond to")
    gen_parser.add_argument("--emotion", type=str, default=None, help="Explicit emotion conditioning (e.g. anxious, sad, proud)")
    gen_parser.add_argument("--data-dir", default="auramind_data")
    gen_parser.add_argument("--temperature", type=float, default=0.75)
    gen_parser.add_argument("--top-p", type=float, default=0.90)
    gen_parser.add_argument("--min-p", type=float, default=0.05, help="Min-p dynamic truncation threshold")
    gen_parser.add_argument("--repetition-penalty", type=float, default=1.15)
    gen_parser.add_argument("--max-tokens", type=int, default=80)
    gen_parser.add_argument("--use-cache", action="store_true", default=True)
    gen_parser.add_argument("--cpu", action="store_true", help="Force CPU mode")

    # Chat
    chat_parser = subparsers.add_parser("chat", help="Interactive terminal chat")
    chat_parser.add_argument("--data-dir", default="auramind_data")
    chat_parser.add_argument("--temperature", type=float, default=0.75)
    chat_parser.add_argument("--top-p", type=float, default=0.90)
    chat_parser.add_argument("--min-p", type=float, default=0.05, help="Min-p dynamic truncation threshold")
    chat_parser.add_argument("--repetition-penalty", type=float, default=1.15)
    chat_parser.add_argument("--cpu", action="store_true", help="Force CPU mode")

    args = parser.parse_args()
    if args.command == "info":
        cmd_info(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "chat":
        cmd_chat(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
