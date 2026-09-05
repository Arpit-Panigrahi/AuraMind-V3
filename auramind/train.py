"""Training engine, learning rate schedule, evaluation, and diagnostic checkpoints."""

from pathlib import Path
from typing import Dict, Any, Optional
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from .config import TrainingConfig, DataConfig
from .model import DecoderTransformer
from .tokenizer import AuraMindTokenizer


def compute_learning_rate(
    step: int,
    warmup_steps: int,
    total_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Linear warmup followed by cosine decay."""
    if step < warmup_steps:
        return max_lr * (step + 1) / max(warmup_steps, 1)

    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (max_lr - min_lr)


@torch.no_grad()
def evaluate_model(
    model: DecoderTransformer,
    loader: DataLoader,
    device: torch.device,
    pad_id: int,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Computes average cross-entropy loss and perplexity on an evaluation DataLoader."""
    model.eval()
    total_loss = 0.0
    count = 0

    for batch_index, (x, y) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            _, loss = model(x, targets=y, ignore_index=pad_id)

        if not torch.isfinite(loss):
            raise RuntimeError("Evaluation loss is non-finite (NaN / Inf).")

        total_loss += loss.item()
        count += 1

    avg_loss = total_loss / max(count, 1)
    perplexity = math.exp(min(avg_loss, 20.0))  # Cap to prevent math overflow

    return {
        "eval_loss": round(avg_loss, 4),
        "perplexity": round(perplexity, 2),
    }


def train_auramind(
    model: DecoderTransformer,
    tokenizer: AuraMindTokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    training_config: TrainingConfig,
    data_config: DataConfig,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Runs the full training loop with warmup-cosine schedule, gradient clipping, and checkpointing."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    pad_id = tokenizer.pad_id

    # Optimizer with weight decay separation
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": training_config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=training_config.max_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    history: Dict[str, list] = {
        "step": [],
        "train_loss": [],
        "val_loss": [],
        "val_perplexity": [],
        "lr": [],
        "grad_norm": [],
    }

    best_val_loss = float("inf")
    best_checkpoint_path = data_config.checkpoint_dir / "best_model.pt"

    total_steps = training_config.total_steps
    train_iter = iter(train_loader)

    progress_bar = tqdm(range(1, total_steps + 1), desc="AuraMind Training")

    for step in progress_bar:
        model.train()
        lr = compute_learning_rate(
            step - 1,
            warmup_steps=training_config.warmup_steps,
            total_steps=total_steps,
            max_lr=training_config.max_lr,
            min_lr=training_config.min_lr,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            _, loss = model(x, targets=y, ignore_index=pad_id)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Step {step}: Non-finite training loss encountered.")

        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(),
            training_config.grad_clip,
        )

        optimizer.step()

        # Validation interval
        if step % training_config.eval_interval == 0 or step == total_steps:
            metrics = evaluate_model(
                model,
                val_loader,
                device=device,
                pad_id=pad_id,
                max_batches=training_config.eval_batches,
            )
            val_loss = metrics["eval_loss"]
            val_ppl = metrics["perplexity"]

            history["step"].append(step)
            history["train_loss"].append(round(loss.item(), 4))
            history["val_loss"].append(val_loss)
            history["val_perplexity"].append(val_ppl)
            history["lr"].append(lr)
            history["grad_norm"].append(round(grad_norm.item(), 4))

            progress_bar.set_postfix({
                "tr_loss": f"{loss.item():.3f}",
                "val_loss": f"{val_loss:.3f}",
                "val_ppl": f"{val_ppl:.1f}",
            })

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "config": model.config.to_dict(),
                    },
                    best_checkpoint_path,
                )

    # Save training curves plot
    plot_training_curves(history, data_config.artifact_dir / "training_curves.png")

    return {
        "best_val_loss": best_val_loss,
        "best_checkpoint": str(best_checkpoint_path),
        "history": history,
    }


def plot_training_curves(history: Dict[str, list], output_path: Path):
    """Generates and saves diagnostic loss & perplexity plots."""
    if not history["step"]:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    steps = history["step"]
    ax1.plot(steps, history["train_loss"], label="Train Loss", color="#1f77b4", alpha=0.7)
    ax1.plot(steps, history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.set_title("AuraMind Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, history["val_perplexity"], label="Val Perplexity", color="#2ca02c", lw=2)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("Validation Perplexity")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
