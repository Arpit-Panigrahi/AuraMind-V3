"""Dataset pipeline for EmpatheticDialogues, synthetic integration, and response-only loss masking."""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import csv
import json
import random
import tarfile
import urllib.request
import unicodedata
import hashlib
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader

from .config import (
    DataConfig,
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    USER_TOKEN,
    COUNSELOR_TOKEN,
    EMOTION_TOKEN,
)
from .tokenizer import AuraMindTokenizer
from .synthetic import (
    build_synthetic_dialogue,
    safety_filter,
    validate_structure,
    score_dialogue,
    SemanticDeduplicator,
)


def clean_text(text: str) -> str:
    """Normalizes Unicode, strips control characters, and cleans token artifacts."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch == "\n" or unicodedata.category(ch)[0] != "C")
    text = text.replace("_comma_", ",")
    return " ".join(text.split()).strip()


def download_and_extract_empathetic(raw_dir: Path) -> Path:
    """Downloads and extracts the EmpatheticDialogues corpus if not already present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    tar_url = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
    tar_path = raw_dir / "empatheticdialogues.tar.gz"

    if not tar_path.exists():
        print(f"📥 Downloading EmpatheticDialogues from {tar_url}...")
        urllib.request.urlretrieve(tar_url, tar_path)

    csv_candidates = list(raw_dir.rglob("train.csv"))
    if not csv_candidates:
        print("📦 Extracting EmpatheticDialogues archive...")
        with tarfile.open(tar_path, "r:gz") as archive:
            archive.extractall(raw_dir)
        csv_candidates = list(raw_dir.rglob("train.csv"))

    if not csv_candidates:
        raise FileNotFoundError("train.csv was not found after downloading/extracting EmpatheticDialogues.")

    return csv_candidates[0]


def load_real_empathetic_records(train_csv_path: Path) -> List[Dict[str, str]]:
    """Loads and reconstructs multi-turn dialogues from EmpatheticDialogues train.csv with emotion prefix."""
    conversations = defaultdict(list)
    conv_emotions = {}
    with open(train_csv_path, "r", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            conv_id = (row.get("conv_id") or "").strip()
            utterance = clean_text(row.get("utterance") or "")
            context_emotion = clean_text(row.get("context") or "").lower()
            if conv_id:
                if context_emotion and conv_id not in conv_emotions:
                    conv_emotions[conv_id] = context_emotion
                if utterance:
                    conversations[conv_id].append(utterance)

    records = []
    seen = set()
    for conv_id, turns in conversations.items():
        if len(turns) < 2:
            continue
        lines = []
        emotion = conv_emotions.get(conv_id, "").strip()
        if emotion:
            lines.append(f"{EMOTION_TOKEN} {emotion}")
        for index, utterance in enumerate(turns):
            role = USER_TOKEN if index % 2 == 0 else COUNSELOR_TOKEN
            lines.append(f"{role} {utterance}")
        doc = "\n".join(lines) + f" {EOS_TOKEN}"
        digest = hashlib.sha256(doc.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        records.append({
            "id": f"real_{len(records) + 1:06d}",
            "doc": doc,
            "hash": digest,
            "source": "real_empathetic",
            "emotion": emotion,
        })
    return records


class TokenBlockDataset(Dataset):
    """
    Dialogue-preserving token block dataset with optional response-only loss masking.
    
    When mask_user_loss=True:
      User turns are masked with pad_id (ignore_index) in the target tensor.
      The model only computes loss on predicting the counselor's tokens and <|eos|>.
    """

    def __init__(
        self,
        records: List[Dict[str, str]],
        tokenizer: AuraMindTokenizer,
        block_size: int,
        mask_user_loss: bool = True,
    ):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.mask_user_loss = mask_user_loss

        self.input_chunks: List[torch.Tensor] = []
        self.target_chunks: List[torch.Tensor] = []

        user_id = tokenizer.user_id
        counselor_id = tokenizer.counselor_id
        pad_id = tokenizer.pad_id
        emotion_id = tokenizer.emotion_id

        for record in records:
            ids = tokenizer.encode(record["doc"])
            if len(ids) < 2:
                continue

            # Create windows strictly within this dialogue boundary
            for start in range(0, max(len(ids) - 1, 1), block_size):
                window_ids = ids[start : start + block_size + 1]
                if len(window_ids) < 2:
                    break

                # Create input (x) and target (y)
                raw_x = window_ids[:-1]
                raw_y = window_ids[1:]

                # Padding
                pad_len = block_size - len(raw_x)
                if pad_len > 0:
                    raw_x = raw_x + [pad_id] * pad_len
                    raw_y = raw_y + [pad_id] * pad_len

                target_y = list(raw_y)
                if mask_user_loss:
                    # Determine whether each target belongs to the counselor turn
                    # In a sequence: <|emotion|> ... <|user|> ... <|counselor|> ...
                    # Targets predicting the prompt/emotion/user turn are masked out.
                    role = "user"
                    for idx, (inp_token, tgt_token) in enumerate(zip(raw_x, raw_y)):
                        if inp_token == pad_id or tgt_token == pad_id:
                            target_y[idx] = pad_id
                            continue

                        if inp_token == user_id:
                            role = "user"
                        elif inp_token == counselor_id:
                            role = "counselor"
                        elif emotion_id is not None and inp_token == emotion_id:
                            role = "emotion"

                        if tgt_token == counselor_id:
                            # Keep target predicting the role transition to counselor
                            pass
                        elif role in ("user", "emotion"):
                            # Mask out token predicting prompt/user content
                            target_y[idx] = pad_id

                self.input_chunks.append(torch.tensor(raw_x, dtype=torch.long))
                self.target_chunks.append(torch.tensor(target_y, dtype=torch.long))

        if not self.input_chunks:
            raise RuntimeError("No valid dialogue windows produced from records.")

    def __len__(self) -> int:
        return len(self.input_chunks)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_chunks[index], self.target_chunks[index]


def create_dataloaders(
    train_records: List[Dict[str, str]],
    val_records: List[Dict[str, str]],
    test_records: List[Dict[str, str]],
    tokenizer: AuraMindTokenizer,
    block_size: int,
    batch_size: int,
    mask_user_loss: bool = True,
    device_type: str = "cpu",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Builds PyTorch DataLoaders for train, validation, and test splits."""
    train_ds = TokenBlockDataset(train_records, tokenizer, block_size, mask_user_loss=mask_user_loss)
    val_ds = TokenBlockDataset(val_records, tokenizer, block_size, mask_user_loss=mask_user_loss)
    test_ds = TokenBlockDataset(test_records, tokenizer, block_size, mask_user_loss=mask_user_loss)

    pin = (device_type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=pin,
    )

    return train_loader, val_loader, test_loader
