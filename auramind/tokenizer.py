"""Byte-Level BPE Tokenizer for AuraMind with dialogue role tokens."""

from pathlib import Path
from typing import List, Dict, Union, Optional
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from .config import (
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    USER_TOKEN,
    COUNSELOR_TOKEN,
    EMOTION_TOKEN,
)


class AuraMindTokenizer:
    """Wrapper around Byte-Level BPE Tokenizer with role and control token management."""

    def __init__(self, tokenizer: Optional[Tokenizer] = None):
        self._tokenizer = tokenizer
        self.special_ids: Dict[str, int] = {}
        if tokenizer is not None:
            self._sync_special_ids()

    def _sync_special_ids(self):
        token_names = [
            ("PAD", PAD_TOKEN),
            ("UNK", UNK_TOKEN),
            ("BOS", BOS_TOKEN),
            ("EOS", EOS_TOKEN),
            ("USER", USER_TOKEN),
            ("COUNSELOR", COUNSELOR_TOKEN),
            ("EMOTION", EMOTION_TOKEN),
        ]
        for name, token in token_names:
            tok_id = self._tokenizer.token_to_id(token)
            if tok_id is None:
                # If loading a legacy tokenizer without EMOTION token, fallback gracefully
                if name == "EMOTION":
                    continue
                raise RuntimeError(f"Special token {token} not found in tokenizer vocabulary.")
            self.special_ids[name] = tok_id

    @property
    def pad_id(self) -> int:
        return self.special_ids["PAD"]

    @property
    def unk_id(self) -> int:
        return self.special_ids["UNK"]

    @property
    def bos_id(self) -> int:
        return self.special_ids["BOS"]

    @property
    def eos_id(self) -> int:
        return self.special_ids["EOS"]

    @property
    def user_id(self) -> int:
        return self.special_ids["USER"]

    @property
    def counselor_id(self) -> int:
        return self.special_ids["COUNSELOR"]

    @property
    def emotion_id(self) -> Optional[int]:
        return self.special_ids.get("EMOTION")

    def token_to_id(self, token: str) -> Optional[int]:
        if self._tokenizer is None:
            return None
        return self._tokenizer.token_to_id(token)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @classmethod
    def train(
        cls,
        corpus_paths: List[Union[str, Path]],
        vocab_size: int = 4096,
        min_frequency: int = 2,
    ) -> "AuraMindTokenizer":
        """Trains a new Byte-Level BPE tokenizer on raw dialogue text files."""
        special_tokens = [
            AddedToken(PAD_TOKEN, special=True),
            AddedToken(UNK_TOKEN, special=True),
            AddedToken(BOS_TOKEN, special=True),
            AddedToken(EOS_TOKEN, special=True),
            AddedToken(USER_TOKEN, special=True),
            AddedToken(COUNSELOR_TOKEN, special=True),
            AddedToken(EMOTION_TOKEN, special=True),
        ]

        tok = Tokenizer(BPE(unk_token=UNK_TOKEN))
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tok.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=True,
        )

        str_paths = [str(p) for p in corpus_paths]
        tok.train(str_paths, trainer)

        instance = cls(tok)
        return instance

    def encode(self, text: str) -> List[int]:
        """Encodes text to a list of token IDs."""
        return self._tokenizer.encode(text).ids

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decodes token IDs back to a string."""
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def save(self, path: Union[str, Path]):
        """Saves the tokenizer definition JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AuraMindTokenizer":
        """Loads a saved tokenizer from file."""
        tok = Tokenizer.from_file(str(path))
        return cls(tok)

    def measure_corpus(self, texts: List[str]) -> Dict[str, float]:
        """Computes diagnostic statistics: avg tokens per text and byte compression ratio."""
        total_chars = sum(len(t) for t in texts)
        total_tokens = sum(len(self.encode(t)) for t in texts)
        compression = total_chars / max(total_tokens, 1)
        return {
            "total_characters": total_chars,
            "total_tokens": total_tokens,
            "compression_ratio_char_per_token": round(compression, 2),
            "avg_tokens_per_dialogue": round(total_tokens / max(len(texts), 1), 1),
        }
