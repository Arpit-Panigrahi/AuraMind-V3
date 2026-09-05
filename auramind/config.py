"""Configuration classes and default hyperparameters for AuraMind."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any

# Special tokens
PAD_TOKEN = "<|pad|>"
UNK_TOKEN = "<|unk|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
USER_TOKEN = "<|user|>"
COUNSELOR_TOKEN = "<|counselor|>"

DEFAULT_SPECIAL_TOKENS = [
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    USER_TOKEN,
    COUNSELOR_TOKEN,
]


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    block_size: int = 384
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.10

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})."
            )
        if self.block_size <= 0 or self.vocab_size <= 0:
            raise ValueError("Invalid model dimensions: block_size and vocab_size must be positive.")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(**data)


@dataclass
class TrainingConfig:
    total_steps: int = 3000
    warmup_steps: int = 75
    eval_interval: int = 50
    eval_batches: int = 25
    batch_size: int = 16
    cpu_batch_size: int = 2
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.10
    grad_clip: float = 1.0
    smoke_test: bool = False
    smoke_steps: int = 25
    seed: int = 42
    
    # Enhancement 1: Response-only loss masking
    # When True, the cross-entropy loss is computed exclusively on counselor turns.
    # User prompts are masked with ignore_index, focusing 100% of gradient updates on response generation.
    mask_user_loss: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SamplingConfig:
    temperature: float = 0.75
    top_p: float = 0.90
    # Enhancement 2: Repetition penalty (Keskar et al., 2019)
    # Values > 1.0 penalize already generated tokens, preventing repetitive empathetic phrasing loops.
    repetition_penalty: float = 1.15
    max_new_tokens: int = 80
    min_new_tokens: int = 5
    use_cache: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataConfig:
    root_dir: Path = field(default_factory=lambda: Path("auramind_data"))
    target_synthetic: int = 10_000
    real_train_fraction: float = 0.80
    real_val_fraction: float = 0.10
    semantic_dup_threshold: float = 0.88
    quality_threshold: float = 22.0
    preserve_dialogue_boundaries: bool = True
    reuse_existing: bool = True

    @property
    def data_dir(self) -> Path:
        return self.root_dir / "data" / "v2"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def artifact_dir(self) -> Path:
        return self.root_dir / "artifacts"

    @property
    def checkpoint_dir(self) -> Path:
        return self.root_dir / "checkpoints"

    def create_dirs(self):
        for path in [self.root_dir, self.data_dir, self.raw_dir, self.artifact_dir, self.checkpoint_dir]:
            path.mkdir(parents=True, exist_ok=True)
