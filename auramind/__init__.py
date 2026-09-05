"""AuraMind: Compact Decoder-Only Transformer for Empathetic Dialogue Generation."""

__version__ = "3.1.0-enhanced"

from .config import ModelConfig, TrainingConfig, DataConfig, SamplingConfig
from .model import DecoderTransformer
from .tokenizer import AuraMindTokenizer
