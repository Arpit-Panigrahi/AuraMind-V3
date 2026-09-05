"""Modern LLaMA-style decoder Transformer implementation with RoPE, RMSNorm, SwiGLU, and KV caching."""

from typing import Optional, Tuple, List, Dict, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return (x_norm.to(self.weight.dtype)) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al., 2021)."""

    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head dimension must be even, got {dim}.")

        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        if offset + seq_len > self.max_seq_len:
            raise ValueError(
                f"Requested position range [{offset}, {offset + seq_len}) exceeds max_seq_len ({self.max_seq_len})."
            )
        return (
            self.cos_cached[offset : offset + seq_len],
            self.sin_cached[offset : offset + seq_len],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # cos, sin: [seq_len, head_dim] -> unsqueeze to [1, 1, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE, PyTorch SDPA, and KV-cache support."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        self.dropout_p = config.dropout

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(self.head_dim, config.block_size)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        offset: int = 0,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        try:
            cos, sin = self.rope(seq_len, offset=offset)
        except TypeError:
            cos, sin = self.rope(seq_len)
        q, k = apply_rope(q, k, cos, sin)

        next_kv = None
        if use_cache:
            if past_kv is not None:
                past_k, past_v = past_kv
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            next_kv = (k, v)

            # When using KV cache at inference (seq_len == 1), the new query attends to all key tokens.
            # No causal mask needed since Q is at the absolute end.
            is_causal = (seq_len > 1)
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=(self.dropout_p if self.training else 0.0),
                is_causal=is_causal,
            )
        else:
            # Training / full sequence evaluation
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=(self.dropout_p if self.training else 0.0),
                is_causal=True,
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.n_embd)
        return self.out_proj(attn_out), next_kv


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit (Shazeer, 2020) with hidden dimension multiple of 8."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = int((8 * config.n_embd) / 3)
        hidden_dim = 8 * (hidden_dim // 8)

        self.gate = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.up = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    """Single Transformer Decoder block with Pre-RMSNorm and SwiGLU."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        offset: int = 0,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, next_kv = self.attn(
            self.norm1(x),
            past_kv=past_kv,
            offset=offset,
            use_cache=use_cache,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, next_kv


class DecoderTransformer(nn.Module):
    """AuraMind Decoder-Only Transformer."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Weight tying: lm_head shares weights with token embedding
        self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        offset: int = 0,
        use_cache: bool = False,
        ignore_index: int = -100,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass.
        Args:
            idx: Token indices of shape [batch, seq_len]
            targets: Target token indices of shape [batch, seq_len] (optional)
            past_kvs: List of past key/values per layer (optional, for cached generation)
            offset: Current sequence position offset for RoPE (used with KV caching)
            use_cache: Whether to update and return KV cache
            ignore_index: Target value ignored in cross entropy loss
        """
        if idx.dtype != torch.long:
            raise TypeError(f"Input token IDs must be torch.long, got {idx.dtype}.")

        batch, seq_len = idx.shape
        if offset + seq_len > self.config.block_size:
            raise ValueError(
                f"Total sequence length {offset + seq_len} exceeds block size {self.config.block_size}."
            )

        # Defensive bounds check against GPU illegal memory access asserts
        if idx.numel() > 0:
            min_id = int(idx.detach().min().item())
            max_id = int(idx.detach().max().item())
            if min_id < 0 or max_id >= self.config.vocab_size:
                raise ValueError(
                    f"Input token ID out of range [0, {self.config.vocab_size}): min={min_id}, max={max_id}"
                )

        x = self.dropout(self.token_embedding(idx))

        next_kvs = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, next_kv = block(x, past_kv=past_kv, offset=offset, use_cache=use_cache)
            if use_cache:
                next_kvs.append(next_kv)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            if targets.dtype != torch.long:
                raise TypeError(f"Targets must be torch.long, got {targets.dtype}.")

            valid = targets[targets != ignore_index]
            if valid.numel() > 0:
                min_target = int(valid.detach().min().item())
                max_target = int(valid.detach().max().item())
                if min_target < 0 or max_target >= self.config.vocab_size:
                    raise ValueError(
                        f"Target token ID out of range [0, {self.config.vocab_size}): min={min_target}, max={max_target}"
                    )

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=ignore_index,
            )

        if use_cache:
            return logits, loss, next_kvs
        return logits, loss

    def parameter_count(self) -> Dict[str, int]:
        """Returns detailed parameter statistics."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = self.token_embedding.weight.numel()
        blocks_params = sum(p.numel() for p in self.blocks.parameters())
        norm_params = self.norm.weight.numel()

        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "embedding_parameters": emb_params,
            "transformer_blocks_parameters": blocks_params,
            "norm_parameters": norm_params,
            "weight_tying": True,
        }
