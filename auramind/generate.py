"""Inference and generation engine with Min-p, top-p sampling, repetition penalty, emotion conditioning, and KV caching."""

from typing import List, Optional, Set
import re
import torch
import torch.nn.functional as F

from .config import SamplingConfig, USER_TOKEN, COUNSELOR_TOKEN, EOS_TOKEN, EMOTION_TOKEN
from .model import DecoderTransformer
from .tokenizer import AuraMindTokenizer

# Lightweight emotion dictionary matching EmpatheticDialogues emotion taxonomy
EMOTION_KEYWORDS = {
    "anxious": ["anxious", "anxiety", "nervous", "worried", "worry", "panic", "scared", "fear", "exam", "presentation", "interview", "test"],
    "sad": ["sad", "depressed", "unhappy", "crying", "tears", "hopeless", "grief", "down", "heartbroken", "loss"],
    "lonely": ["lonely", "alone", "isolated", "nobody", "friendless", "abandoned"],
    "overwhelmed": ["overwhelmed", "drowning", "too much", "burnout", "burned out", "exhausted", "tired"],
    "angry": ["angry", "furious", "mad", "frustrated", "irritated", "annoyed"],
    "ashamed": ["ashamed", "embarrassed", "shame", "guilty", "guilt", "humiliated"],
    "hopeful": ["hopeful", "optimistic", "looking forward", "better"],
    "proud": ["proud", "accomplished", "succeeded", "won", "passed"],
    "grateful": ["grateful", "thankful", "blessed", "appreciate"],
    "caring": ["caring", "help", "support", "love", "friend"],
}


def detect_emotion(text: str) -> str:
    """Classifies user utterance into an empathetic emotion category for prompt conditioning."""
    text_lower = text.lower()
    for emotion, keywords in EMOTION_KEYWORDS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", text_lower):
                return emotion
    return "anxious"  # Default empathetic supportive posture


def top_p_sample(
    logits: torch.Tensor,
    temperature: float = 0.75,
    top_p: float = 0.90,
    min_p: float = 0.05,
    repetition_penalty: float = 1.15,
    generated_tokens: Optional[Set[int]] = None,
) -> torch.Tensor:
    """
    Samples next token with repetition penalty, Min-p truncation, temperature scaling, and nucleus (top-p) filtering.
    
    Args:
        logits: Unnormalized token scores of shape [batch, vocab_size] (or [vocab_size])
        temperature: Temperature scaling factor (>0).
        top_p: Nucleus cumulative probability mass (0.0 < top_p <= 1.0).
        min_p: Dynamic probability threshold relative to the most likely token (0.0 <= min_p < 1.0).
        repetition_penalty: Multiplicative penalty for previously generated tokens (1.0 = off).
        generated_tokens: Set of token IDs already generated in the current response.
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    # Repetition penalty (Keskar et al., 2019)
    if repetition_penalty > 1.0 and generated_tokens:
        for tok_id in generated_tokens:
            if logits[0, tok_id] > 0:
                logits[0, tok_id] /= repetition_penalty
            else:
                logits[0, tok_id] *= repetition_penalty

    # Temperature scaling
    scaled_logits = logits / max(temperature, 1e-5)
    probs = F.softmax(scaled_logits, dim=-1)

    # Enhancement: Min-p dynamic thresholding
    # Truncates tail tokens whose probability is less than min_p * max_prob
    if min_p > 0.0:
        max_prob = probs.max(dim=-1, keepdim=True).values
        min_p_mask = probs < (min_p * max_prob)
        scaled_logits = scaled_logits.masked_fill(min_p_mask, -float("inf"))

    # Nucleus (top-p) filtering
    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Mask out tokens beyond cumulative probability threshold
    sorted_mask = cumulative_probs > top_p
    # Shift mask right by 1 so at least one token is kept
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    filtered_logits = sorted_logits.masked_fill(sorted_mask, -float("inf"))
    final_probs = F.softmax(filtered_logits, dim=-1)

    if not torch.isfinite(final_probs).all():
        raise RuntimeError("Generation probabilities contain non-finite values (NaN / Inf).")

    sampled_position = torch.multinomial(final_probs, num_samples=1)
    next_token = sorted_indices.gather(-1, sampled_position)
    return next_token


@torch.no_grad()
def generate_response(
    model: DecoderTransformer,
    tokenizer: AuraMindTokenizer,
    user_text: str,
    emotion: Optional[str] = None,
    sampling_config: Optional[SamplingConfig] = None,
    device: Optional[torch.device] = None,
) -> str:
    """
    Generates an empathetic counselor response for a given user prompt.
    Supports emotion-conditioned prefixing, min-p sampling, and KV-cached generation.
    """
    if sampling_config is None:
        sampling_config = SamplingConfig()

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Determine emotion conditioning
    if emotion is None:
        emotion = detect_emotion(user_text)

    # Build emotion-conditioned prompt
    if tokenizer.emotion_id is not None:
        prompt = f"{EMOTION_TOKEN} {emotion}\n{USER_TOKEN} {user_text.strip()}\n{COUNSELOR_TOKEN}"
    else:
        prompt = f"{USER_TOKEN} {user_text.strip()}\n{COUNSELOR_TOKEN}"

    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("Prompt produced no token IDs.")

    use_cache = sampling_config.use_cache
    eos_id = tokenizer.eos_id

    generated_ids: List[int] = []
    seen_token_set: Set[int] = set()

    if use_cache:
        # 1. Warm up prompt and build KV cache
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        logits, _, past_kvs = model(input_ids, use_cache=True, offset=0)
        next_token_logits = logits[:, -1, :]

        next_token = top_p_sample(
            next_token_logits,
            temperature=sampling_config.temperature,
            top_p=sampling_config.top_p,
            min_p=sampling_config.min_p,
            repetition_penalty=sampling_config.repetition_penalty,
            generated_tokens=seen_token_set,
        ).item()

        if next_token == eos_id:
            return ""

        generated_ids.append(next_token)
        seen_token_set.add(next_token)

        # 2. Autoregressive loop with cached past_kvs
        current_offset = len(prompt_ids)
        for _ in range(sampling_config.max_new_tokens - 1):
            curr_input = torch.tensor([[next_token]], dtype=torch.long, device=device)
            logits, _, past_kvs = model(
                curr_input,
                past_kvs=past_kvs,
                offset=current_offset,
                use_cache=True,
            )
            current_offset += 1
            next_token_logits = logits[:, -1, :]

            next_token = top_p_sample(
                next_token_logits,
                temperature=sampling_config.temperature,
                top_p=sampling_config.top_p,
                min_p=sampling_config.min_p,
                repetition_penalty=sampling_config.repetition_penalty,
                generated_tokens=seen_token_set,
            ).item()

            if next_token == eos_id:
                break

            generated_ids.append(next_token)
            seen_token_set.add(next_token)

            if current_offset >= model.config.block_size - 1:
                break
    else:
        # Standard full-context autoregression (O(N^2))
        curr_ids = list(prompt_ids)
        for _ in range(sampling_config.max_new_tokens):
            input_tensor = torch.tensor([curr_ids], dtype=torch.long, device=device)
            logits, _ = model(input_tensor, use_cache=False)
            next_token_logits = logits[:, -1, :]

            next_token = top_p_sample(
                next_token_logits,
                temperature=sampling_config.temperature,
                top_p=sampling_config.top_p,
                min_p=sampling_config.min_p,
                repetition_penalty=sampling_config.repetition_penalty,
                generated_tokens=seen_token_set,
            ).item()

            if next_token == eos_id:
                break

            generated_ids.append(next_token)
            seen_token_set.add(next_token)
            curr_ids.append(next_token)

            if len(curr_ids) >= model.config.block_size:
                break

    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return decoded
