# AuraMind: Research & Engineering Findings Report

**Project**: AuraMind V3.1  
**Architecture**: Compact Decoder-Only Transformer (~12.2M Parameters)  
**Task**: Multi-Turn Empathetic & Emotional-Support Dialogue Generation  
**Dataset**: Hybrid Real (`EmpatheticDialogues`) + Gated Synthetic Scenarios  

---

## 1. Executive Summary

AuraMind explores the training of compact, specialized conversational models from scratch for empathetic dialogue. Large Language Models (LLMs) often require billions of parameters to hold conversational context; however, in constrained, private, or edge environments, specialized lightweight architectures are critical.

This project investigates the convergence, stability, generation quality, and architectural choices of a **12.2M parameter LLaMA-style decoder-only Transformer** trained on multi-turn emotional support dialogues.

---

## 2. Model Architecture & Specifications

The model adopts modern frontier LLM design primitives:

| Component | Specification | Technical Rationale |
| :--- | :--- | :--- |
| **Parameters** | 12,194,688 (~12.2M) | Highly compact; runnable on commodity CPUs and edge devices. |
| **Context Window** | 384 tokens | Accommodates 3–5 dialogue turns while fitting within minimal VRAM. |
| **Layers (`n_layer`)** | 6 | Balanced depth for hierarchical feature extraction. |
| **Heads (`n_head`)** | 6 (dim = 64) | Head dimension matches modern standards ($d_k = 64$). |
| **Hidden Size (`n_embd`)** | 384 | Evenly divisible by head count. |
| **Activation** | SwiGLU | Superior gradient flow over standard ReLU/GELU; hidden dimension padded to multiples of 8. |
| **Normalization** | Pre-RMSNorm ($\epsilon=10^{-6}$) | Eliminates mean-centering overhead of LayerNorm. |
| **Position Encoding** | RoPE (Rotary Position Embeddings) | Preserves relative token distance without learned absolute embeddings. |
| **Weight Tying** | `token_embedding` $\leftrightarrow$ `lm_head` | Saves 1,572,864 parameters and regularizes vocabulary embeddings. |
| **Attention Kernel** | PyTorch SDPA | Dispatches to FlashAttention / Memory-Efficient attention on CUDA. |

---

## 3. Dataset Pipeline & Data Integrity

### 3.1 Hybrid Dataset Composition
* **Real Corpus**: 17,780 multi-turn human dialogues from Facebook's `EmpatheticDialogues`.
  * Train: 14,224 dialogues
  * Validation (Human-only): 1,778 dialogues
  * Test (Human-only): 1,778 dialogues
* **Synthetic Corpus**: Scenario engine spanning 6 emotional domains:
  * `ACADEMIC`, `CAREER`, `RELATIONSHIPS`, `FAMILY`, `LIFE_TRANSITIONS`, `PERSONAL_GROWTH`.
  * Generated with transparent provenance and strict safety constraints.

### 3.2 Five-Dimension Quality Gate & Guardrails
All synthetic samples were scored across five heuristic dimensions (threshold: $\ge 22.0/25$):
1. **Empathy**: Presence of validation markers (*"sounds", "hear", "understandable", "difficult"*).
2. **Relevance**: Contextual alignment with the user's emotional situation.
3. **Non-Directiveness**: Strict penalty against prescriptive or bossy instructions (*"you must", "just do"*).
4. **Clinical Boundary**: Complete rejection of unlicensed medical/psychiatric diagnosing (*"you have PTSD/depression"*, *"as your doctor"*).
5. **Length Appropriateness**: Enforces realistic dialogue turn lengths (40–280 words).

### 3.3 Zero Data Leakage
Data integrity checks verified through SHA-256 disjunction assertions:
$$\text{Train} \cap \text{Validation} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Validation} \cap \text{Test} = \emptyset$$
*No synthetic data enters validation or test splits.*

---

## 4. Key Engineering Enhancements & Experimental Findings

### 4.1 Response-Only Loss Masking
* **Hypothesis**: In standard causal language modeling, loss is computed across all tokens ($x_t \to x_{t+1}$), including user prompts. In dialogue, predicting user prompts wastes ~50% of parameter bandwidth.
* **Finding**: Masking user tokens with `PAD_ID` (`ignore_index`) forces $100\%$ of gradient backpropagation into the counselor's replies. The model learns turn transitions and empathetic formulations significantly faster per training step.

### 4.2 Repetition Penalty in Small Models
* **Hypothesis**: Models under 50M parameters trained from scratch frequently fall into repetitive loops (*"I hear that you feel... I hear that you feel..."*).
* **Finding**: Incorporating a Keskar et al. (2019) multiplicative penalty ($\alpha = 1.15$) on already-generated tokens during nucleus sampling ($p = 0.90, T = 0.75$) effectively broke repetitive phrase cycles and increased lexical diversity in qualitative testing.

### 4.3 Key-Value (KV) Caching
* **Hypothesis**: Autoregressive generation that recomputes full attention matrices is $O(N^2)$ in context length.
* **Finding**: Implementing layer-wise KV caching reduced token generation complexity to $O(N)$ with near-zero latency overhead on token-by-token decoding.

---

## 5. Quantitative & Qualitative Evaluation

### 5.1 Training Dynamics & Evaluation
* **Vocabulary Coverage**: 4,096 Byte-Level BPE tokens (measured $0.00\%$ UNK rate on dialogue corpus).
* **Weight Initialization**: Gaussian ($\mu=0.0, \sigma=0.02$) with zero biases.
* **Optimization**: AdamW ($\beta_1=0.9, \beta_2=0.95, \text{weight\_decay}=0.10$, decoupled for 1D/2D tensors) with cosine decay and linear warmup.

### 5.2 Qualitative Test Scenarios
The model was tested against 4 emotional prompts:
1. *"I am feeling really anxious about my exam tomorrow."*
2. *"I feel like I am falling behind everyone at work."*
3. *"I have been lonely because my friends seem distant lately."*
4. *"I am having a difficult time being patient with myself."*

---

## 6. Honest Limitations & Future Work

1. **Vocabulary Bottleneck**: A 4,096 vocabulary keeps embedding parameters low (~1.57M), but rare words require multiple subword tokens. Increasing to 8,192 or 16,384 would improve fluency.
2. **Scale vs Fluency**: Training an LLM from scratch at 12.2M parameters means the network must learn basic English syntax and empathy simultaneously. 
3. **Next Steps**:
   - Initialize weights from a compact pre-trained base (e.g., SmolLM-135M or Qwen2.5-0.5B).
   - Implement Direct Preference Optimization (DPO) using paired empathetic vs dismissive responses.
