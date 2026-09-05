# AuraMind: Compact Decoder-Only Empathetic Transformer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Colab GPU](https://img.shields.io/badge/Trained%20on-Tesla%20T4%20GPU-green.svg)](https://colab.research.google.com)
[![Validation PPL](https://img.shields.io/badge/Validation%20PPL-31.43-brightgreen.svg)](docs/FINDINGS.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AuraMind** is a compact, decoder-only Transformer (~12.2M parameters) trained strictly from scratch for multi-turn empathetic and emotional-support dialogue generation. Built on modern frontier LLM design primitives (LLaMA/Mistral-style), it combines **RMSNorm**, **RoPE (Rotary Position Embeddings)**, **SwiGLU feed-forward networks**, **weight-tied embeddings**, and PyTorch native **SDPA (FlashAttention)**.

---

## 🌟 Benchmark Results & Key Highlights

* **Full 3,000-Step Convergence on Tesla T4 GPU (58m 33s)**:
  * Initial Batch Loss: `8.536` $\to$ Final Training Loss: **`0.894`** (Step) / **`1.601`** (Train Eval)
  * Held-Out Validation Loss: **`3.448`** (Perplexity: **`31.43`**)
  * Held-Out Test Loss: **`3.502`** (Perplexity: **`33.17`**)
  * Final Gradient Norm: `0.55` (exceptionally stable convergence with AdamW decoupled decay)
* **Emotion-Conditioned Token Prefixing (`<|emotion|>`)**: Prepends ground-truth emotion categories (e.g., `anxious`, `sad`, `lonely`, `proud`) from `EmpatheticDialogues` context and synthetic domain tags. Resolves sentiment inversion ("toxic positivity" where distressed users received cheerful openers) by explicitly conditioning cross-attention.
* **Min-$p$ Sampling ($p_{\text{min}} = 0.05$)**: Truncates unlikely tail noise tokens whose probability is less than $p_{\text{min}} \times p_{\max}$, preventing hallucinations far better than static top-$p$ on compact models.
* **Response-Only Loss Masking**: Masks user prompts and emotion prefixes with `PAD_ID` (`ignore_index`), focusing 100% of gradient backpropagation on counselor responses rather than memorizing user inputs.
* **Repetition Penalty**: Integrated Keskar et al. (2019) multi-token repetition penalty ($\alpha = 1.15$) to prevent repetitive loops in nucleus sampling ($p = 0.90, T = 0.75$).
* **$O(N)$ KV-Cached Inference**: Step-by-step cached attention across all 6 decoder layers for fast autoregressive token generation.
* **Curated Hybrid Dataset (24,224 Dialogues)**: 14,224 human dialogues from `EmpatheticDialogues` combined with 10,000 synthetic dialogues validated through a 5-dimension quality judge and strict clinical boundary guardrails.
* **Zero Data Leakage**: Enforced through SHA-256 assertions: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$.

Detailed research, convergence dynamics, and behavioral analysis are documented in [**docs/FINDINGS.md**](docs/FINDINGS.md).

---

## 📊 Training Dynamics (3,000 Steps on Tesla T4)

![AuraMind 3000-Step Training Curves](assets/training_curves.png)

The loss curve demonstrates rapid exponential descent during the first 500 steps, transitioning into stable monotonic refinement with zero overfitting divergence between validation and test splits.

---

## 📐 Architecture Overview

```
Input Tokens [B, S]
       │
       ▼
Token Embedding (4096, 384) [Weight-tied to LM Head]
       │
       ▼
┌──────────────────────────────────────────────┐
│  TransformerBlock (x6 Layers)                │
│                                              │
│  x ──► RMSNorm ──► CausalSelfAttention ──► + │
│  │                    (RoPE + SDPA)        │ │
│  │                                         │ │
│  └───► RMSNorm ──► SwiGLU (dim: 1024) ────► + │
└──────────────────────────────────────────────┘
       │
       ▼
Final RMSNorm (384)
       │
       ▼
LM Head (384 -> 4096) [Tied Weights]
       │
       ▼
Logits / Cross Entropy Loss
```

---

## 📁 Repository Structure

```
├── AuraMind_V3_Enhanced.ipynb  # End-to-end self-contained Colab notebook (tested & verified)
├── AuraMind_V3_Executed.ipynb  # Executed run notebook containing all 3,000-step training outputs
├── AuraMind_V3.ipynb           # Original reference notebook
├── run.py                      # Unified CLI runner (info, train, generate, chat)
├── requirements.txt            # Python dependencies
├── docs/
│   └── FINDINGS.md             # In-depth research & engineering findings report
├── assets/
│   ├── training_curves.png     # Full 3,000-step training loss & perplexity curves
│   └── training_curves_3000_steps.png
└── auramind/                   # Modular Python library
    ├── __init__.py
    ├── config.py               # ModelConfig, TrainingConfig, SamplingConfig
    ├── model.py                # DecoderTransformer, RMSNorm, RoPE, SwiGLU, KV Cache
    ├── tokenizer.py            # Byte-Level BPE tokenizer & special token handlers
    ├── synthetic.py            # Scenario bank, safety filter, quality judge, deduplicator
    ├── dataset.py              # EmpatheticDialogues loader & TokenBlockDataset
    ├── generate.py             # Top-p sampling with repetition penalty & cached inference
    └── train.py                # Training loop, cosine scheduler, and diagnostics
```

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/<your-username>/auramind.git
cd auramind
pip install -r requirements.txt
```

### 2. View Architecture & Parameter Summary
```bash
python run.py info
```

### 3. Run a Quick Smoke Test (CPU or GPU)
```bash
python run.py train --smoke --cpu
```

### 4. Full Training on GPU
```bash
python run.py train --steps 3000 --batch-size 16
```

### 5. Generate Responses via CLI
```bash
python run.py generate "I am feeling really anxious about my exam tomorrow."
```

### 6. Interactive Empathetic Chat
```bash
python run.py chat
```

---

## ☁️ Google Colab

To train or reproduce the 3,000-step run on free Google Colab GPUs (Tesla T4):
1. Open [Google Colab](https://colab.research.google.com).
2. Upload [AuraMind_V3_Enhanced.ipynb](AuraMind_V3_Enhanced.ipynb).
3. Select **Runtime > Change runtime type > T4 GPU**.
4. Click **Runtime > Run all**. All 57 blocks execute sequentially out of the box!

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
