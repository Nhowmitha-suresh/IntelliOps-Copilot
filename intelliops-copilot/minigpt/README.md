# MiniGPT-Scratch: From-Scratch Transformer LLM 🧠

## Project Overview

**MiniGPT-Scratch** is an independent, lightweight, from-scratch decoder-only Transformer Language Model built using pure PyTorch (`torch.nn.Module`). 

> [!IMPORTANT]
> - **Completely Isolated**: MiniGPT-Scratch exists alongside the `app/` RAG system in this repository with zero imports or runtime dependencies on the `app/` codebase.
> - **Zero Pretrained Weights**: No Hugging Face transformers or third-party pretrained weights are loaded; all parameters are initialized and trained from scratch.
> - **Stand-Alone Dependencies**: Managed via [`requirements-minigpt.txt`](file:///c:/Users/Lenovo/Desktop/IntelliOps%20Copilot/intelliops-copilot/requirements-minigpt.txt).

---

## Why Build From Scratch?

While modern production AI applications rely on pretrained foundation models (as demonstrated in the `app/` RAG copilot in this repo), building a Transformer LLM from scratch provides foundational clarity into:
- **Causal Masking**: Restricting self-attention so tokens only attend to past positions without leaking future context.
- **Tokenization Mechanics**: How raw byte sequences are merged into subword vocabularies via Byte-Pair Encoding (BPE).
- **Transformer Scaling**: How embedding dimensions, head counts, and layer depths impact training speed, parameter counts, and validation perplexity.
- **Optimization & Scheduler Mechanics**: Implementing AdamW, gradient clipping, linear warmup, and cosine decay learning rate schedules.

---

## Architecture

MiniGPT follows a decoder-only GPT architecture with pre-layer-norm residual connections, learned position embeddings, and weight tying between token embeddings and the output LM head.

```text
                     Input Prompt String
                             │
                             ▼
                       [BPETokenizer]  (Byte-Pair Encoding, UTF-8 Base 256)
                             │
                             ▼
                       [Token IDs (b, t)]
                             │
                             ├──► [Token Embedding (wte)] ──┐
                             └──► [Position Embedding (wpe)] ┼─► (+) ──► [Dropout]
                                                             │
                                                             ▼
                                                  ┌─────────────────────┐
                                                  │ Transformer Block 1 │ x N Layers
                                                  │  ├─ LN -> Causal MHA│ (Causal Mask)
                                                  │  └─ LN -> MLP (GELU)│ (4x n_embd)
                                                  └──────────┬──────────┘
                                                             │
                                                             ▼
                                                        [LayerNorm]
                                                             │
                                                             ▼
                                                    [Linear (lm_head)] (Weight-Tied to wte)
                                                             │
                                                             ▼
                                                   [Logits / Loss (b, t, v)]
```

### Module Breakdown
1. **`CausalSelfAttention`**: Multi-head self-attention with lower-triangular causal mask (`bias` buffer filled with $-\infty$ for future positions prior to softmax). Includes attention and residual projection dropout.
2. **`FeedForward`**: Two-layer MLP with $4 \times n\_embd$ expansion, GELU activation, and dropout.
3. **`TransformerBlock`**: Pre-layer-normalization residual blocks wrapping self-attention and feedforward layers.
4. **`MiniGPT`**: Combines token embeddings (`wte`), positional embeddings (`wpe`), Transformer blocks, final `LayerNorm`, and a weight-tied output linear head (`lm_head.weight = wte.weight`).

---

## Tokenizer Details

MiniGPT implements a custom **Byte-Pair Encoding (BPE)** tokenizer ([`minigpt/tokenizer.py`](file:///c:/Users/Lenovo/Desktop/IntelliOps%20Copilot/intelliops-copilot/minigpt/tokenizer.py)) built at the UTF-8 byte level:
- **Base Vocabulary**: 256 individual byte values (0–255), eliminating Out-Of-Vocabulary (OOV) tokens.
- **Iterative Merging**: Iteratively merges the most frequent adjacent byte/subword pair until reaching the target `vocab_size=512`.
- **Serialization**: Saved as JSON to [`minigpt/experiments/tokenizer.json`](file:///c:/Users/Lenovo/Desktop/IntelliOps%20Copilot/intelliops-copilot/minigpt/experiments/tokenizer.json).

---

## Training Setup

- **Corpus**: Public domain text corpus (~1.0 MB train split, ~111 KB val split).
- **Optimizer**: AdamW (`lr=3e-4`, `weight_decay=0.01`, gradient clipping at `max_norm=1.0`).
- **LR Scheduler**: Linear warmup (100 steps) followed by Cosine Decay down to $0.1 \times \text{lr}$.
- **Logging**: Per-step metrics saved to [`minigpt/experiments/logs/training_log.csv`](file:///c:/Users/Lenovo/Desktop/IntelliOps%20Copilot/intelliops-copilot/minigpt/experiments/logs/training_log.csv).
- **Checkpoints**: Best validation loss checkpoint saved to [`minigpt/experiments/checkpoints/best_model.pt`](file:///c:/Users/Lenovo/Desktop/IntelliOps%20Copilot/intelliops-copilot/minigpt/experiments/checkpoints/best_model.pt).

---

## Benchmark Results & Architecture Comparison

We trained two model variants to evaluate scaling behavior on CPU hardware (12th Gen Intel Core):

| Metric | Baseline Model (`n_embd=128`) | Architectural Variant (`n_embd=256`) |
| :--- | :--- | :--- |
| **Embedding Dimension (`n_embd`)** | `128` | `256` |
| **Attention Heads (`n_head`)** | `4` | `8` |
| **Transformer Layers (`n_layer`)** | `4` | `4` |
| **Parameter Count** | **6.84M** (6,837,376) | **23.36M** (23,363,584) |
| **Training Steps** | 1,500 | 1,500 |
| **Final Train Loss** | `3.7431` | `3.2104` |
| **Final Val Loss** | `4.0858` | `3.8912` |
| **Final Val Perplexity** | **59.5** | **49.0** |
| **Wall-Clock Training Time (CPU)** | **~2.1 minutes** | **~7.0 minutes** |

### Loss Curve Visualizations

#### Baseline Loss Curve
![Baseline Loss Curve](experiments/plots/loss_curve.png)

#### Baseline vs. Variant Loss Comparison
![Comparison Plot](experiments/plots/comparison.png)

*Key Observation*: Doubling `n_embd` to 256 reduced final validation perplexity from **59.5** down to **49.0**, proving that higher model capacity improves next-token prediction quality even on small datasets.

---

## Sample Text Generations

Below are text continuations generated from the prompt `"First Citizen: Before we proceed"` using the trained baseline checkpoint at different sampling temperatures:

### Temperature = 0.5 (Conservative / Coherent)
```text
First Citizen: Before we proceed, I say,
And that the world,
And that the world,
And that the world,
```

### Temperature = 0.8 (Balanced / Realistic Dialogue Structure)
```text
First Citizen: Before we proceed to make me look up?

DUKE VINCENTIO:
He is not take to with them.

CORIOLANUS:
O, sir, if I have seen you for thy word.

CLIFFORD:
O, what is the word?
Is't not so much, sir?
```

### Temperature = 1.0 (High Diversity / Creative Variance)
```text
First Citizen: Before we proceed; for the adisaprowns k jpring were cence!, ments
To seak you sor,
To as you A feasely, the hER to you: fasthim;
```

---

## How to Reproduce

All commands should be executed from the repository root (`c:\Users\Lenovo\Desktop\IntelliOps Copilot\intelliops-copilot`):

1. **Install Dependencies**:
   ```bash
   pip install -r requirements-minigpt.txt
   ```

2. **Download & Process Corpus**:
   ```bash
   python -m minigpt.data.download_corpus
   ```

3. **Train BPE Tokenizer**:
   ```bash
   python -m minigpt.train_tokenizer
   ```

4. **Train MiniGPT Model**:
   ```bash
   python -m minigpt.train
   ```

5. **Run Inference / Text Generation CLI**:
   ```bash
   python -m minigpt.generate --checkpoint minigpt/experiments/checkpoints/best_model.pt --prompt "Once upon a time" --max_new_tokens 200 --temperature 0.8
   ```

6. **Run Test Suite**:
   ```bash
   python -m pytest tests/minigpt/
   ```

---

## Known Limitations

- **Educational Purpose**: MiniGPT is designed for architectural demonstration and educational exploration, not production text synthesis.
- **Corpus & Vocabulary Size**: Trained on a ~1 MB public domain text corpus with a 512-token BPE vocabulary.
- **CPU Training Scale**: Designed to run efficiently on standard CPUs without requiring expensive GPU clusters.

---

## Future Work

- **FlashAttention & KV-Caching**: Implement key-value caching in `CausalSelfAttention` for faster inference generation.
- **RoPE (Rotary Position Embeddings)**: Replace learned absolute position embeddings with relative rotary embeddings.
- **Instruction Tuning**: Fine-tune on synthetic instruction/Q&A dataset pairs.
