# transformer-from-scratch

> Decoder-only transformer built from primitives. Every component individually, full math exposed.  
> Trains on tiny Shakespeare. Character-level. No HuggingFace. No lightning.

**Paper:** [Attention Is All You Need — Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)

## Architecture
```
Input tokens (B, T)
     │
     ├── Token Embedding   (V, C)
     ├── Position Embedding (T, C)
     │        +
     │   Dropout
     │
     ├── TransformerBlock × N
     │     ├── LayerNorm
     │     ├── MultiHeadAttention (causal)
     │     │     └── ScaledDotProductAttention × H
     │     ├── residual +
     │     ├── LayerNorm
     │     ├── FeedForward
     │     └── residual +
     │
     ├── LayerNorm
     └── Linear → logits (B, T, V)
```
## Components

### `LayerNorm`
```
μ  = mean(x)          over feature dim C
σ² = var(x)           over feature dim C
x̂  = (x - μ) / √(σ² + ε)
out = γ·x̂ + β         γ,β learnable
```
Why not BatchNorm? BN normalises over the batch — breaks at inference with batch_size=1 and is undefined for variable-length sequences.

### `ScaledDotProductAttention`

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
```
**Why √d_k?** Q·Kᵀ has variance ≈ d_k if Q,K ~ N(0,1). Large dot products → softmax saturates → gradient ≈ 0. Dividing by √d_k restores variance to 1.
**Causal mask:** upper-triangular -inf before softmax. Future tokens → weight = 0 after softmax.
```
scores mask (T=4):
  [ 0  -∞  -∞  -∞ ]
  [ 0   0  -∞  -∞ ]
  [ 0   0   0  -∞ ]
  [ 0   0   0   0 ]
```

### `MultiHeadAttention`

```
head_i = Attention(X·Wᵢ_Q,  X·Wᵢ_K,  X·Wᵢ_V)
MultiHead(X) = Concat(head₁,...,headₕ) · W_O
```

H heads run in parallel, each at dimension D = C/H. Implementation fuses H separate (C,D) projections into one (C,C) linear, then reshapes.

---

### `FeedForward`
```
FFN(x) = ReLU(x·W₁ + b₁)·W₂ + b₂
W₁: (C, 4C)    W₂: (4C, C)
```
Applied position-wise (same weights, each token independently). 4× expansion is empirical — Vaswani et al. use d_ff = 4·d_model.

### `TransformerBlock` (pre-norm, GPT-style)
```
x = x + MHA( LayerNorm(x) )    ← attention sub-layer
x = x + FFN( LayerNorm(x) )    ← FFN sub-layer
```
Pre-norm (LN before sublayer) vs original post-norm (LN after residual). Pre-norm: more stable training, direct gradient path through residual.

### `GPT`
- **Weight tying:** `lm_head.weight = tok_emb.weight`. Same matrix for embedding and unembedding. Halves parameters. GPT-2 does this.
- **He init** for linear layers, N(0, 0.02) for embeddings.
- **Autoregressive generation:** temperature + top-k sampling.

## Results (tiny model, 500 steps, CPU)
```
Step   | Train Loss | Val Loss
-------+------------+---------
     1 |   4.1730   |  4.0968     ← ≈ ln(65) — random init
   100 |   3.0374   |  3.9695
   500 |   2.5884   |  4.4207
```

Sample output (500 steps):
```
Tupll this, t thenou fore te bere',
Onireraas, VilllGut...
```
Full 5000-step run (recommended) produces recognisable Shakespeare structure.

## Optimiser: AdamW
```
m_t = β₁·m_{t-1} + (1-β₁)·g          (gradient EMA)
v_t = β₂·v_{t-1} + (1-β₂)·g²         (squared grad EMA)
θ   = θ(1-λ) - lr·m̂_t/(√v̂_t + ε)   (decoupled weight decay)
```
Weight decay applied directly to parameters, not via gradient — correct L2 regularisation.

## Project Structure
```
transformer-from-scratch/
├── model.py          # LayerNorm, SDPA, MHA, FFN, TransformerBlock, GPT
├── data.py           # CharDataset, download Shakespeare, DataLoader
├── train.py          # training loop, AdamW, cosine LR, checkpointing
├── test_shapes.py    # shape tests for every component
└── README.md
```

## steps to follow:
check whether pytorch is installed or not.
# Verify shapes first
python test_shapes.py
# Full training (recommended)
python train.py --max_iters 5000 --n_embd 128 --n_heads 8 --n_layers 6
# Tiny smoke test
python train.py --max_iters 500 --block_size 64 --batch_size 16

**All CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--n_embd` | 64 | Model dimension C |
| `--n_heads` | 4 | Number of attention heads H |
| `--n_layers` | 4 | Number of transformer blocks |
| `--block_size` | 128 | Context window length T |
| `--batch_size` | 32 | Batch size B |
| `--max_iters` | 3000 | Training steps |
| `--lr` | 3e-4 | Peak learning rate |
| `--dropout` | 0.1 | Dropout probability |

## Key Concepts Implemented

| Concept | Location |
|---------|----------|
| Layer Normalisation (Ba et al. 2016) | `model.py → LayerNorm` |
| Scaled dot-product attention | `model.py → ScaledDotProductAttention` |
| Causal (triangular) mask | `model.py → GPT.forward` |
| Multi-head attention | `model.py → MultiHeadAttention` |
| Position-wise FFN 4× expansion | `model.py → FeedForward` |
| Pre-norm residual block | `model.py → TransformerBlock` |
| Learned positional embeddings | `model.py → GPT` |
| Weight tying (embed ↔ unembed) | `model.py → GPT.__init__` |
| He weight init + residual scaling | `model.py → GPT._init_weights` |
| AdamW with decoupled weight decay | `train.py` |
| Cosine LR annealing | `train.py → cosine_lr` |
| Gradient clipping | `train.py → clip_grad_norm_` |
| Top-k + temperature sampling | `model.py → GPT.generate` |
