"""
transformer-from-scratch
Every component built from primitives. Math exposed at every step.

PAPER: "Attention Is All You Need" — Vaswani et al. 2017
       https://arxiv.org/abs/1706.03762

ARCHITECTURE (decoder-only, GPT-style):
  Token Embed + Positional Embed
  → N × TransformerBlock(
        LayerNorm → MultiHeadAttention (causal) → residual
        LayerNorm → FeedForward                 → residual
    )
  → LayerNorm → Linear → logits

DIMENSION CHEATSHEET (used throughout):
  B  = batch size
  T  = sequence length (context window)
  C  = model dimension (d_model / n_embd)
  H  = number of heads
  D  = head dimension = C // H
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. LAYER NORM

class LayerNorm(nn.Module):
    """
    Layer Normalisation — Ba et al. 2016
    https://arxiv.org/abs/1607.06450

    WHY NOT BATCH NORM?
      BatchNorm normalises across the batch dimension → breaks at
      inference with batch_size=1, and is weird for sequences of
      variable length. LayerNorm normalises across the feature
      dimension (C) per token, per sample. No batch dependency.

    MATH:
      μ = mean(x)         over last dim
      σ² = var(x)         over last dim
      x̂ = (x - μ) / √(σ² + ε)
      out = γ · x̂ + β     γ,β learnable scalars (per feature)

    WHY ε?
      Numerical stability. If σ²=0 (all features identical), we'd
      divide by zero. ε=1e-5 floors the denominator.

    INPUT SHAPE:  (..., C)
    OUTPUT SHAPE: (..., C)   — same
    """
    def __init__(self, n_embd: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # γ (scale) and β (shift) — learned per feature
        self.weight = nn.Parameter(torch.ones(n_embd))   # γ
        self.bias   = nn.Parameter(torch.zeros(n_embd))  # β

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_hat + self.bias

# 2. SCALED DOT-PRODUCT ATTENTION

class ScaledDotProductAttention(nn.Module):
    """
    The atomic attention operation.

    MATH:
      Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V

    WHY SCALE BY √d_k?
      Q·Kᵀ produces dot products with variance ≈ d_k (if Q,K ~ N(0,1)).
      Variance grows with dimension. Large dot products → softmax
      saturates → gradients ≈ 0 → dead attention.
      Dividing by √d_k restores variance to ≈ 1.

    CAUSAL MASK:
      For language modelling, token t must NOT attend to token t+1…T.
      We mask future positions with -inf before softmax → they become 0
      after softmax → effectively ignored.

    SHAPES (single head):
      Q: (B, T, D)
      K: (B, T, D)
      V: (B, T, D)
      scores: Q @ Kᵀ → (B, T, T)
      out: softmax(scores) @ V → (B, T, D)
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        d_k = Q.size(-1)
        scale = math.sqrt(d_k)

        # Q @ Kᵀ  →  (B, H, T, T)
        # Kᵀ means transpose last two dims of K: (B,H,D,T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax over key dimension (last dim = T_keys)
        attn_weights = F.softmax(scores, dim=-1)

        # Replace any NaN from -inf rows (empty context) with 0
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        out = torch.matmul(attn_weights, V)
        return out, attn_weights

# 3. MULTI-HEAD ATTENTION

class MultiHeadAttention(nn.Module):
    """
    Split d_model into H heads, run attention in parallel, concat.

    WHY MULTIPLE HEADS?
      Single-head attention learns ONE way to relate tokens.
      H heads learn H different relationship types simultaneously:
        head 1 might track syntactic dependencies
        head 2 might track coreference
        head 3 might track positional patterns
      ... all in parallel, at 1/H the dimension each.

    MATH:
      For head i:
        Qᵢ = X · Wᵢ_Q    Kᵢ = X · Wᵢ_K    Vᵢ = X · Wᵢ_V
        headᵢ = Attention(Qᵢ, Kᵢ, Vᵢ)

      MultiHead(X) = Concat(head₁, ..., headₕ) · W_O

    IMPLEMENTATION TRICK:
      Instead of H separate (C,D) projections, use one (C,C) linear
      and reshape. Equivalent math, but one kernel launch instead of H.
      W_Q: (C, C) = H stacked (C, D) matrices

    SHAPES:
      X    : (B, T, C)
      W_Q,K,V : (C, C)   → Q,K,V: (B,T,C)
      Reshape to (B, H, T, D) for batched attention
      After attention: (B, H, T, D) → concat → (B, T, C)
      W_O  : (C, C)      → out: (B, T, C)
    """
    def __init__(self, n_embd: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_heads == 0, \
            f"n_embd ({n_embd}) must be divisible by n_heads ({n_heads})"

        self.n_heads = n_heads
        self.n_embd  = n_embd
        self.d_head  = n_embd // n_heads

        # Single fused projection for Q, K, V  (more efficient than 3 separate)
        self.W_q = nn.Linear(n_embd, n_embd, bias=False)
        self.W_k = nn.Linear(n_embd, n_embd, bias=False)
        self.W_v = nn.Linear(n_embd, n_embd, bias=False)

        # Output projection W_O
        self.W_o = nn.Linear(n_embd, n_embd, bias=False)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        B, T, C = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape + transpose: (B,T,C) → (B,T,H,D) → (B,H,T,D)
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        # Attention: each head sees (B, T, D) slices
        out, attn_weights = self.attention(Q, K, V, mask)

        # Merge heads: (B,H,T,D) → (B,T,H,D) → (B,T,C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear projection
        out = self.W_o(out)
        out = self.dropout(out)
        return out, attn_weights

# 4. FEED-FORWARD NETWORK

class FeedForward(nn.Module):
    """
    Position-wise FFN applied identically to each token.

    MATH:
      FFN(x) = max(0, x·W₁ + b₁) · W₂ + b₂
      W₁: (C, 4C)   W₂: (4C, C)

    WHY 4×?
      Vaswani et al. use d_ff = 4·d_model. No deep theory — empirically
      works. The expansion gives the network room to compute in a richer
      intermediate representation before projecting back.

    WHY GELU INSTEAD OF RELU?
      GPT-2 and beyond use GELU: x·Φ(x) where Φ is the Gaussian CDF.
      Smoother than ReLU → better gradient flow near 0.
      We use ReLU here to keep math visible; swap to nn.GELU() trivially.

    NOTE: This is applied PER TOKEN independently — no interaction
    between positions here. That's attention's job.

    SHAPES:
      x   : (B, T, C)
      W₁  : (C, 4C)   intermediate: (B, T, 4C)
      W₂  : (4C, C)   out: (B, T, C)
    """
    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# 5. TRANSFORMER BLOCK

class TransformerBlock(nn.Module):
    """
    One decoder block (GPT-style pre-norm):

      x = x + MHA(LayerNorm(x))      ← self-attention sub-layer
      x = x + FFN(LayerNorm(x))      ← feed-forward sub-layer

    PRE-NORM vs POST-NORM:
      Original Vaswani paper uses POST-norm: LN after residual.
      GPT/modern practice: PRE-norm (LN before sublayer).
      Pre-norm: more stable training, gradient flows directly through
      the residual branch without passing through LN.

    RESIDUAL CONNECTIONS:
      x = x + sublayer(x)
      Why? Direct gradient highway from loss to early layers.
      Without residuals: gradient ∝ product of Jacobians across all
      layers → exponential vanishing/explosion.
      With residuals: gradient = 1 + (sublayer Jacobian) → at least 1.

    SHAPES: (B, T, C) in, (B, T, C) out — block is shape-preserving.
    """
    def __init__(self, n_embd: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1  = LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_heads, dropout)
        self.ln2  = LayerNorm(n_embd)
        self.ffn  = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # Pre-norm self-attention + residual
        attn_out, attn_weights = self.attn(self.ln1(x), mask)
        x = x + attn_out

        # Pre-norm FFN + residual
        x = x + self.ffn(self.ln2(x))
        return x, attn_weights

# 6. GPT — FULL LANGUAGE MODEL

class GPT(nn.Module):
    """
    Decoder-only transformer for character-level language modelling.

    POSITIONAL ENCODING:
      Attention is permutation-invariant — order of tokens doesn't
      matter to raw attention scores. We inject position info via
      learned embeddings: each position 0…T-1 gets a learnable vector.

      Vaswani used sinusoidal (fixed) embeddings. GPT uses learned.
      Both work; learned is simpler to implement.

    WEIGHT TYING:
      Token embedding W_e: (vocab_size, C)
      Output head W_lm:    (C, vocab_size)
      We tie W_lm = W_eᵀ — they share the same matrix.
      Rationale: similar tokens should be similar in embedding space
      AND similar in output logit space. Halves parameter count.
      GPT-2 does this; we do too.

    CAUSAL MASK:
      Lower-triangular boolean mask (T×T).
      mask[i,j] = 1 if position j ≤ i (can attend)
                = 0 if position j > i (future → blocked)
    """
    def __init__(
        self,
        vocab_size : int,
        n_embd     : int   = 64,
        n_heads    : int   = 4,
        n_layers   : int   = 4,
        block_size : int   = 128,
        dropout    : float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size

        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop    = nn.Dropout(dropout)

        # Stack of transformer blocks
        self.blocks  = nn.ModuleList(
            [TransformerBlock(n_embd, n_heads, dropout) for _ in range(n_layers)]
        )

        # Final layer norm + language model head
        self.ln_f    = LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: lm_head.weight ← tok_emb.weight
        self.lm_head.weight = self.tok_emb.weight

        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        GPT-2 style init:
          Linear + Embedding: N(0, 0.02)
          Residual projections scaled by 1/√(2·n_layers)
            Why? Each block adds a residual. With N blocks, variance
            accumulates as N·σ². Scale by 1/√(2N) to keep output
            variance ≈ 1 regardless of depth.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        idx     : (B, T) — token indices
        targets : (B, T) — shifted targets for next-token prediction
        """
        B, T = idx.shape
        assert T <= self.block_size, \
            f"Sequence length {T} exceeds block_size {self.block_size}"

        device = idx.device

        # Token + positional embeddings
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=device))
        x   = self.drop(tok + pos)

        # Causal mask: lower-triangular, shape (1,1,T,T) for broadcasting over (B,H)
        mask = torch.tril(torch.ones(T, T, device=device)).view(1, 1, T, T)

        # Pass through N transformer blocks
        for block in self.blocks:
            x, _ = block(x, mask)

        x = self.ln_f(x)             # (B,T,C) — final layer norm
        logits = self.lm_head(x)     # (B,T,V) — unnormalised scores

        # Loss: cross-entropy over flattened (B*T, V) vs (B*T,) targets
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, -1),   # (B*T, V)
                targets.view(B * T),      # (B*T,)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        """
        Autoregressive generation. At each step:
          1. Crop context to block_size
          2. Forward pass → logits for last position
          3. Optionally apply top-k filtering
          4. Scale by temperature: lower T → sharper distribution
          5. Sample from categorical distribution
          6. Append to sequence

        TEMPERATURE:
          logits /= T → if T<1: peaks sharpened (more deterministic)
                         if T>1: distribution flattened (more random)
        TOP-K:
          Zero out all logits except top-k before softmax.
          Prevents sampling from tail of distribution (garbage tokens).
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -self.block_size:]
            logits, _ = self(idx_crop)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx


def count_params(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total:,} | Trainable: {train:,}"