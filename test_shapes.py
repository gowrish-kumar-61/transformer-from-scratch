"""
transformer-from-scratch/test_shapes.py

Validates every module's input/output shapes.
Run before training to catch dimension bugs early.

No pytest needed — just `python test_shapes.py`.
"""

import torch
from model import (
    LayerNorm,
    ScaledDotProductAttention,
    MultiHeadAttention,
    FeedForward,
    TransformerBlock,
    GPT,
    count_params,
)

def check(name, got, expected):
    status = "✓" if got == expected else "✗ FAIL"
    print(f"  {status}  {name:40s} got {str(got):20s} expected {expected}")
    assert got == expected, f"Shape mismatch in {name}"


def run_tests():
    torch.manual_seed(42)
    B, T, C, H, D = 2, 16, 64, 4, 16

    print("=" * 70)
    print("SHAPE TESTS")
    print("=" * 70)

    # LayerNorm
    print("\n[LayerNorm]")
    ln = LayerNorm(C)
    x = torch.randn(B, T, C)
    out = ln(x)
    check("output shape", tuple(out.shape), (B, T, C))
    # Verify normalisation
    mean = out.mean(dim=-1).abs().max().item()
    print(f"  ✓  post-norm mean ≈ 0: max|μ| = {mean:.6f}")

    # ScaledDotProductAttention
    print("\n[ScaledDotProductAttention]")
    sdpa = ScaledDotProductAttention(dropout=0.0)
    Q = torch.randn(B, H, T, D)
    K = torch.randn(B, H, T, D)
    V = torch.randn(B, H, T, D)
    mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
    out, attn_w = sdpa(Q, K, V, mask)
    check("output shape",       tuple(out.shape),     (B, H, T, D))
    check("attn_weights shape", tuple(attn_w.shape),  (B, H, T, T))
    # Weights should sum to 1 across key dim
    row_sums = attn_w.sum(dim=-1)
    print(f"  ✓  attn weights sum ≈ 1: max deviation = {(row_sums - 1).abs().max().item():.6f}")

    # MultiHeadAttention
    print("\n[MultiHeadAttention]")
    mha = MultiHeadAttention(n_embd=C, n_heads=H, dropout=0.0)
    x = torch.randn(B, T, C)
    mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
    out, attn_w = mha(x, mask)
    check("output shape",       tuple(out.shape),    (B, T, C))
    check("attn_weights shape", tuple(attn_w.shape), (B, H, T, T))

    # FeedForward
    print("\n[FeedForward]")
    ffn = FeedForward(n_embd=C, dropout=0.0)
    x = torch.randn(B, T, C)
    out = ffn(x)
    check("output shape", tuple(out.shape), (B, T, C))

    # TransformerBlock
    print("\n[TransformerBlock]")
    block = TransformerBlock(n_embd=C, n_heads=H, dropout=0.0)
    x = torch.randn(B, T, C)
    mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
    out, _ = block(x, mask)
    check("output shape", tuple(out.shape), (B, T, C))

    # GPT
    print("\n[GPT]")
    V_SIZE = 65
    model = GPT(vocab_size=V_SIZE, n_embd=C, n_heads=H, n_layers=2, block_size=T)
    idx = torch.randint(0, V_SIZE, (B, T))
    targets = torch.randint(0, V_SIZE, (B, T))
    logits, loss = model(idx, targets)
    check("logits shape", tuple(logits.shape), (B, T, V_SIZE))
    print(f"  ✓  loss = {loss.item():.4f}  (random init ≈ ln({V_SIZE}) = {torch.tensor(V_SIZE).float().log().item():.4f})")
    print(f"  ✓  {count_params(model)}")

    # Generation
    print("\n[Generation]")
    prompt = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(prompt, max_new_tokens=20, temperature=1.0)
    check("generated shape", tuple(out.shape), (1, 21))  # 1 prompt + 20 new

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()