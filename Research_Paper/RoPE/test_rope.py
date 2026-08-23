"""Tests that pin down the properties the RoPE paper actually claims."""

import torch

from rope import (
    SelfAttention,
    TinyTransformer,
    apply_rope,
    apply_rope_interleaved,
    linear_attention_rope,
    relative_upper_bound,
    rope_cache,
    rope_cache_interleaved,
    rope_matrix,
    sinusoidal_pe,
)

D = 32          # head dim
T = 64          # sequence length
TOL = 1e-4


def _qk(seed=0, b=2, h=3, t=T, d=D):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, h, t, d, generator=g)
    k = torch.randn(b, h, t, d, generator=g)
    return q, k


def test_rotation_is_orthogonal_norm_preserved():
    """A rotation cannot change the length of q or k - so it cannot rescale attention."""
    q, _ = _qk()
    cos, sin = rope_cache(D, T)
    assert torch.allclose(apply_rope(q, cos, sin).norm(dim=-1), q.norm(dim=-1), atol=TOL)


def test_dense_matrix_matches_elementwise_form():
    """apply_rope_interleaved == multiplying by the block-diagonal R of eq. (15)."""
    q, _ = _qk(t=8, b=1, h=1)
    cos, sin = rope_cache_interleaved(D, 8)
    fast = apply_rope_interleaved(q, cos, sin)[0, 0]
    slow = torch.stack([rope_matrix(m, D) @ q[0, 0, m] for m in range(8)])
    assert torch.allclose(fast, slow, atol=TOL)


def test_logits_depend_only_on_relative_distance():
    """The central claim: <f_q(x,m), f_k(x,n)> = g(x_m, x_n, m-n)."""
    q, k = _qk()
    cos, sin = rope_cache(D, 4 * T)
    base = apply_rope(q, cos, sin) @ apply_rope(k, cos, sin).transpose(-1, -2)
    for shift in (1, 7, 33, 100):
        shifted = (
            apply_rope(q, cos, sin, offset=shift)
            @ apply_rope(k, cos, sin, offset=shift).transpose(-1, -2)
        )
        assert torch.allclose(base, shifted, atol=1e-3), f"shift {shift} changed the logits"


def test_additive_absolute_pe_is_not_shift_invariant():
    """The baseline fails the same test - which is why RoPE exists."""
    q, k = _qk(t=T, b=1, h=1, d=D)
    pe = sinusoidal_pe(4 * T, D)
    base = (q[0, 0] + pe[:T]) @ (k[0, 0] + pe[:T]).T
    shifted = (q[0, 0] + pe[10 : 10 + T]) @ (k[0, 0] + pe[10 : 10 + T]).T
    assert not torch.allclose(base, shifted, atol=1e-2)


def test_both_channel_pairings_share_the_relative_property():
    """Half-split (HF/NeoX) and interleaved (paper/GPT-J) are both valid bases."""
    q, k = _qk()
    for cache, apply in (
        (rope_cache, apply_rope),
        (rope_cache_interleaved, apply_rope_interleaved),
    ):
        cos, sin = cache(D, 4 * T)
        a = apply(q, cos, sin) @ apply(k, cos, sin).transpose(-1, -2)
        b = apply(q, cos, sin, offset=21) @ apply(k, cos, sin, offset=21).transpose(-1, -2)
        assert torch.allclose(a, b, atol=1e-3)


def test_zero_position_is_identity():
    q, _ = _qk()
    cos, sin = rope_cache(D, T)
    assert torch.allclose(apply_rope(q[..., :1, :], cos, sin), q[..., :1, :], atol=TOL)


def test_long_term_decay_bound_decreases():
    """Paper section 3.4.3 / figure 2: the bound trends down with relative distance."""
    _, bound = relative_upper_bound(D, 512)
    assert bound[0] > bound[64] > bound[256]


def test_attention_module_is_translation_equivariant():
    """Padding-free check: the same content at a later offset gives the same output."""
    attn = SelfAttention(64, 4, pos_mode="rope", max_len=512).eval()
    x = torch.randn(2, 16, 64)
    with torch.no_grad():
        a, b = attn(x, offset=0), attn(x, offset=37)
    assert torch.allclose(a, b, atol=1e-4)


def test_rope_composes_with_linear_attention():
    """Section 3.3: a rotation survives the kernel trick; an additive T x T bias cannot."""
    q, k = _qk(t=16, b=1, h=1)
    v = torch.randn(1, 1, 16, D)
    cos, sin = rope_cache(D, 16)
    out = linear_attention_rope(q, k, v, cos, sin)
    assert out.shape == v.shape and torch.isfinite(out).all()


def test_model_runs_in_every_pos_mode():
    for mode in ("none", "learned", "sinusoidal", "rope"):
        m = TinyTransformer(22, d_model=64, n_heads=4, n_layers=2,
                            pos_mode=mode, max_len=128).eval()
        with torch.no_grad():
            out = m(torch.randint(0, 22, (2, 40)))
        assert out.shape == (2, 40, 22)


def test_rope_has_no_position_parameters():
    """RoPE adds zero learnable parameters over the no-position baseline."""
    kw = dict(vocab_size=22, d_model=64, n_heads=4, n_layers=2, max_len=128)
    n = lambda mode: sum(p.numel() for p in TinyTransformer(pos_mode=mode, **kw).parameters())
    assert n("rope") == n("none") == n("sinusoidal")
    assert n("learned") > n("rope")
