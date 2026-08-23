"""
Rotary Position Embedding (RoPE) - reference implementation.

Paper: RoFormer: Enhanced Transformer with Rotary Position Embedding
       Su, Lu, Pan, Murtadha, Wen, Liu (2021), arXiv:2104.09864

The library half of this folder. The notebook (`rope.ipynb`) derives the same
math step by step and cross-checks itself against these functions.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. The rotation itself
# ---------------------------------------------------------------------------


def rope_frequencies(head_dim: int, base: float = 10000.0) -> torch.Tensor:
    """theta_i = base ** (-2i/d) for i in [0, d/2). Paper eq. (15)."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    i = torch.arange(0, head_dim, 2, dtype=torch.float32)
    return base ** (-i / head_dim)


def rope_cache(
    head_dim: int,
    seq_len: int,
    base: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape (seq_len, head_dim), duplicated for the half-split layout."""
    theta = rope_frequencies(head_dim, base).to(device=device)
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    angles = torch.outer(pos, theta)                     # (T, d/2)
    angles = torch.cat([angles, angles], dim=-1)         # (T, d) -> matches rotate_half
    return angles.cos().to(dtype), angles.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(x1, x2) -> (-x2, x1) over the last dim split in half."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    offset: int = 0,
) -> torch.Tensor:
    """
    Rotate x by its position. x: (..., T, head_dim); cos/sin: (max_T, head_dim).

    This is the elementwise form of paper eq. (34) - a block-diagonal rotation
    matrix applied without ever materialising the matrix.
    """
    t = x.shape[-2]
    c = cos[offset : offset + t].to(x.dtype)
    s = sin[offset : offset + t].to(x.dtype)
    return x * c + rotate_half(x) * s


# --- the paper's original channel pairing (GPT-J style), for equivalence checks ---


def rope_cache_interleaved(head_dim, seq_len, base=10000.0, device=None, dtype=torch.float32):
    theta = rope_frequencies(head_dim, base).to(device=device)
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    angles = torch.outer(pos, theta).repeat_interleave(2, dim=-1)   # (T, d)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def rotate_pairs(x: torch.Tensor) -> torch.Tensor:
    """(x0, x1, x2, x3, ...) -> (-x1, x0, -x3, x2, ...)"""
    x = x.view(*x.shape[:-1], -1, 2)
    x0, x1 = x[..., 0], x[..., 1]
    return torch.stack([-x1, x0], dim=-1).flatten(-2)


def apply_rope_interleaved(x, cos, sin, offset: int = 0):
    t = x.shape[-2]
    c = cos[offset : offset + t].to(x.dtype)
    s = sin[offset : offset + t].to(x.dtype)
    return x * c + rotate_pairs(x) * s


def rope_matrix(pos: int, head_dim: int, base: float = 10000.0) -> torch.Tensor:
    """
    The dense block-diagonal rotation matrix R^d_{Theta,pos} of paper eq. (15),
    in the paper's interleaved basis. O(d^2) - for teaching / verification only.
    """
    theta = rope_frequencies(head_dim, base)
    r = torch.zeros(head_dim, head_dim)
    for i, th in enumerate(theta):
        a = pos * float(th)
        c, s = math.cos(a), math.sin(a)
        r[2 * i, 2 * i], r[2 * i, 2 * i + 1] = c, -s
        r[2 * i + 1, 2 * i], r[2 * i + 1, 2 * i + 1] = s, c
    return r


# ---------------------------------------------------------------------------
# 2. Baseline position encodings to compare against
# ---------------------------------------------------------------------------


def sinusoidal_pe(seq_len: int, d_model: int, device=None) -> torch.Tensor:
    """Vaswani et al. (2017) additive absolute position encoding."""
    pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    i = torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
    div = torch.exp(-math.log(10000.0) * i / d_model)
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# ---------------------------------------------------------------------------
# 3. Attention + a tiny decoder-only transformer
# ---------------------------------------------------------------------------

POS_MODES = ("none", "learned", "sinusoidal", "rope")


class SelfAttention(nn.Module):
    """Causal multi-head self-attention. pos_mode='rope' rotates q and k."""

    def __init__(self, d_model: int, n_heads: int, pos_mode: str = "rope",
                 max_len: int = 4096, rope_base: float = 10000.0):
        super().__init__()
        if pos_mode not in POS_MODES:
            raise ValueError(f"pos_mode must be one of {POS_MODES}")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads, self.head_dim, self.pos_mode = n_heads, d_model // n_heads, pos_mode
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        if pos_mode == "rope":
            cos, sin = rope_cache(self.head_dim, max_len, base=rope_base)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (b, t, self.n_heads, self.head_dim)
        q, k, v = (z.view(shape).transpose(1, 2) for z in (q, k, v))   # (B, H, T, Dh)

        if self.pos_mode == "rope":
            # Position enters *here*: multiplicatively, on q and k only - never on v,
            # and never added into the residual stream.
            q = apply_rope(q, self.cos, self.sin, offset)
            k = apply_rope(k, self.cos, self.sin, offset)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(b, t, -1))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, pos_mode, max_len, mlp_ratio=4):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, pos_mode, max_len)
        self.n2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )

    def forward(self, x, offset: int = 0):
        x = x + self.attn(self.n1(x), offset)
        return x + self.mlp(self.n2(x))


class TinyTransformer(nn.Module):
    """Decoder-only LM whose *only* difference across runs is pos_mode."""

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=3,
                 pos_mode="rope", max_len=512):
        super().__init__()
        self.pos_mode, self.max_len, self.d_model = pos_mode, max_len, d_model
        self.tok = nn.Embedding(vocab_size, d_model)
        if pos_mode == "learned":
            self.pos = nn.Embedding(max_len, d_model)
        elif pos_mode == "sinusoidal":
            self.register_buffer("pe", sinusoidal_pe(max_len, d_model), persistent=False)
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, pos_mode, max_len) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        t = idx.shape[1]
        x = self.tok(idx)
        if self.pos_mode == "learned":
            # Hard length ceiling: there is no defined embedding past max_len.
            # We clamp to the last row so the forward pass at least completes.
            pos = torch.arange(min(t, self.max_len), device=idx.device)
            emb = self.pos(pos)
            if t > self.max_len:
                emb = torch.cat([emb, emb[-1:].expand(t - self.max_len, -1)], dim=0)
            x = x + emb
        elif self.pos_mode == "sinusoidal":
            # Vaswani et al. scale the token embeddings by sqrt(d_model) before adding
            # the PE. Without it the unit-amplitude sinusoids drown out the (std=0.02)
            # token vectors and the model learns nothing - an unfair strawman baseline.
            x = x * math.sqrt(self.d_model) + self.pe[:t]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))


# ---------------------------------------------------------------------------
# 4. Linear attention with RoPE (paper section 3.3)
# ---------------------------------------------------------------------------


def linear_attention_rope(q, k, v, cos=None, sin=None, eps: float = 1e-6):
    """
    O(T) linear attention with phi(x) = elu(x) + 1 (Katharopoulos et al., 2020).

    Additive relative-position biases cannot be used here: they need the explicit
    T x T logit matrix, which linear attention never forms. A rotation can, because
    it acts on q and k *before* the kernel feature map.
    """
    q, k = F.elu(q) + 1, F.elu(k) + 1
    if cos is not None:
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    kv = torch.einsum("bhtd,bhte->bhde", k, v)
    z = k.sum(dim=-2, keepdim=True)
    num = torch.einsum("bhtd,bhde->bhte", q, kv)
    den = torch.einsum("bhtd,bhkd->bhtk", q, z).clamp_min(eps)
    return num / den


# ---------------------------------------------------------------------------
# 5. The paper's long-term-decay bound (section 3.4.3)
# ---------------------------------------------------------------------------


def relative_upper_bound(head_dim: int, max_distance: int, base: float = 10000.0):
    """
    (1/(d/2)) * sum_i |sum_{j<=i} exp(i * n * theta_j)| as a function of the relative
    distance n - the quantity plotted in figure 2 of the paper.
    """
    theta = rope_frequencies(head_dim, base)                        # (d/2,)
    n = torch.arange(max_distance, dtype=torch.float32)             # (N,)
    phase = torch.outer(n, theta)                                   # (N, d/2)
    partial = torch.cumsum(torch.polar(torch.ones_like(phase), phase), dim=-1)
    return n, partial.abs().mean(dim=-1)
