# Research Paper Implementations

From-scratch PyTorch implementations of ML papers, one folder per paper, each with a
notebook that walks through *the problem*, *the fix the paper proposes*, and *the code*.

| Folder | Paper |
| --- | --- |
| `Attention_all_you_need/` | Vaswani et al., 2017 — Attention Is All You Need |
| `RoPE/` | Su et al., 2021 — RoFormer: Enhanced Transformer with Rotary Position Embedding |

## RoPE

`RoPE/rope.ipynb` is written to be **taught out loud** — 22 numbered steps following this
roadmap, so the audience always knows where in the story they are:

```
PROBLEM → OLD SOLUTIONS → ROPE INTUITION → ROTATION MATH
        → PAPER EQUATION → NUMERICAL EXAMPLE → PYTORCH CODE
```

| Steps | What they cover |
| --- | --- |
| 1–2 | why attention needs position; Q, K, V in one minute |
| 3–4 | absolute positional encoding, and why address ≠ distance |
| 5–6 | relative position, and why using it directly breaks (N×N, and linear attention) |
| **7** | **the core idea: position → ROTATE, not ADD** |
| 8–11 | 2D rotation, the rotation matrix, position as the angle, Q and K rotating separately |
| **12–13** | **`7θ − 2θ = 5θ`; pair (2,4) and pair (10,12) agree** |
| 14–15 | the paper equation `Qᵀ R₍ₙ₋ₘ₎ K`, then a full numerical example |
| 16–19 | many dimensions, different frequencies, norm preservation, `rotate_half` |
| 20–22 | the five code demos, practical details, final summary and limits |

The table at the top of the notebook gives, per step, the line the audience must leave
with and roughly how long it takes. **25-minute version:** steps 1 → 7 → 8 → 10 → 12 → 13
→ 22. **Five minutes:** step 1's cell, then step 12's cell.

Two extras follow step 22: evidence (four models trained here, only the position encoding
differs) and Appendix A (every paper equation transcribed and checked).

| File | What it is |
| --- | --- |
| `RoPE/rope.ipynb` | the 22-step notebook, outputs pre-populated |
| `RoPE/teaching-diagrams.md` | a Mermaid diagram per step, validated to import into Excalidraw as editable shapes |
| `RoPE/rope.png` | the whiteboard plan |
| `RoPE/rope.py` | the library: rotation, attention, tiny transformer, linear attention |
| `RoPE/task.py` | the position-critical copy task + training loop |
| `RoPE/test_rope.py` | one assertion per property the paper claims (`make test`) |

## Setup

One shared `uv` environment lives at the root of this folder and is used by every paper.

```bash
cd Research_Paper
make setup      # installs uv (if missing), creates .venv, installs deps, registers a Jupyter kernel
make gpu        # optional: replace CPU torch with the CUDA build (RTX / any NVIDIA card)
make lab        # open JupyterLab and pick the "Python (research-papers)" kernel
```

Useful targets (`make help` lists them all):

| Target | What it does |
| --- | --- |
| `make setup` | full bootstrap: venv + deps + kernel |
| `make install` | `uv sync --all-groups` |
| `make add PKG=wandb` | add a dependency and update `uv.lock` |
| `make gpu` / `make cpu` | swap the torch build |
| `make info` | python / torch / CUDA / device summary |
| `make run NB=RoPE/rope.ipynb` | execute a notebook end-to-end |
| `make fmt` / `make lint` / `make test` | ruff format, ruff check, pytest |
| `make clean` / `make clean-venv` | drop caches / drop the venv |

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock` — torch, numpy,
pandas, matplotlib/seaborn, scikit-learn, scipy, einops, transformers/tokenizers/datasets,
jupyterlab, and dev tooling (ruff, pytest, nbstripout).
