# Research Paper Implementations

From-scratch PyTorch implementations of ML papers, one folder per paper, each with a
notebook that walks through *the problem*, *the fix the paper proposes*, and *the code*.

| Folder | Paper |
| --- | --- |
| `Attention_all_you_need/` | Vaswani et al., 2017 — Attention Is All You Need |
| `RoPE/` | Su et al., 2021 — RoFormer: Enhanced Transformer with Rotary Position Embedding |

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
