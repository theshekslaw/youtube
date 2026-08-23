# Research paper implementations - environment management via uv
#
# Quick start:
#   make setup        # install uv (if missing) + create .venv + install all deps
#   make gpu          # (optional) swap CPU torch for the CUDA build
#   make lab          # open JupyterLab
#
# Everything lives in ./.venv and is driven by ./pyproject.toml + ./uv.lock

SHELL := /bin/bash

ifeq ($(OS),Windows_NT)
VENV_BIN := .venv/Scripts
EXE      := .exe
else
VENV_BIN := .venv/bin
EXE      :=
endif

PY     := $(VENV_BIN)/python$(EXE)
UV     := uv
KERNEL := research-papers
CUDA   := cu130      # CUDA channel on download.pytorch.org; must carry a wheel for
                     # this python/platform - check https://download.pytorch.org/whl/

.DEFAULT_GOAL := help

## ---------------------------------------------------------------- environment

.PHONY: help
help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

.PHONY: uv
uv: ## install the uv package manager if it is not on PATH
	@command -v $(UV) >/dev/null 2>&1 && echo "uv already installed: $$($(UV) --version)" || { \
		echo "installing uv..."; \
		if [ "$(OS)" = "Windows_NT" ]; then \
			powershell -ExecutionPolicy ByPass -NoProfile -c "irm https://astral.sh/uv/install.ps1 | iex"; \
		else \
			curl -LsSf https://astral.sh/uv/install.sh | sh; \
		fi; \
		echo "uv installed - reopen your shell if 'uv' is still not found"; \
	}

.PHONY: venv
venv: uv ## create ./.venv with the pinned python
	$(UV) venv --python 3.12
	@echo "venv ready -> $(VENV_BIN)"

.PHONY: install
install: uv ## resolve + install all project dependencies into ./.venv
	$(UV) sync --all-groups
	@echo "deps installed"

.PHONY: setup
setup: venv install kernel ## one-shot: uv + venv + deps + jupyter kernel
	@$(PY) -c "import torch, numpy, pandas; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available()); print('numpy', numpy.__version__, '| pandas', pandas.__version__)"

.PHONY: gpu
gpu: ## reinstall torch with CUDA wheels (default PyPI torch on Windows is CPU-only)
	# --index-url only, with NO --extra-index-url: adding PyPI as a fallback lets the
	# plain (CPU) wheel win the resolution and you silently end up back on CPU.
	$(UV) pip install --python $(PY) --upgrade --force-reinstall \
		--index-url https://download.pytorch.org/whl/$(CUDA) torch torchvision
	@$(PY) -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

.PHONY: cpu
cpu: ## reinstall the CPU-only torch build
	$(UV) pip install --python $(PY) --upgrade --force-reinstall \
		--index-url https://download.pytorch.org/whl/cpu torch torchvision

.PHONY: add
add: ## add a dependency: make add PKG=wandb
	@test -n "$(PKG)" || { echo "usage: make add PKG=<package>"; exit 1; }
	$(UV) add $(PKG)

.PHONY: lock
lock: uv ## refresh uv.lock without installing
	$(UV) lock

.PHONY: kernel
kernel: ## register ./.venv as a Jupyter kernel named "$(KERNEL)"
	$(PY) -m ipykernel install --user --name $(KERNEL) --display-name "Python (research-papers)"

## ------------------------------------------------------------------ notebooks

.PHONY: lab
lab: ## launch JupyterLab in this folder
	$(PY) -m jupyterlab

.PHONY: notebook
notebook: ## launch classic Jupyter Notebook
	$(PY) -m notebook

.PHONY: run
run: ## execute a notebook end-to-end: make run NB=RoPE/rope.ipynb
	@test -n "$(NB)" || { echo "usage: make run NB=<path/to.ipynb>"; exit 1; }
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 "$(NB)"

## ------------------------------------------------------------------- quality

.PHONY: fmt
fmt: ## format code with ruff
	$(UV) run ruff format .

.PHONY: lint
lint: ## lint code with ruff
	$(UV) run ruff check .

.PHONY: test
test: ## run pytest
	$(UV) run pytest -q

.PHONY: info
info: ## print environment / hardware summary
	@$(PY) -c "import sys, torch; print('python', sys.version.split()[0]); print('torch  ', torch.__version__); print('cuda   ', torch.version.cuda, '| available:', torch.cuda.is_available()); print('device ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

## -------------------------------------------------------------------- cleanup

.PHONY: clean
clean: ## remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache **/__pycache__ __pycache__

.PHONY: clean-venv
clean-venv: ## delete the virtual environment
	rm -rf .venv
