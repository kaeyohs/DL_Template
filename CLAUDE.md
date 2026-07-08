# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A deep learning project template: PyTorch + Hydra + uv. The `src/DL_Template` package, `configs/`, and `scripts/` form the reusable skeleton — real projects built from this template extend `ExampleDataset`/`ExampleModel` and add config groups as needed.

## Commands

Dependencies are managed with `uv`; always prefix Python invocations with `uv run` (do not assume a pre-activated venv).

```bash
uv sync --group dev            # install runtime + dev deps (ruff, pytest, pre-commit)
uv run pre-commit install      # one-time: enable lint/format hooks on commit

uv run python scripts/data_download.py
uv run python scripts/train.py
uv run python scripts/test.py

uv run pytest                  # run all tests
uv run pytest tests/test_model.py::test_model_forward_shape   # single test
uv run ruff check .            # lint
uv run ruff format .           # format
uv run tensorboard --logdir runs
```

## Architecture

**Config flow (Hydra).** `scripts/*.py` are the only entry points and are all decorated with `@hydra.main(config_path="../configs", config_name="config")`. `configs/config.yaml` composes three config groups via its `defaults` list — `data/default.yaml`, `model/default.yaml`, `train/default.yaml` — each mapped into the runtime `cfg` under `cfg.data`, `cfg.model`, `cfg.train`. Adding a new variant (e.g. a second model) means adding `configs/model/<name>.yaml` and selecting it with `model=<name>` on the CLI, not branching in code. Any config value can be overridden from the CLI, e.g. `train.epochs=20`.

**Package layout (`src/DL_Template`).** Uses a src-layout package; the package directory is named `DL_Template` to match the project name (not the conventional lowercase). Scripts import from it as `from DL_Template.xxx import ...` — this only resolves after `uv sync` installs the package in editable mode.

- `data/dataset.py`, `data/datamodule.py` — `ExampleDataset` (placeholder `Dataset` reading `cfg.data.root/<split>`) and `build_dataloaders(cfg)`, which wires the dataset into train/val `DataLoader`s using `cfg.data.batch_size`/`num_workers`. Replace `ExampleDataset.__getitem__` for a real dataset.
- `models/model.py` — `ExampleModel` (placeholder MLP) and `build_model(cfg)`, which reads `cfg.model.*` dims. Swap in a real architecture here; keep the `build_model(cfg) -> nn.Module` factory shape so `scripts/train.py`/`test.py` don't need to change.
- `training/trainer.py` — framework-agnostic `train_one_epoch`/`evaluate` functions (not a class-based `Trainer`). They take an already-built model/loader/optimizer/loss/device and don't read `cfg` directly. Each returns `(loss, metric)`; both accept an optional torchmetrics `metric`. `train_one_epoch` also takes an optional AMP `scaler` (autocast mixed precision) and `grad_clip` (max-norm clipping, unscaled first under AMP).
- `utils.py` — flat module (deliberately not a package) with `seed_everything` (also sets cuDNN deterministic flags), `get_device`, `get_logger`. Reproducible multi-worker loading also depends on `_seed_worker`/`generator` wired into the loaders in `data/datamodule.py`.

**Script responsibilities.** `train.py` seeds, builds dataloaders/model/optimizer/scheduler, runs the epoch loop with optional AMP + gradient clipping, logs loss/accuracy/lr to TensorBoard (`cfg.train.log_dir`), and checkpoints to `cfg.train.checkpoint_dir`. Checkpoints are **dicts** (`epoch`, `model`, `optimizer`, `scheduler`, `scaler`, `best_val_loss`, resolved `cfg`) — not bare state_dicts — so `train.resume=true` can restart from `last.pt`. `best.pt` is written on val-loss improvement, `last.pt` every epoch. AMP is gated on `cfg.train.amp` *and* a CUDA device. The scheduler is chosen by `cfg.train.scheduler.name` (`none`/`cosine`/`step`) in `build_scheduler`. `test.py` rebuilds model/dataloaders from the same config and loads `ckpt["model"]` from `checkpoint_dir/best.pt` — no checkpoint-path CLI arg, so switch checkpoints by overriding `train.checkpoint_dir`. `data_download.py` is a stub gated on `cfg.data.source_url` being set.

Note the example pipeline assumes a **multiclass classification** task (CrossEntropyLoss + `MulticlassAccuracy` sized by `cfg.model.output_dim`); change the loss and metric in the scripts for other task types.

## Dev container

`.devcontainer/` builds from `nvidia/cuda:13.0.3-devel-ubuntu24.04` (CUDA 13.0 toolkit + `nvcc` for building CUDA extensions), installs `uv` via the official `ghcr.io/astral-sh/uv` image, and runs `uv sync --group dev && uv run pre-commit install` as `postCreateCommand`. There is no system Python in the image — `uv` installs and manages the Python 3.14 toolchain (pinned by `.python-version`) into `/opt/python`. GPU passthrough in `docker-compose.yml` is present but commented out (requires NVIDIA Container Toolkit on the host) — uncomment the `deploy.resources.reservations.devices` block to enable.

## PyTorch / CUDA wheels

`torch` is pinned to `>=2.12` and resolved from the `pytorch-cu130` index declared in `pyproject.toml` (`[[tool.uv.index]]` + `[tool.uv.sources]`), **not** PyPI. This is required because the Python 3.14 (cp314) CUDA 13.0 wheels are only published on `download.pytorch.org/whl/cu130`; the PyPI build is CPU-only. If you change the CUDA version, update both the index URL and the Dockerfile base image together.
