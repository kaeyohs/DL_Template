# DL_Template

A deep learning project template: PyTorch + Hydra + uv.

## Stack

- **Package management**: [uv](https://docs.astral.sh/uv/)
- **Framework**: PyTorch
- **Config management**: Hydra / YAML (`configs/`)
- **Experiment tracking**: TensorBoard
- **Lint/format**: Ruff
- **Tests**: pytest
- **Dev environment**: `.devcontainer/` (Docker)

## Setup

```bash
uv sync --group dev
uv run pre-commit install
```

## Usage

```bash
uv run python scripts/data_download.py
uv run python scripts/train.py
uv run python scripts/test.py
```

Override any config value from the CLI, e.g.:

```bash
uv run python scripts/train.py train.epochs=20 train.lr=0.0005 data.batch_size=64
```

View training curves:

```bash
uv run tensorboard --logdir runs
```

## Project layout

```
configs/          Hydra config groups (data, model, train)
scripts/          Entry points (data_download, train, test)
src/DL_Template/  Package: data, models, training, utils
tests/            pytest tests
.devcontainer/    Dev container (Dockerfile, docker-compose, devcontainer.json)
```
