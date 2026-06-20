# Repository Guidelines

## Project Structure & Module Organization

The repository is organized as a dual-layout Python project — source code lives under `src/cvnets/` (canonical, installed package) while legacy code and research scripts sit at the root level.

```
src/cvnets/          # Installed package (the canonical source)
├── core/            # Base classes, registry, exceptions
├── layers/          # Neural network layers (conv, attention, normalisation, pooling, activation)
├── blocks/          # Composable building blocks (residual, inverted residual, transformer, MLP, SE)
├── models/          # Model factory, zoo of pretrained backbones via timm
├── loss_fn/         # Loss functions (classification, detection, segmentation, regression, metric learning, SSL)
├── optim/           # Optimiser wrappers (Adam, SGD) and scheduler registry
├── scheduler/       # LR schedulers (cosine, step, one-cycle)
├── trainer/         # BaseTrainer with AMP, gradient accumulation, EMA, WandB logging
├── data/            # Dataset adapters, transforms, DataLoader configuration
├── config/          # YAML schema and resolver
├── export/          # ONNX and TorchScript export
├── research/        # Probe, benchmark, stats, report tools
├── papers/          # ArXiv client, paper storage, orchestrator
└── utils/           # File I/O, logger, misc helpers

tests/               # pytest suite, mirrors src structure (test_blocks/, test_layers/, test_loss_fn/, …)
scripts/             # CLI entry points (train.py, evaluate.py) and ad-hoc test runners
docs/                # Reference documentation (00_INDEX.md → 06_ADVANCED_FEATURES.md)
configs/             # YAML configuration files
data/                # Datasets (CIFAR-100, MNIST) and adapters
papers/              # Research PDFs and analysis notes
```

## Build, Test, and Development Commands

This project uses **uv** for dependency management and **pytest** for testing.

| Command | Purpose |
|---|---|
| `uv sync` | Install all project dependencies (from `pyproject.toml`/`uv.lock`) |
| `uv run pytest` | Run the full test suite |
| `uv run python scripts/test_all_losses.py` | Run loss function tests |
| `uv run python scripts/test_attention.py` | Run attention layer tests |
| `uv run python scripts/test_builder.py` | Run model builder & EMA tests |
| `uv run python scripts/test_trainer.py` | Run BaseTrainer integration tests |
| `uv run python scripts/test_heads.py` | Run network heads tests |
| `uv run python scripts/test_export.py` | Run inference & deployment tests |
| `uv run python scripts/train.py --config configs/demo.yaml` | Launch a training run |
| `uv run python scripts/evaluate.py --checkpoint <path>` | Evaluate a checkpoint |

CLI entry points (after `uv sync`):
- `cvnets-train` — launch training
- `cvnets-eval` — evaluate a model
- `cvnets-download-papers` — download arXiv papers

## Coding Style & Naming Conventions

- **Python**: Follow [PEP 8](https://peps.python.org/pep-0008/). Indent with 4 spaces.
- **Naming**:
  - `snake_case` for functions, methods, variables, file names.
  - `PascalCase` for classes (e.g., `BaseTrainer`, `SimplexETFClassifier`).
  - `UPPER_CASE` for module-level constants.
- **Type hints**: Required for all function signatures.
- **Docstrings**: Google-style (with `Args:`, `Returns:`, `Raises:` sections).
- **Imports**: Standard library → third-party (torch, numpy) → local modules. One import per line.
- **No linter/formatter config is committed** — `ruff` or `black` usage is optional but recommended.

## Testing Guidelines

- **Framework**: `pytest` (no unittest boilerplate).
- **Location**: Tests live in `tests/`, mirroring the `src/cvnets/` module hierarchy.
- **Naming**: Test files must be named `test_*.py`. Test functions must be named `test_*`.
- **Coverage**: Core modules (layers, blocks, loss functions, trainer) should have unit tests. PRs that add new modules should include corresponding tests.
- **Running**: `uv run pytest` at the root, or `uv run pytest tests/test_layers/` for a specific module.
- **CI**: Every push/PR to `main` runs the full test matrix (Python 3.10–3.12) via GitHub Actions (see `.github/workflows/ci.yml`).

## Commit & Pull Request Guidelines

The project uses **Conventional Commits**:

```
<type>: <short description>

feat:     New feature (adds functionality)
fix:      Bug fix
docs:     Documentation-only changes
test:     Adding or modifying tests
ci:       CI configuration changes
chore:    Maintenance, dependency updates, config changes
perf:     Performance improvements
style:    Formatting, whitespace
```

Examples from the project history:
- `feat: Add inference export (ONNX, TorchScript) and quantization (FP16, INT8)`
- `fix: Correct Quantizer method name to_int8_dynamic in custom notebook`
- `test: Add Post-Quantization Sanity Check to prevent numerical instability`
- `docs: Add detailed reference for data factory, advanced features, and index`

**Pull request requirements**:
- Title should follow the conventional commit format.
- Description must explain *what* and *why* (not just *how*).
- Link related issues using `Closes #N` or `Related to #N`.
- Include screenshots for visual changes (plots, training curves, notebook outputs).
- Ensure all CI checks pass before requesting review.
- Keep PRs focused on a single concern — split large features across multiple PRs.

## Configuration & YAML

Training is configured declaratively via YAML files (see `configs/demo.yaml` and `CONFIG_GUIDE.md`). The YAML schema defines model architecture, optimiser, scheduler, dataset, and trainer settings. When adding new configurable parameters, update the schema in `src/cvnets/config/schema.py` and the resolver in `src/cvnets/config/resolver.py`.

## Agent-Specific Instructions (AI Assistants)

When working with this codebase through an AI agent:
- **Prefer reading YAML configs** to understand a training setup before inspecting code.
- **Register new components** (layers, blocks, loss functions, models) in the appropriate registry module — the registry pattern is used throughout (`core/registry.py`, `loss_fn/__init__.py`, etc.).
- **Keep research code separate** — experimental probes and benchmarks go in `src/cvnets/research/`, not in the core engine.
- **Use the test scripts** in `scripts/` for quick smoke tests of individual subsystems.
