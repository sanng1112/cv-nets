# cv-nets Clean Architecture Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the cv-nets framework into a clean-architecture, pip-installable CV research library that is modular, config-driven, well-tested, and reusable across projects.

**Architecture:** Hexagonal/Clean Architecture with 4 concentric layers — (1) **Core/Domain**: stable abstractions (BaseLayer, Registry, Config schema) with minimal external coupling, (2) **Application**: use-case orchestration (Trainer, ModelFactory, ConfigResolver), (3) **Infrastructure**: external integrations (PyTorch, data loading, disk I/O), (4) **Interface**: user-facing API (YAML config, Python API). The framework is config-driven — model architectures defined declaratively in YAML and built at runtime via Registry + Factory pattern.

# Sub-Plan A: Foundation & Bug Fixes

**Goal:** Fix all critical bugs, establish the package structure, and build the core abstractions and config system.

---

### A.1: Fix Critical Bug #1 — `import_utils.py` missing `common` module

**Files:** Modify: `utils/import_utils.py`

- [ ] **Step 1: Fix import** — Replace `from common import LIBRARY_ROOT` with path resolution.
- [ ] **Step 2: Verify** — `python -c "from utils.import_utils import import_modules_from_folder; print('OK')"`
- [ ] **Step 3: Commit**

---

### A.2: Fix Critical Bug #2 — `build_normalization_layer` invalid kwargs

**Files:** Modify: `layers/normalization/__init__.py`

- [ ] **Step 1: Fix builder** — Use `inspect.signature` to pass only accepted kwargs.
- [ ] **Step 2: Verify** — Build a BatchNorm2d via builder, confirm no TypeError.
- [ ] **Step 3: Commit**

---

### A.3: Fix Critical Bug #3 — Config key mismatch

**Files:** Modify: `layers/activation/__init__.py`

- [ ] **Step 1: Fix resolution** — Add hyphenated key variants (`neg-slope`, `act.neg-slope`).
- [ ] **Step 2: Verify** — Build LeakyReLU with `neg-slope` in config.
- [ ] **Step 3: Commit**

---

### A.4: Fix Minor Bugs

**Files:** `layers/activation/relu.py`, `layers/normalization/__init__.py`

- [ ] **Step 1**: Fix typo `inplance` → `inplace` in ReLU.
- [ ] **Step 2**: Fix `--groups type=str` → `type=int` in norm args.
- [ ] **Step 3: Commit**

---

### A.5: Create Installable Package

**Files:** Create: `pyproject.toml`, `src/cvnets/__init__.py`

- [ ] **Step 1**: Write `pyproject.toml` with name=``cvnets``, Python 3.10+, deps (torch, pyyaml, numpy, tqdm), optional dev deps (pytest, torchvision).
- [ ] **Step 2**: Create `src/cvnets/__init__.py` with `__version__ = "0.1.0"`.
- [ ] **Step 3**: Verify: `pip install -e ".[dev]"`
- [ ] **Step 4: Commit**

---

### A.6: Core Abstractions

**Files:** Create: `src/cvnets/core/__init__.py`, `base_layer.py`, `base_block.py`, `base_model.py`, `exceptions.py`

- [ ] **Step 1-5**: `BaseLayer(nn.Module, abc.ABC)` with abstract `forward()`. `BaseBlock` same. `BaseModel` with `save()`/`load()` (stores weights + config). `exceptions.py`: `ConfigError`, `LayerDefinitionError`, `ModelBuildError`.
- [ ] **Step 6: Commit**

---

### A.7: Unified Config System

**Files:** Create: `src/cvnets/config/__init__.py`, `resolver.py`, `schema.py`; `tests/test_config/test_resolver.py`

- [ ] **Step 1**: `ConfigResolver` — reads YAML/dict/SimpleNamespace, dotted-path `get()`, to_dict/to_namespace, deep-merge.
- [ ] **Step 2**: `ConfigSchema` — `validate_model_config()` checks `model.layers` is a list with `type` fields.
- [ ] **Step 3**: 6 tests covering all features.
- [ ] **Step 4**: Run tests → pass.
- [ ] **Step 5: Commit**

---

### A.8: Refactor Logger

**Files:** Create: `src/cvnets/utils/__init__.py`, `logger.py`; `tests/test_utils/test_logger.py`

- [ ] **Step 1**: New logger — `error()` raises `LoggerError` instead of `sys.exit()`.
- [ ] **Step 2**: Test: `pytest.raises(LoggerError)`.
- [ ] **Step 3: Commit**


**Tech Stack:** Python 3.10+, PyTorch 2.x, pytest, pyproject.toml (setuptools), YAML (PyYAML), type hints throughout.

---


# Sub-Plan B: Model System

**Goal:** Refactor layers, blocks, and models with clean registry + factory pattern.

---

### B.1: Universal Registry

**Files:** `src/cvnets/core/registry.py`; `tests/test_core/test_registry.py`

- [ ] **Step 1**: `Registry` class with `register(key, category)` decorator, `build(key, ...)` factory, `keys()`, `contains()`. Global singletons: `ACTIVATION_REGISTRY`, `NORMALIZATION_REGISTRY`, `POOLING_REGISTRY`, `BLOCK_REGISTRY`, `MODEL_REGISTRY`, `LOSS_REGISTRY`.
- [ ] **Step 2**: 6 tests (register+build, duplicate, missing, category, key filter, singleton).
- [ ] **Step 3**: Run → pass.
- [ ] **Step 4: Commit**

---

### B.2: Refactor Activation System

**Files:** `src/cvnets/layers/activation/__init__.py`

- [ ] **Step 1**: Dual-registration: `register_act_fn()` registers into old dicts AND new `ACTIVATION_REGISTRY`. Fixed hyphenated key support.
- [ ] **Step 2**: Verify: `python -c "from cvnets.layers.activation import SUPPORTED_ACT_FNS; print(len(SUPPORTED_ACT_FNS))"` → ~40.
- [ ] **Step 3: Commit**

---

### B.3: Block Registry + ConvBNAct

**Files:** `src/cvnets/blocks/__init__.py`, `registry.py`, `conv_bn_act.py`

- [ ] **Step 1**: `build_block(config)` dispatches via `BLOCK_REGISTRY`.
- [ ] **Step 2**: `ConvBNAct(BaseBlock)` — uses `ConfigResolver`, registered as `conv_bn_act`.
- [ ] **Step 3: Commit**

---

### B.4: ModelFactory + Zoo

**Files:** `src/cvnets/models/__init__.py`, `base.py`, `factory.py`; `zoo/__init__.py`, `zoo/simple_cnn.py`; `tests/test_models/test_factory.py`


# Sub-Plan C: Training Pipeline

**Goal:** Reusable, callback-driven Trainer.

---

### C.1: MetricsTracker

**Files:** `src/cvnets/trainer/__init__.py`, `metrics.py`; `tests/test_trainer/test_metrics.py`

- [ ] **Step 1**: `Accuracy`, `AverageLoss`, `MetricsTracker`.
- [ ] **Step 2**: 7 tests.
- [ ] **Step 3**: Run → pass.
- [ ] **Step 4: Commit**

---

### C.2: Callback System

**Files:** `src/cvnets/trainer/callbacks.py`

- [ ] **Step 1**: `Callback` ABC with 8 hooks. `CallbackList`. Built-in: `MetricsLogger`, `ModelCheckpoint`, `EarlyStopping`, `ProgressBar`.
- [ ] **Step 2: Commit**

---

### C.3: Trainer

# Sub-Plan D: Testing, Examples & Documentation

---

### D.1: Activation Integration Tests

**Files:** `tests/test_layers/test_activations.py`

- [ ] **Step 1**: Parametrized test over all `SUPPORTED_ACT_FNS` — forward pass shape + finite values.
- [ ] **Step 2**: Run → pass.
- [ ] **Step 3: Commit**

---

### D.2: Consolidate Examples

**Files:** `examples/01_mlp_mnist/`, `examples/02_cnn_mnist/`, `examples/03_resnet_emotion/`

- [ ] **Step 1**: Migrate demos to use new `ModelFactory` + `Trainer`.
- [ ] **Step 2: Commit**

---

### D.3: Root README

**Files:** `README.md`

- [ ] **Step 1**: Write clean README with philosophy, quick start, install, structure.
- [ ] **Step 2: Commit**

---

## Self-Review

### 1. Spec Coverage

| Requirement | Task(s) |
|------------|---------|
| Fix critical bugs | A.1, A.2, A.3 |
| Minor fixes | A.4 |
| pip-installable | A.5 |
| Core abstractions | A.6 |
| Config system | A.7 |
| Clean logger | A.8 |
| Universal Registry | B.1 |
| Refactored activations | B.2 |
| Block registry | B.3 |
| ModelFactory | B.4 |
| Metrics | C.1 |
| Callbacks | C.2 |
| Trainer | C.3 |
| Activation tests | D.1 |
| Examples | D.2 |
| README | D.3 |

### 2. Placeholder Scan

No TBD, TODOs, "implement later", or similar.

### 3. Type Consistency

- `ConfigResolver.get(key, default)` → `Any`
- `Registry.build(key, ...)` → constructed instance
- `BaseLayer.forward(x: Tensor) -> Tensor`

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-06-08-cvnets-clean-architecture-refactor.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** — Fresh subagent per task, review between tasks.

**2. Inline Execution** — Execute in this session with checkpoints.

**Which approach do you prefer?**
