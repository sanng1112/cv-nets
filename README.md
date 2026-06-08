# cv-nets

<div align="center">

**Clean-Architecture Computer Vision Neural Network Research Framework**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-236%20passed-brightgreen.svg)](tests/)

*Modular · Config-Driven · Research-Grade · Pip-Installable*

</div>

---

## 📖 Giới Thiệu (Introduction)

**cv-nets** là một framework nghiên cứu Computer Vision được xây dựng theo kiến trúc Clean Architecture (Hexagonal Architecture). Framework cung cấp các khối dựng sẵn (building blocks) để xây dựng, huấn luyện, kiểm tra (introspect) và đánh giá (benchmark) mô hình neural network một cách có hệ thống.

### ✨ Tính Năng Chính (Key Features)

| Tính năng | Mô tả |
|-----------|-------|
| 🏗️ **Kiến trúc phân lớp** | 4 tầng rõ ràng: Core → Application → Infrastructure → Interface |
| ⚙️ **Config-driven** | Định nghĩa mô hình qua YAML, build tự động qua Registry + Factory |
| 🔬 **Research toolkit** | LayerProbe, StatsCollector, BenchmarkRunner, ExperimentTracker |
| 🧩 **Module hóa** | Layers, Blocks, Models, Loss Functions, Trainer — mỗi package độc lập, testable |
| 🧪 **Test đầy đủ** | Unit test + integration test cho mọi component |
| 📦 **Pip-installable** | `pip install -e .` — dùng được ngay |

---

## 🏛️ Kiến Trúc (Architecture)

```
┌─────────────────────────────────────────────────────────┐
│                    INTERFACE LAYER                       │
│  YAML config, Python API, CLI                           │
├─────────────────────────────────────────────────────────┤
│                   APPLICATION LAYER                      │
│  Trainer, ModelFactory, ConfigResolver, Research tools   │
├─────────────────────────────────────────────────────────┤
│                   INFRASTRUCTURE LAYER                   │
│  PyTorch, DataLoader, Disk I/O                          │
├─────────────────────────────────────────────────────────┤
│                      CORE LAYER                          │
│  BaseLayer, Registry, ConfigSchema, Exceptions           │
└─────────────────────────────────────────────────────────┘
```

| Tầng | Package | Vai trò |
|------|---------|---------|
| **Core** | `cvnets.core` | Abstractions ổn định: `BaseLayer`, `Registry`, `ConfigSchema` |
| **Application** | `cvnets.trainer`, `cvnets.models`, `cvnets.loss_fn`, **`cvnets.research`** | Orchestration use-case: Trainer, ModelFactory, Loss Functions, Layer inspection |
| **Infrastructure** | `cvnets.layers`, `cvnets.blocks` | Triển khai cụ thể: Conv2d, Linear, ReLU, BatchNorm, ResBlock... |
| **Interface** | `cvnets.config` | API người dùng: YAML config, `ConfigResolver` |

---

## 📦 Cài Đặt (Installation)

```bash
# Clone repository
git clone https://github.com/your-org/cv-nets.git
cd cv-nets

# Tạo virtual environment (khuyên dùng)
python3 -m venv .venv
source .venv/bin/activate

# Cài đặt
pip install -e .              # Cài cơ bản
pip install -e ".[dev]"       # + pytest, pytest-cov, torchvision
pip install -e ".[demo]"      # + torchvision, pygame

# Kiểm tra
python -c "import cvnets; print(cvnets.__version__)"
# Output: 0.1.0
```

**Yêu cầu:** Python ≥ 3.10, PyTorch ≥ 2.0

---

## 🚀 Quick Start

### 1. Xây dựng mô hình từ YAML config

```yaml
# config.yaml
model:
  name: "simple_cnn"
  layers:
    - type: Conv2d
      in_channels: 3
      out_channels: 16
      kernel_size: 3
    - type: ReLU
    - type: Flatten
    - type: Linear
      in_features: 2304
      out_features: 10
```

```python
from cvnets.config import ConfigResolver
from cvnets.models import ModelFactory

config = ConfigResolver.from_yaml("config.yaml")
model = ModelFactory.create(config.model)
```

### 2. Huấn luyện với Trainer

```python
from cvnets.trainer import Trainer
from cvnets.loss_fn import build_loss_fn

# Chọn loss function phù hợp
criterion = build_loss_fn("focal_loss", category="classification", gamma=2.0)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,  # <-- bất kỳ loss function nào cũng được
    epochs=10,
    learning_rate=1e-3,
)
trainer.fit()
```


---

## 📉 cvnets.loss_fn — Thư Viện Loss Function Hiện Đại (21 Losses)

Package `cvnets.loss_fn` cung cấp **21 loss functions** hiện đại, được tổ chức theo 6 problem domain, tất cả đều đăng ký vào `LOSS_REGISTRY` và sử dụng được ngay với `Trainer`.

### Tổng Quan

| Domain | Loss Functions | Số lượng |
|--------|---------------|:--------:|
| 🏷️ **Classification** | `CrossEntropyLoss`, `FocalLoss`, `AsymmetricLoss`, `ArcFaceLoss`, `CosFaceLoss` | 5 |
| 🧠 **Segmentation** | `DiceLoss`, `TverskyLoss`, `LovaszSoftmax`, `ComboLoss` | 4 |
| 📦 **Detection** | `IoULoss` (IoU/GIoU/DIoU/CIoU), `SmoothL1Loss` | 2 |
| 📏 **Metric Learning** | `TripletLoss`, `ContrastiveLoss`, `NTXentLoss`, `CircleLoss` | 4 |
| 🔄 **Self-Supervised** | `NegativeFreeLoss` (BYOL/SimSiam), `VICRegLoss`, `BarlowTwinsLoss` | 3 |
| 📊 **Regression** | `HuberLoss`, `QuantileLoss`, `WingLoss` | 3 |
| | **Tổng cộng** | **21** |

### Cách Sử Dụng

```python
from cvnets.loss_fn import build_loss_fn

# Classification — Focal Loss cho dữ liệu mất cân bằng
criterion = build_loss_fn("focal_loss", category="classification", gamma=2.0)

# Segmentation — Dice Loss
criterion = build_loss_fn("dice_loss", category="segmentation")

# Detection — CIoU Loss cho bounding box regression
criterion = build_loss_fn("iou_loss", category="detection", mode="ciou")

# Metric Learning — Triplet Loss với batch-hard mining
criterion = build_loss_fn("triplet_loss", category="metric_learning", margin=1.0)

# Self-Supervised — VICReg
criterion = build_loss_fn("vicreg_loss", category="ssl")

# Regression — Huber Loss
criterion = build_loss_fn("huber_loss", category="regression", delta=1.0)

# Dùng trực tiếp với Trainer (drop-in replacement)
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    criterion=criterion,  # <-- bất kỳ loss nào cũng được
    num_epochs=10,
    device="cpu",
)
trainer.fit()
```

### Bảng Chọn Loss Theo Bài Toán

| Bài Toán | Loss Phù Hợp | Khi Nào Dùng |
|----------|-------------|-------------|
| **Phân loại** (cân bằng) | `CrossEntropyLoss` | Mặc định cho multi-class |
| **Phân loại** (mất cân bằng) | `FocalLoss` | Lớp thiểu số, hard examples |
| **Multi-label** | `AsymmetricLoss` | Nhiều nhãn trên 1 mẫu |
| **Face/Fine-grained ID** | `ArcFaceLoss` / `CosFaceLoss` | Angular margin cho embeddings |
| **Segmentation** (overlap) | `DiceLoss` | Y tế, foreground/background |
| **Segmentation** (mất cân bằng) | `TverskyLoss` | Chi phí FP/FN không đối xứng |
| **Segmentation** (IoU) | `LovaszSoftmax` | Tối ưu trực tiếp Jaccard index |
| **Segmentation** (cân bằng) | `ComboLoss` | CE + Dice hybrid |
| **Object Detection** (box) | `IoULoss` (GIoU/DIoU/CIoU) | Bounding box regression |
| **Object Detection** (class) | `FocalLoss` | Dense detection (RetinaNet) |
| **Detection** (robust) | `SmoothL1Loss` | Delta regression |
| **Metric Learning** | `TripletLoss` | Face recognition, re-ID |
| **Siamese Networks** | `ContrastiveLoss` | Similarity learning |
| **Contrastive SSL** | `NTXentLoss` (InfoNCE) | SimCLR, MoCo, CLIP |
| **Unified Metric** | `CircleLoss` | Flexible similarity optimization |
| **Self-Supervised** (neg-free) | `NegativeFreeLoss` | BYOL, SimSiam |
| **Self-Supervised** | `VICRegLoss` | Variance-covariance regularization |
| **Self-Supervised** | `BarlowTwinsLoss` | Redundancy reduction |
| **Regression** | `HuberLoss` | Robust to outliers |
| **Quantile Regression** | `QuantileLoss` | Prediction intervals |
| **Landmark Detection** | `WingLoss` | Facial landmarks, keypoints |

### Cấu Trúc Package

```python
loss_fn/
├── __init__.py              # register_loss_fn(), build_loss_fn(), SUPPORTED_LOSSES
├── base_loss.py             # BaseLoss abstract class
├── reduction.py             # reduce_loss() helper
├── classification/
│   ├── cross_entropy.py     # CrossEntropyLoss + label smoothing
│   ├── focal_loss.py        # FocalLoss (Lin et al., 2017)
│   ├── asymmetric_loss.py   # ASL (Ridnik et al., 2021)
│   ├── arcface_loss.py      # ArcFace (Deng et al., 2019)
│   └── cosface_loss.py      # CosFace (Wang et al., 2018)
├── segmentation/
│   ├── dice_loss.py         # Dice (Milletari et al., 2016)
│   ├── tversky_loss.py      # Tversky (Salehi et al., 2017)
│   ├── lovasz_softmax.py    # Lovász-Softmax (Berman et al., 2018)
│   └── combo_loss.py        # CE + Dice
├── detection/
│   ├── iou_loss.py          # IoU/GIoU/DIoU/CIoU
│   └── smooth_l1_loss.py    # SmoothL1
├── metric_learning/
│   ├── triplet_loss.py      # Triplet (batch-hard)
│   ├── contrastive_loss.py  # Siamese Contrastive
│   ├── ntxent_loss.py       # InfoNCE / NT-Xent
│   └── circle_loss.py       # Circle Loss
├── ssl/
│   ├── negative_free_loss.py # BYOL / SimSiam
│   ├── vicreg_loss.py       # VICReg
│   └── barlow_twins_loss.py # Barlow Twins
└── regression/
    ├── huber_loss.py         # Huber
    ├── quantile_loss.py      # Quantile
    └── wing_loss.py          # Wing Loss
```

### Tích Hợp Sâu

- **Tất cả loss** extend `BaseLoss` (abstract class) với `reduction='mean'|'sum'|'none'`
- **Tự động đăng ký** vào `LOSS_REGISTRY` qua decorator `@register_loss_fn(name, category=...)`
- **Auto-import**: sub-packages tự động quét và import module, không cần maintain registry thủ công
- **236 tests** — mỗi loss có test registration, forward shape, reduction modes, gradient flow, extra_repr

---

## 🔬 cvnets.research — Bộ Công Cụ Nghiên Cứu Layer

Package `cvnets.research` cung cấp 5 công cụ chuyên sâu để kiểm tra (introspect), thống kê (statistics), đánh giá (benchmark), và theo dõi thí nghiệm (experiment tracking) từng layer trong mô hình.

### Tổng Quan

| Công cụ | File | Mô tả |
|---------|------|-------|
| `LayerProbe` | `probe.py` | Gắn hooks để bắt activation/gradient |
| `StatsCollector` | `stats.py` | Tính thống kê từ tensor đã thu thập |
| `LayerReport` | `report.py` | Tạo báo cáo có cấu trúc (JSON) |
| `BenchmarkRunner` | `benchmark.py` | So sánh hiệu năng giữa các biến thể layer |
| `ExperimentTracker` | `tracker.py` | Lưu trữ thí nghiệm với metadata |

---

### 1. LayerProbe — Bắt Activation & Gradient

`LayerProbe` gắn forward/backward hooks vào bất kỳ `nn.Module` nào, ghi lại mọi activation và gradient vào buffer trong bộ nhớ.

```python
import torch
from torch import nn
from cvnets.research import LayerProbe

# Tạo model
model = nn.Sequential(
    nn.Conv2d(3, 16, 3),
    nn.ReLU(),
    nn.Linear(16 * 30 * 30, 10),
)

# Gắn probe vào layer cần kiểm tra
probe = LayerProbe()
probe.attach(model[0])   # Conv2d
probe.attach(model[1])   # ReLU

# Forward + backward
x = torch.randn(2, 3, 32, 32)
out = model(x)
out.sum().backward()

# Đọc kết quả
print(f"Conv2d activations: {probe.activations[0].shape}")  # [2, 16, 30, 30]
print(f"ReLU   activations: {probe.activations[1].shape}")  # [2, 16, 30, 30]
print(f"Conv2d gradients:   {probe.gradients[0].shape}")    # [2, 16, 30, 30]

probe.clear()  # Xóa buffer
```

**Context Manager:**
```python
with LayerProbe() as probe:
    probe.attach(model[0])
    out = model(x)
    out.sum().backward()
    print(len(probe.activations))  # 1
# Tự động detach_all() khi thoát context
```

**API:**
| Method | Mô tả |
|--------|-------|
| `attach(module)` | Gắn forward + backward hooks |
| `detach()` | Gỡ hooks của layer gắn gần nhất |
| `detach_all()` | Gỡ tất cả hooks |
| `clear()` | Xóa buffer activations/gradients |
| `activations: List[Tensor]` | Danh sách activation đã clone |
| `gradients: List[Tensor]` | Danh sách gradient đã clone |

---

### 2. StatsCollector — Thống Kê Tensor

`StatsCollector` tính toán các chỉ số thống kê từ danh sách tensor (activation hoặc gradient).

```python
from cvnets.research import StatsCollector

# Thống kê cơ bản
stats = StatsCollector.compute(probe.activations)
print(stats)
# {
#     "mean": 0.234,   "std": 0.567,   "min": -1.234,  "max": 2.456,
#     "l2_norm": 45.678,
#     "sparsity": 0.350,          # Tỷ lệ phần tử = 0
#     "dead_neuron_ratio": 0.05   # Tỷ lệ neuron chết (tổng activation = 0)
# }

# Gradient norm
grad_stats = StatsCollector.gradient_norm(probe.gradients)
# {"grad_l2_norm": 12.345}

# Histogram
hist = StatsCollector.histogram(probe.activations, bins=20)
# {"hist_bin_edges": [-1.5, -1.2, ..., 2.6],
#  "hist_counts": [3, 12, 45, ..., 8]}
```

**Các metric:**
| Key | Ý nghĩa |
|-----|---------|
| `mean` / `std` | Giá trị trung bình / độ lệch chuẩn (population) |
| `min` / `max` | Giá trị nhỏ nhất / lớn nhất |
| `l2_norm` | Chuẩn L2 |
| `sparsity` | Tỷ lệ phần tử = 0 (0.0 → 1.0) |
| `dead_neuron_ratio` | Tỷ lệ neuron không kích hoạt (≥ 2D tensor) |
| `grad_l2_norm` | Chuẩn L2 của gradient |

---

### 3. LayerReport — Báo Cáo Có Cấu Trúc

`LayerReport` kết hợp `LayerProbe` + `StatsCollector` để tạo báo cáo hoàn chỉnh, xuất ra JSON.

```python
from cvnets.research import LayerProbe, LayerReport

# Thu thập dữ liệu
conv = nn.Conv2d(3, 16, 3)
with LayerProbe() as probe:
    probe.attach(conv)
    for _ in range(10):
        x = torch.randn(4, 3, 32, 32)
        out = conv(x)
        out.sum().backward()

# Tạo báo cáo
report = LayerReport.generate(
    name="conv1", layer_type="Conv2d", probe=probe,
    include_histogram=True, histogram_bins=30,
)

# Xuất JSON
json_str = LayerReport.to_json(report)
with open("report_conv1.json", "w") as f:
    f.write(json_str)

# In ra terminal
LayerReport.print_summary(report)
```

**Output mẫu:**
```
============================================================
  Layer: conv1  (Conv2d)
  Forward passes: 10
============================================================
  Activations:
                mean: 0.0234       std: 0.5671
                 min: -1.2340      max: 2.4560
            l2_norm: 45.6780  sparsity: 0.3500
  Gradients:
                mean: -0.0012      std: 0.0890
```

---

### 4. BenchmarkRunner — So Sánh Hiệu Năng Layer

`BenchmarkRunner` chạy forward/backward pass chuẩn hóa và so sánh thời gian giữa các biến thể layer. Hỗ trợ cả CPU và CUDA.

```python
from cvnets.research import BenchmarkRunner

# Định nghĩa các biến thể cần so sánh
variants = {
    "ReLU":      lambda: nn.ReLU(),
    "GELU":      lambda: nn.GELU(),
    "LeakyReLU": lambda: nn.LeakyReLU(),
    "SiLU":      lambda: nn.SiLU(),
}

# Chạy benchmark
results = BenchmarkRunner.run(
    variants=variants,
    input_shape=(64, 128),     # (batch, features)
    num_steps=200,             # Số bước đo
    num_warmup=20,             # Số bước warmup (bỏ qua)
)

# In bảng so sánh
BenchmarkRunner.print_table(results)

# Hoặc lấy dữ liệu dạng list
table = BenchmarkRunner.compare(results)
# [
#   {"variant": "ReLU", "forward_ms": 0.0123, "backward_ms": 0.0089, "num_params": 0},
#   ...
# ]
```

**Output bảng:**
```
Variant              Fwd(ms)    Bwd(ms)   Params
---------------------------------------------------
ReLU                   0.0123     0.0089         0
GELU                   0.0234     0.0156         0

---

### 5. ExperimentTracker — Theo Dõi Thí Nghiệm

`ExperimentTracker` lưu trữ toàn bộ thí nghiệm: config, metrics, artifacts vào thư mục có cấu trúc.

```python
from cvnets.research import ExperimentTracker

# Khởi tạo tracker
tracker = ExperimentTracker(base_dir="./experiments")

# Bắt đầu một run mới
run_dir = tracker.start(run_name="conv_benchmark")
# Tạo thư mục: ./experiments/conv_benchmark_20260608-143022/

# Lưu config
tracker.log_config({
    "model": {"name": "resnet18", "num_layers": 18},
    "optimizer": {"name": "adam", "lr": 1e-4},
})

# Lưu metrics (append, không ghi đè)
for epoch in range(5):
    tracker.log_metrics({
        "epoch": epoch,
        "train_loss": 0.5 - epoch * 0.05,
        "val_acc": 0.7 + epoch * 0.05,
    })

# Lưu artifact (file hoặc thư mục)
tracker.log_artifact("plots/loss_curve.png")
tracker.log_artifact("checkpoints/best_model.pth")

# Kết thúc run — ghi summary.json
tracker.finish()
```

**Cấu trúc thư mục sau khi chạy:**
```
experiments/
└── conv_benchmark_20260608-143022/
    ├── config.yaml          # Cấu hình thí nghiệm
    ├── metrics.json         # Danh sách metrics [{...}, {...}, ...]
    ├── summary.json         # Tổng kết: run_name, finished_at, num_metrics
    └── artifacts/
        ├── loss_curve.png
        └── best_model.pth
```

**API:**
| Method | Mô tả |
|--------|-------|
| `start(run_name=None)` | Tạo thư mục run mới, trả về path |
| `log_metrics(metrics)` | Append metrics → `metrics.json` |
| `log_config(config)` | Ghi config → `config.yaml` |
| `log_artifact(src)` | Copy file/thư mục → `artifacts/` |
| `finish()` | Ghi `summary.json` |

---

### 🔗 Workflow Kết Hợp (End-to-End)

```python
import torch
from torch import nn
from cvnets.research import (
    LayerProbe, StatsCollector, LayerReport,
    BenchmarkRunner, ExperimentTracker,
)

# 1. Khởi tạo tracker
tracker = ExperimentTracker(base_dir="./runs")
tracker.start(run_name="layer_analysis")

# 2. Định nghĩa model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return x

model = SimpleCNN()

# 3. Probe từng layer
probes = {}
for name, layer in model.named_children():
    probe = LayerProbe()
    probe.attach(layer)
    probes[name] = probe

# 4. Chạy forward + backward
x = torch.randn(4, 3, 32, 32)
out = model(x)
out.sum().backward()

# 5. Tạo báo cáo cho từng layer
for name, probe in probes.items():
    report = LayerReport.generate(
        name=name,
        layer_type=type(model.get_submodule(name)).__name__,
        probe=probe,
        include_histogram=True,
    )
    # Lưu report ra file JSON
    with open(f"{tracker.run_dir}/{name}_report.json", "w") as f:
        f.write(LayerReport.to_json(report))

# 6. Benchmark các activation function
activation_results = BenchmarkRunner.run(
    variants={
        "ReLU": lambda: nn.ReLU(),
        "GELU": lambda: nn.GELU(),
        "SiLU": lambda: nn.SiLU(),
    },
    input_shape=(256, 512),
    num_steps=100,
)
tracker.log_metrics({"benchmark": BenchmarkRunner.compare(activation_results)})

# 7. Kết thúc
tracker.log_config({"model": "SimpleCNN"})
tracker.finish()

print(f"Results saved to: {tracker.run_dir}")
```

---

## 📁 Cấu Trúc Thư Mục (Package Structure)

```
src/cvnets/
├── __init__.py              # Package root, version = "0.1.0"
├── core/                    # 🟦 Core Layer
│   ├── base_layer.py        #   Abstract BaseLayer
│   ├── base_block.py        #   Abstract BaseBlock
│   ├── base_model.py        #   Abstract BaseModel
│   ├── registry.py          #   Registry pattern
│   ├── exceptions.py        #   Custom exceptions
│   └── config/schema.py     #   Config validation
├── layers/                  # 🟩 Infrastructure — Layer implementations
│   ├── conv_layer.py
│   ├── linear_layer.py
│   ├── flatten.py
│   ├── activation/          #   ReLU, etc.
│   ├── normalization/       #   BatchNorm, etc.
│   └── pooling/             #   MaxPool, AvgPool, etc.
├── blocks/                  # 🟩 Infrastructure — Block implementations
│   ├── conv_bn_act.py
│   └── registry.py
├── models/                  # 🟨 Application — Model factory & zoo
│   ├── factory.py
│   ├── base.py
│   └── zoo/                 #   Pre-built model architectures
├── trainer/                 # 🟨 Application — Training pipeline
│   ├── trainer.py
│   ├── metrics.py
│   └── callbacks.py
├── loss_fn/                 # 🟨 Application — Loss Function Library ⭐ MỚI
│   ├── base_loss.py         #   BaseLoss (abstract)
│   ├── reduction.py         #   reduce_loss() helper
│   ├── classification/      #   5 losses
│   ├── segmentation/        #   4 losses
│   ├── detection/           #   2 losses
│   ├── metric_learning/     #   4 losses
│   ├── ssl/                 #   3 losses
│   └── regression/          #   3 losses
├── research/                # 🟨 Application — Layer inspection toolkit
│   ├── probe.py             #   LayerProbe
│   ├── stats.py             #   StatsCollector
│   ├── report.py            #   LayerReport
│   ├── benchmark.py         #   BenchmarkRunner
│   └── tracker.py           #   ExperimentTracker
├── config/                  # 🟥 Interface
│   └── resolver.py          #   ConfigResolver (YAML → dict)
└── utils/                   # 🔧 Utilities
    └── logger.py

tests/
├── test_core/               # Tests for core layer
├── test_layers/             # Tests for layer implementations
├── test_blocks/             # Tests for block implementations
├── test_models/             # Tests for model factory & zoo
├── test_trainer/            # Tests for training pipeline
├── test_config/             # Tests for config resolution
├── test_utils/              # Tests for utilities
├── test_loss_fn/            # Tests for loss function library ⭐ MỚI (236 tests)
│   ├── test_base_loss.py        #   3
│   ├── test_reduction.py        #   4
│   ├── test_loss_registration.py #  11
│   ├── classification/          #  52 tests
│   ├── segmentation/            #  33 tests
│   ├── detection/               #  24 tests
│   ├── metric_learning/         #  38 tests
│   ├── ssl/                     #  33 tests
│   └── regression/              #  36 tests
└── test_research/           # Tests for research toolkit (38 tests)
    ├── test_probe.py        #   8 tests
    ├── test_stats.py        #   9 tests
    ├── test_report.py       #   4 tests
    ├── test_benchmark.py    #   5 tests
    ├── test_tracker.py      #   7 tests
    └── test_integration.py  #   5 tests
```

---

## 🧪 Chạy Tests

Chạy toàn bộ **274 tests** (236 loss_fn + 38 research):

```bash
# Chạy toàn bộ test suite
pytest tests/ -v

# Loss function tests
pytest tests/test_loss_fn/ -v
# Kết quả: 236 passed

# Chỉ chạy research tests
pytest tests/test_research/ -v
# Kết quả: 38 passed

# Với coverage
pytest tests/test_research/ --cov=cvnets.research --cov-report=term-missing

---

## 📄 Giấy Phép (License)

```
MIT License

Copyright (c) 2026 Anng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🤝 Đóng Góp (Contributing)

1. Fork repository
2. Tạo branch: `git checkout -b feat/ten-tinh-nang`
3. Viết code + tests
4. Chạy tests: `pytest tests/ -v`
5. Commit: `git commit -m "feat: mô tả"`
6. Push và tạo Pull Request

---

<div align="center">
Made with ❤️ by <b>Anng</b>
</div>
