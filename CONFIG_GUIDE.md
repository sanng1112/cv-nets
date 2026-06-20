# Hướng Dẫn Thiết Lập Configuration (YAML) cho cv-nets

Mọi thông số của mô hình từ kiến trúc, dữ liệu, huấn luyện cho đến đánh giá hiện tại đều có thể được chuẩn hóa thông qua file YAML.

## 1. Cấu hình cho Classification (Phân loại ảnh)
Tạo file `configs/classification.yaml`:

```yaml
task_type: "classification"
num_classes: 1000

# === Data ===
dataset_root: "./dataset/imagenet"
dataset_name: "image_folder"
transform_name: "imagenet"
batch_size: 256
num_workers: 8
pin_memory: true
prefetch_factor: 2
persistent_workers: true

# === Architecture ===
model_name: "vit_base"
attention_type: "inla"     # Sử dụng kiến trúc Inverted Nonlinear Linear Attention tối tân
bottleneck_dim: 32         # Chống Information Bottleneck
expansion_dim: 256         # Chống Rank Collapse
use_etf_classifier: true   # Tự động ghim lớp cuối thành Simplex ETF chống Neural Collapse

# === Loss ===
loss_name: "focal_loss"
gamma: 2.0
alpha: 0.25

# === Optimizer & Scheduler ===
optim_name: "adamw"
lr: 0.001
weight_decay: 0.05

scheduler_name: "cosine"
epochs: 300
warmup_epochs: 20
min_lr: 1e-6

# === Trainer Engine ===
device: "cuda"
use_amp: true              # Bật FP16 Mixed Precision
accumulation_steps: 4      # Giả lập Batch Size = 256 x 4 = 1024
clip_grad_norm: 1.0
save_dir: "./checkpoints/classification_run"
```

## 2. Cấu hình cho Object Detection (Phát hiện vật thể)
Tạo file `configs/detection.yaml`:

```yaml
task_type: "detection"
num_classes: 80

# === Data ===
dataset_root: "./dataset/coco"
dataset_name: "coco_detection"
batch_size: 16

# === Architecture ===
model_name: "yolo_like_or_retinanet"

# === Loss ===
loss_name: "ciou_loss"     # Sử dụng Complete IoU (vừa tối ưu tâm, hình dáng, và tỷ lệ khung)
box_format: "xyxy"

# === Optimizer ===
optim_name: "sgd"
lr: 0.01
momentum: 0.937
weight_decay: 0.0005

scheduler_name: "step"
step_size: 30
gamma: 0.1
epochs: 100

# === Trainer Engine ===
device: "cuda"
use_amp: true
accumulation_steps: 1
save_dir: "./checkpoints/detection_run"
```

## 3. Cấu hình cho Segmentation (Phân vùng ảnh)
Tạo file `configs/segmentation.yaml`:

```yaml
task_type: "segmentation"
num_classes: 21

# === Data ===
dataset_root: "./dataset/pascal_voc"
batch_size: 32

# === Architecture ===
model_name: "unet_inla"    # Thay lõi Attention của Unet bằng INLA

# === Loss ===
loss_name: "focal_tversky_loss" # Tốt nhất cho dữ liệu mất cân bằng nặng (như y tế)
alpha: 0.7
beta: 0.3
gamma: 1.5

# === Optimizer ===
optim_name: "adamw"
lr: 0.0005
weight_decay: 0.01

scheduler_name: "cosine"
epochs: 150
warmup_epochs: 5

# === Trainer Engine ===
device: "cuda"
use_amp: true
accumulation_steps: 2
save_dir: "./checkpoints/segmentation_run"
```

## 4. Tại sao cần lưu `MỌI THAM SỐ` trong History?
Khi train xong, hệ thống `cv-nets` sẽ tự động đổ toàn bộ các tham số (Hyperparameters), cùng với điểm đánh giá từng Epoch (như Loss, mIoU, F1, Accuracy) ra file `history.json` và `config_dump.yaml`.
Điều này đảm bảo **Khả năng Tái hiện (Reproducibility)** - bạn có thể khôi phục 100% môi trường thí nghiệm của tháng trước chỉ với 1 file config.
