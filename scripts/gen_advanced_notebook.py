import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Markdown cell
text = """# Trình diễn Nâng cao: Transfer Learning trên bộ dữ liệu 100 Classes (CIFAR-100)
**"Dùng dao mổ trâu để mổ trâu thật sự!"**

Trong Notebook này, chúng ta sẽ không dùng mô hình nhỏ bé nữa. Chúng ta sẽ:
1. Load bộ dữ liệu **CIFAR-100** (100 lớp vật thể khác nhau).
2. Phóng to ảnh lên `224x224` để sử dụng Transfer Learning.
3. Sử dụng siêu mô hình **`convnext_tiny`** đã được huấn luyện trước trên ImageNet (Pretrained) từ thư viện `timm`.
4. Gắn `ClassificationHead` với 100 classes.
5. Sử dụng `AdamW` + `CosineAnnealingLR` (Lịch trình giảm LR hình sin) để tinh chỉnh (Fine-tune).
6. Áp dụng QAT (Quantization-Aware Training) thay vì PTQ để giữ độ chính xác tối đa.
7. Đánh giá tốc độ và độ mượt mà của kiến trúc CV-Nets!
"""
nb['cells'].append(nbf.v4.new_markdown_cell(text))

# Code cell
code = """import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import CV-Nets
from models.builder import CVNetModel
from engine.trainer import BaseTrainer
from engine.sanity_check import run_sanity_check
from engine.quantization import Quantizer
from engine.exporter import Exporter
from utils.plotter import HistoryPlotter

BATCH_SIZE = 64 # Giảm batch size vì ảnh 224x224 và ConvNeXt khá nặng
EPOCHS = 3 # Chạy thử 3 Epochs để thấy sự hội tụ nhanh nhờ Transfer Learning
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Sử dụng thiết bị: {DEVICE}")"""
nb['cells'].append(nbf.v4.new_code_cell(code))

# Dataloader
code2 = """# Chuẩn bị dữ liệu CIFAR-100
# Lưu ý: Ta phải Resize ảnh từ 32x32 lên 224x224 để tận dụng trọng số Pretrained của ImageNet
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Chuẩn hóa ImageNet
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Số lượng ảnh Train: {len(train_dataset)} | 100 Classes")"""
nb['cells'].append(nbf.v4.new_code_cell(code2))

# Model Builder
code3 = """# Khởi tạo mô hình SOTA (State of the art)
# Ta sẽ dùng convnext_tiny, một mạng CNN hiện đại có khả năng vượt mặt cả Vision Transformers
# Tự động gỡ bỏ Head 1000 classes của ImageNet, nối vào Head 100 classes của chúng ta.
model = CVNetModel(
    backbone_name='convnext_tiny', 
    head_type='classification', 
    head_kwargs={'num_classes': 100}, 
    pretrained=True # CHÌA KHÓA: Sử dụng tri thức từ ImageNet
)
model = model.to(DEVICE)
print("Model Built Successfully với ConvNeXt-Tiny!")"""
nb['cells'].append(nbf.v4.new_code_cell(code3))

# Trainer
code4 = """# Khởi tạo Criterion, Optimizer và Scheduler (Bộ điều phối siêu tham số)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Fake opts để truyền vào Trainer
class Opts:
    device = DEVICE
    use_amp = True
    accumulation_steps = 2 # Gradient Accumulation (Batch thực tế = 64 * 2 = 128)
    epochs = EPOCHS
    task_type = "classification"
    num_classes = 100
    save_dir = "checkpoints_cifar100"
    eval_interval = 1
    clip_grad_norm = 1.0

trainer = BaseTrainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    train_loader=train_loader,
    val_loader=val_loader,
    opts=Opts()
)

print("Đang chạy Sanity Checks...")
is_healthy = run_sanity_check(trainer)
if not is_healthy:
    print("Có lỗi trong quy trình, hãy kiểm tra lại!")"""
nb['cells'].append(nbf.v4.new_code_cell(code4))

# Training
code5 = """# Tiến hành huấn luyện (Chỉ 3 Epochs nhưng nhờ Transfer Learning kết quả sẽ rất ấn tượng)
import json
if is_healthy:
    trainer.train()
    
    # Đọc file history và vẽ đồ thị
    with open("checkpoints_cifar100/history.json", "r") as f:
        history_list = json.load(f)
    history_dict = {k: [d[k] for d in history_list] for k in history_list[0].keys()}
    
    if 'train_MulticlassAccuracy' in history_dict:
        history_dict['train_metric'] = history_dict.pop('train_MulticlassAccuracy')
    if 'val_MulticlassAccuracy' in history_dict:
        history_dict['val_metric'] = history_dict.pop('val_MulticlassAccuracy')
        
    HistoryPlotter.plot_and_save(history_dict, save_dir='checkpoints_cifar100')
    print("Đã lưu Biểu đồ tại: checkpoints_cifar100/training_history.png")"""
nb['cells'].append(nbf.v4.new_code_cell(code5))

# Export & QAT
code6 = """# Trình diễn Xuất mô hình và Chuẩn bị Quantization Aware Training (QAT)
# 1. Chuyển mô hình sang CPU để chuẩn bị QAT
model.to('cpu')
model.train() # Yêu cầu của QAT là model phải ở chế độ train

try:
    print("--- Chuẩn bị mạng cho QAT ---")
    qat_model = Quantizer.prepare_qat(model)
    print("Mô hình đã sẵn sàng để Fine-tune thêm với QAT!")
    
    print("--- Thử nghiệm Xuất ONNX ---")
    dummy_input = torch.randn(1, 3, 224, 224)
    Exporter.export_onnx(model, dummy_input, 'checkpoints_cifar100/convnext_cifar100.onnx')
    print("Hoàn tất mọi thủ tục xuất!")
except Exception as e:
    print(f"Cảnh báo xuất: {e}")"""
nb['cells'].append(nbf.v4.new_code_cell(code6))

with open('advanced_cifar100.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Đã tạo file advanced_cifar100.ipynb")
