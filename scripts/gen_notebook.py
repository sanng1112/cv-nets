import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Markdown cell
text = """# Trình diễn End-to-End: Nhận dạng chữ số MNIST với CV-Nets
Mục tiêu: Xây dựng một mạng ResNet18 từ `timm`, gắn `ClassificationHead`, huấn luyện 1 Epoch, 
sau đó sử dụng `ModelPruner`, vẽ lịch sử bằng `HistoryPlotter`, và xuất mô hình ra chuẩn `ONNX` và `BF16`.
"""
nb['cells'].append(nbf.v4.new_markdown_cell(text))

# Code cell
code = """import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

# Import từ CV-Nets
from models.builder import CVNetModel
from engine.trainer import BaseTrainer
from engine.sanity_check import run_sanity_check
from engine.quantization import Quantizer
from engine.exporter import Exporter
from engine.pruning import ModelPruner
from utils.plotter import HistoryPlotter

# Thiết lập Hyperparameters
BATCH_SIZE = 128
EPOCHS = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Sử dụng thiết bị: {DEVICE}")"""
nb['cells'].append(nbf.v4.new_code_cell(code))

# Dataloader
code2 = """# Chuẩn bị dữ liệu MNIST (ảnh trắng đen 1 kênh màu)
transform = transforms.Compose([
    transforms.ToTensor(),
    # Ảnh MNIST nhỏ xíu, mạng timm ResNet thường cần ít nhất 3 kênh (RGB)
    # Ta sẽ dùng transforms.Lambda để lặp lại kênh ảnh 3 lần
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Dữ liệu mẫu
print(f"Số lượng ảnh Train: {len(train_dataset)}")"""
nb['cells'].append(nbf.v4.new_code_cell(code2))

# Model Builder
code3 = """# Khởi tạo mô hình thông qua Builder
# ResNet18 siêu mạnh sẽ được bẻ gãy Head mặc định và thay bằng Head phân loại 10 class.
model = CVNetModel(
    backbone_name='resnet18', 
    head_type='classification', 
    head_kwargs={'num_classes': 10}, 
    pretrained=False # Không cần pretrained cho MNIST để nhẹ nhàng
)
model = model.to(DEVICE)
print("Model Built Successfully!")"""
nb['cells'].append(nbf.v4.new_code_cell(code3))

# Trainer
code4 = """# Khởi tạo Criterion và Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Fake một object opts
class Opts:
    device = DEVICE
    use_amp = True
    accumulation_steps = 1
    epochs = EPOCHS
    task_type = "classification"
    num_classes = 10
    save_dir = "temp_checkpoints"
    eval_interval = 1
    clip_grad_norm = 1.0

# Khởi tạo Trainer
trainer = BaseTrainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=None,
    train_loader=train_loader,
    val_loader=val_loader,
    opts=Opts()
)

# Chạy Sanity Check trước khi Train
is_healthy = run_sanity_check(trainer)
if not is_healthy:
    print("Dừng huấn luyện vì Sanity Check phát hiện lỗi!")"""
nb['cells'].append(nbf.v4.new_code_cell(code4))

# Training
code5 = """# Huấn luyện
import json
if is_healthy:
    trainer.train()
    
    # Đọc file history.json và vẽ đồ thị
    with open("temp_checkpoints/history.json", "r") as f:
        history_list = json.load(f)
    
    # Convert list of dicts to dict of lists
    history_dict = {k: [d[k] for d in history_list] for k in history_list[0].keys()}
    
    # Đổi tên cho đúng với HistoryPlotter
    if 'train_MulticlassAccuracy' in history_dict:
        history_dict['train_metric'] = history_dict.pop('train_MulticlassAccuracy')
    if 'val_MulticlassAccuracy' in history_dict:
        history_dict['val_metric'] = history_dict.pop('val_MulticlassAccuracy')
        
    # Vẽ biểu đồ lịch sử Offline
    HistoryPlotter.plot_and_save(history_dict, save_dir='temp_checkpoints')
    print("Biểu đồ đã được lưu tại temp_checkpoints/training_history.png")"""
nb['cells'].append(nbf.v4.new_code_cell(code5))

# Pruning & Export
code6 = """# 1. Pruning (Cắt tỉa 30% trọng số rác)
print("--- PRUNING ---")
pruned_model = ModelPruner.prune_unstructured(model, amount=0.3)

# 2. Lượng tử hóa xuống BF16
print("--- QUANTIZATION ---")
bf16_model = Quantizer.to_bf16(pruned_model)

dummy_input = torch.randn(1, 3, 28, 28).to(DEVICE)
# Kiểm tra sức khỏe hậu lượng tử hóa
is_safe = Quantizer.check_quantization_health(bf16_model, dummy_input.to(torch.bfloat16))

# 3. Xuất ra ONNX
if is_safe:
    print("--- ONNX EXPORT ---")
    Exporter.export_onnx(pruned_model, dummy_input, 'temp_checkpoints/mnist_resnet18.onnx')
    print("HOÀN TẤT TOÀN BỘ PIPELINE!")"""
nb['cells'].append(nbf.v4.new_code_cell(code6))

with open('notebooks.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Đã tạo file notebooks.ipynb")
