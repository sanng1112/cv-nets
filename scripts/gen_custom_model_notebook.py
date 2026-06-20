import nbformat as nbf

nb = nbf.v4.new_notebook()

text = """# Trình diễn Custom Model với CV-Nets Layers
Thay vì dùng mạng có sẵn từ timm, ta sẽ tự lắp ráp một mạng nơ-ron hoàn chỉnh bằng các khối (blocks) nội bộ do CV-Nets cung cấp: `Conv2d`, `LinearLayer`...
"""
nb['cells'].append(nbf.v4.new_markdown_cell(text))

# Code Cell
code = """import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

# Import các module nhà làm từ cvnets
from layers.conv_layer import Conv2d
from layers.linear_layer import LinearLayer
from engine.trainer import BaseTrainer
from engine.sanity_check import run_sanity_check
from engine.quantization import Quantizer
from engine.exporter import Exporter
from engine.pruning import ModelPruner
from utils.plotter import HistoryPlotter

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Định nghĩa Custom Model
class CustomMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Sử dụng Conv2d của CV-Nets
            Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            # Sử dụng LinearLayer của CV-Nets
            LinearLayer(in_features=32 * 7 * 7, out_features=128),
            nn.ReLU(),
            LinearLayer(in_features=128, out_features=10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = CustomMNISTNet().to(DEVICE)
print("Đã tạo mạng Custom thành công từ các Layers của CV-Nets!")"""
nb['cells'].append(nbf.v4.new_code_cell(code))

code2 = """# 2. Dataset MNIST 1 Kênh 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)"""
nb['cells'].append(nbf.v4.new_code_cell(code2))

code3 = """# 3. Khởi tạo BaseTrainer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

class Opts:
    device = DEVICE
    use_amp = True
    accumulation_steps = 1
    epochs = 2
    task_type = "classification"
    num_classes = 10
    save_dir = "checkpoints_custom"
    eval_interval = 1
    clip_grad_norm = 1.0
    batch_sleep_sec = 0.0
    epoch_sleep_sec = 0.0

trainer = BaseTrainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=None,
    train_loader=train_loader,
    val_loader=val_loader,
    opts=Opts()
)

print("Đang chạy Sanity Checks...")
is_healthy = run_sanity_check(trainer)

if is_healthy:
    print("Bắt đầu huấn luyện Custom Model...")
    trainer.train()
else:
    print("Dừng huấn luyện do lỗi Sanity Check!")"""
nb['cells'].append(nbf.v4.new_code_cell(code3))

code4 = """# 4. Kiểm tra Lịch sử vẽ biểu đồ
import json
with open("checkpoints_custom/history.json", "r") as f:
    history_list = json.load(f)
history_dict = {k: [d[k] for d in history_list] for k in history_list[0].keys()}

HistoryPlotter.plot_and_save(history_dict, save_dir='checkpoints_custom')
print("Vẽ biểu đồ thành công! Xem file tại checkpoints_custom/training_history.png")"""
nb['cells'].append(nbf.v4.new_code_cell(code4))

code5 = """# 5. Cắt gọt, Lượng tử hóa và Xuất xưởng
print("--- PRUNING ---")
pruned_model = ModelPruner.prune_unstructured(model, amount=0.2)

model.to('cpu')
model.eval()

print("--- QUANTIZATION (PTQ) ---")
int8_model = Quantizer.to_int8_dynamic(pruned_model)

print("--- ONNX EXPORT ---")
dummy_input = torch.randn(1, 1, 28, 28)
try:
    Exporter.export_onnx(pruned_model, dummy_input, 'checkpoints_custom/custom_mnist.onnx')
    print("Hoàn tất xuất ONNX!")
except Exception as e:
    print(f"Lỗi xuất ONNX: {e}")"""
nb['cells'].append(nbf.v4.new_code_cell(code5))

with open("custom_model.ipynb", "w", encoding='utf-8') as f:
    nbf.write(nb, f)
