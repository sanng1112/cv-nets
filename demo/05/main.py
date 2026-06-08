import yaml
from types import SimpleNamespace
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Lưu ý: Import từ models của bạn
from models import * 


# ==========================================
# 1. HÀM TIỆN ÍCH VÀ CẤU HÌNH
# ==========================================
def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{
            k: dict_to_namespace(v)
            for k, v in d.items()
        })
    elif isinstance(d, list):  
        return [dict_to_namespace(v) for v in d]
    return d

CONFIG = 'config.yaml'

with open(CONFIG, "r") as f:
    cfg = yaml.safe_load(f)

opts = dict_to_namespace(cfg)
print(opts)

# Khởi tạo mô hình
model = CNN(opts=opts)
print(model)
model.save()

# ==========================================
# 2. XỬ LÝ DỮ LIỆU & DATASET
# ==========================================
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.15):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)), # Đưa về dải [-1, 1]
    transforms.RandomApply([AddGaussianNoise(0.0, 0.05)], p=0.5) 
])

train_dataset = datasets.ImageFolder(
    root='./data/train', 
    transform=transform
)
test_dataset = datasets.ImageFolder(
    root='./data/test', 
    transform=transform
)

# ==========================================
# 3. CÁC HÀM CẬP NHẬT ĐỘNG (BATCH SIZE & LR)
# ==========================================
def update_dataloader(dataset, current_batch_size, scale=1.2):
    """Tính toán batch size mới và khởi tạo lại DataLoader"""
    new_batch_size = int(current_batch_size * scale)
    print(f"\n🚀 [CẬP NHẬT] Batch size tăng từ {current_batch_size} lên {new_batch_size}")
    
    new_loader = DataLoader(
        dataset,
        batch_size=new_batch_size,
        num_workers=16, 
        shuffle=True,
        pin_memory=True
    )
    return new_loader, new_batch_size

def update_learning_rate(optimizer, decrement=1e-4, min_lr=1e-5):
    """Giảm learning rate đi một lượng cố định"""
    for param_group in optimizer.param_groups:
        old_lr = param_group['lr']
        new_lr = max(old_lr - decrement, min_lr) # Đảm bảo không bị âm
        param_group['lr'] = new_lr
        print(f"📉 [CẬP NHẬT] Learning Rate giảm từ {old_lr:.6f} xuống {new_lr:.6f}")

# ==========================================
# 4. KHỞI TẠO DATALOADER & TRAINING CƠ BẢN
# ==========================================
current_batch_size = 128
train_loader = DataLoader(
    train_dataset,
    batch_size=current_batch_size,
    num_workers=16, 
    shuffle=True,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    num_workers=8, 
)

data_iter = iter(train_loader)
images, labels = next(data_iter)
print(f"Kích thước tensor ảnh: {images.shape}")
print(f"Số lượng nhãn: {len(train_dataset.classes)}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-3)
num_epochs = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion.to(device)

# ==========================================
# 5. VÒNG LẶP HUẤN LUYỆN CHÍNH
# ==========================================
best_val_acc = None  # Cột mốc để kiểm tra xem đã tăng 10% chưa

for epoch in range(num_epochs):
    # --- TRAINING ---
    model.train()
    running_loss = 0.0
    correct = 0  
    total = 0    
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        running_loss += loss.item() * images.size(0)
        loop.set_postfix(loss=loss.item())

    # --- VALIDATION (TEST) ---
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    val_loop = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [TEST]")
    
    with torch.no_grad():
        for images, labels in val_loop:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            val_loop.set_postfix(loss=loss.item(), acc=100.*val_correct/val_total)

    epoch_val_loss = val_loss / len(test_dataset)
    epoch_val_acc = 100. * val_correct / val_total

    # --- SUMMARY CỦA EPOCH ---
    print(f"\nSummary Epoch {epoch+1}:")
    if best_val_acc is not None:
        print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}% (Mốc hiện tại: {best_val_acc:.2f}%)")
    else:
        print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
    print("-" * 40)

    # --- ĐIỀU KIỆN CẬP NHẬT ĐỘNG ---
    if best_val_acc is None:
        # Lấy epoch đầu tiên làm mốc cơ sở
        best_val_acc = epoch_val_acc
    elif epoch_val_acc >= best_val_acc + 10.0:
        print(f"🔥 Tuyệt vời! Val Acc tăng vượt 10% (Từ {best_val_acc:.2f}% lên {epoch_val_acc:.2f}%)")
        
        # 1. Cập nhật mốc mới để theo dõi cho lần tiếp theo
        best_val_acc = epoch_val_acc
        
        # 2. Tăng Batch Size lên 20%
        train_loader, current_batch_size = update_dataloader(
            train_dataset, current_batch_size, scale=1.2
        )
        
        # 3. Giảm Learning Rate đi 10^-4
        update_learning_rate(optimizer, decrement=1e-4)

# Lưu model cuối cùng sau khi train xong 100 epochs
model.save()
print("\nHoàn tất huấn luyện và đã lưu mô hình!")