from models import *
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{
            k: dict_to_namespace(v)
            for k, v in d.items()
        })
    return d

CONFIG = 'config.yaml'

with open(CONFIG, "r") as f:
    cfg = yaml.safe_load(f)

opts = dict_to_namespace(cfg)
print(opts)

model = CNN(opts = opts)
print(model)
model.save()

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
    transforms.RandomApply([AddGaussianNoise(0.0, 0.05)], p=0.5) # Giảm std nhiễu xuống 0.05
])
train_dataset = datasets.ImageFolder(
    root='./data/train', 
    transform=transform
)
test_dataset = datasets.ImageFolder(
    root='./data/test', 
    transform=transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=1024,
    num_workers=16, 
    shuffle = True,
    pin_memory= True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    num_workers=8, 
)


data_iter = iter(train_loader)
images, labels = next(data_iter)
print(f"Kích thước tensor ảnh: {images.shape}") # Nên ra: [64, 1, 48, 48]
print(f"Số lượng nhãn: {len(train_dataset.classes)}") # Nên ra: 7

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-3)
num_epochs =400


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion.to(device)

for epoch in range(num_epochs):
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

    # --- IN KẾT QUẢ CUỐI EPOCH ---
    print(f"\nSummary Epoch {epoch+1}:")
    print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
    print("-" * 30)

model.save()