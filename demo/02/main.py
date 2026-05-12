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


class AddGaussianNoise(object):
    """
    Thêm nhiễu Gaussian vào Tensor.
    """
    def __init__(self, mean=0.0, std=0.15):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        # Tạo nhiễu cùng kích thước với tensor
        noise = torch.randn_like(tensor) * self.std + self.mean
        # Cộng nhiễu và kẹp giá trị về dải [0.0, 1.0] để tránh pixel bị sai màu
        return torch.clamp(tensor + noise, 0.0, 1.0)

transform = transforms.Compose([
   transforms.RandomAffine(
        degrees=10,               # Xoay ngẫu nhiên từ -10 đến 10 độ
        translate=(0.1, 0.1),     # Dịch chuyển lên/xuống/trái/phải tối đa 10% (khoảng 2-3 pixel)
        scale=(0.9, 1.1),         # Thu nhỏ hoặc phóng to ngẫu nhiên từ 90% đến 110%
        fill=0                    # Các khoảng trống sinh ra khi dịch chuyển sẽ được điền màu đen (0)
    ),
    transforms.ToTensor(),
    transforms.RandomApply(
        [AddGaussianNoise(mean=0.0, std=0.15)], 
        p=0.5 
    )
])
dataset = datasets.MNIST(root = 'data', train = True, transform=transform)
dataloader = DataLoader(
    dataset,
    batch_size = 64,
    num_workers =  8,
    shuffle = True
)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs =5


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0  
    total = 0    
    
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
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
        
        current_acc = 100. * correct / total
        loop.set_postfix(loss=loss.item(), acc=f"{current_acc:.2f}%")


    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100. * correct / total
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

model.save()