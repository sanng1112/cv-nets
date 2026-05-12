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

model = MLP(opts = opts)
print(model)


transform = transforms.Compose([
    transforms.ToTensor(),              
    transforms.Lambda(lambda x: x.view(-1)) 
])
dataset = datasets.MNIST(root = 'data', train = True, transform=transform)
dataloader = DataLoader(
    dataset,
    batch_size = 64,
    num_workers =  8,
    shuffle = True
)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs =20


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

