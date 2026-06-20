import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from types import SimpleNamespace
from models.builder import CVNetModel

def main():
    print("[*] Bắt đầu script huấn luyện đơn giản...")
    
    # 1. Cấu hình cơ bản giả lập
    # Không cần file YAML phức tạp, chỉ cần Namespace cơ bản
    opts = SimpleNamespace(
        model=SimpleNamespace(name="resnet18", num_classes=10)
    )
    
    # 2. Chuẩn bị dữ liệu (CIFAR-10)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    print("[*] Đang tải CIFAR-10 dataset...")
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    
    # 3. Khởi tạo Mô hình
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Sử dụng thiết bị: {device}")
    model = CVNetModel.build_from_yaml(opts).to(device)
    
    # 4. Định nghĩa Loss và Optimizer cơ bản
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 5. Vòng lặp huấn luyện đơn giản
    epochs = 2
    print("[*] Bắt đầu huấn luyện...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:    # In ra sau mỗi 100 mini-batches
                print(f'[Epoch: {epoch + 1}, Batch: {i + 1}] Loss: {running_loss / 100:.3f}')
                running_loss = 0.0
                
    print("[*] Đã hoàn thành huấn luyện cơ bản!")

if __name__ == '__main__':
    main()
