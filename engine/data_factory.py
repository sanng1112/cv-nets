import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def build_dataloaders_from_yaml(opts):
    """
    [Data Factory]: Trạm nạp Dữ liệu Tự động.
    Phân tích opts (YAML) và tự động lắp ráp Transform, Dataset và Dataloader.
    Hỗ trợ DDP (Multi-GPU) một cách trong suốt.
    """
    dataset_name = getattr(opts, "dataset_name", "mnist").lower()
    batch_size = getattr(opts, "batch_size", 64)
    num_workers = getattr(opts, "num_workers", 2)
    img_size = getattr(opts, "img_size", 224)
    
    # 1. Transform Factory (Xử lý ảnh chuẩn ImageNet)
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Ngoại lệ cho MNIST (1 kênh màu, size nhỏ)
    if dataset_name == 'mnist':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        val_transform = train_transform
        
    # 2. Dataset Factory
    print(f"[Data Factory] Đang khởi tạo bộ dữ liệu: {dataset_name.upper()}...")
    if dataset_name == 'mnist':
        train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
        val_ds = datasets.MNIST(root='./data', train=False, download=True, transform=val_transform)
    elif dataset_name == 'cifar100':
        train_ds = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        val_ds = datasets.CIFAR100(root='./data', train=False, download=True, transform=val_transform)
    elif dataset_name == 'cifar10':
        train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        val_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
    elif dataset_name == 'image_folder':
        data_root = getattr(opts, "dataset_root", "./data/custom")
        train_ds = datasets.ImageFolder(root=f"{data_root}/train", transform=train_transform)
        val_ds = datasets.ImageFolder(root=f"{data_root}/val", transform=val_transform)
    elif dataset_name == 'coco':
        from engine.adapters.coco import CocoDetectionAdapter
        data_root = getattr(opts, "dataset_root", "./data/coco")
        train_ds = CocoDetectionAdapter(root=data_root, split='train2017', transform=train_transform)
        val_ds = CocoDetectionAdapter(root=data_root, split='val2017', transform=val_transform)
    elif dataset_name == 'imagenet':
        from engine.adapters.imagenet import ImageNetAdapter
        data_root = getattr(opts, "dataset_root", "./data/imagenet")
        train_ds = ImageNetAdapter(root=data_root, split='train', transform=train_transform)
        val_ds = ImageNetAdapter(root=data_root, split='val', transform=val_transform)
    else:
        raise ValueError(f"Dataset '{dataset_name}' chưa được hỗ trợ trong Data Factory.")
        
    # 3. Dataloader Factory & DDP Support
    import torch.distributed as dist
    is_ddp = dist.is_available() and dist.is_initialized()
    
    if is_ddp:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=True)
        print(f"[Data Factory] Đã gắn DistributedSampler cho cấu hình Multi-GPU (DDP).")
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        
    return train_loader, val_loader
