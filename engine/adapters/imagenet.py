import os
from torchvision import datasets
from engine.adapters.base_adapter import BaseDatasetAdapter

class ImageNetAdapter(BaseDatasetAdapter):
    """
    Adapter chuyên biệt để đọc bộ dữ liệu ImageNet khổng lồ.
    """
    def __init__(self, root, split='train', transform=None):
        super().__init__(root, transform)
        split_dir = os.path.join(root, split)
        self.dataset = datasets.ImageFolder(split_dir, transform=transform)
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return {"image": image, "target": label}
