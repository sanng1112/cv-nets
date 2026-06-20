from torch.utils.data import Dataset

class BaseDatasetAdapter(Dataset):
    """
    Khuôn mẫu chung (Interface) cho mọi bộ đọc dữ liệu.
    Các Adapter con (như COCO, VOC, ImageNet) phải kế thừa và trả về đúng chuẩn.
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        
    def __len__(self):
        raise NotImplementedError
        
    def __getitem__(self, idx):
        # Bắt buộc trả về format: {"image": tensor, "target": target}
        raise NotImplementedError
