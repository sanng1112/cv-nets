import os
import torch
from torchvision import datasets
from engine.adapters.base_adapter import BaseDatasetAdapter

class CocoDetectionAdapter(BaseDatasetAdapter):
    """
    Adapter chuyên biệt để đọc bộ dữ liệu COCO (chứa bounding box và class).
    Xử lý tự động file annotations JSON phức tạp của COCO thành Format chuẩn.
    """
    def __init__(self, root, split='train2017', transform=None):
        super().__init__(root, transform)
        img_dir = os.path.join(root, split)
        ann_file = os.path.join(root, 'annotations', f'instances_{split}.json')
        
        # Cần thư viện pycocotools. Tạm thời sử dụng hàm của torchvision
        self.dataset = datasets.CocoDetection(img_dir, annFile=ann_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        
        # COCO trả về một list các boxes. Ta bóc tách thành format riêng cho Model
        boxes = []
        labels = []
        for obj in target:
            boxes.append(obj['bbox']) # [x, y, w, h]
            labels.append(obj['category_id'])
            
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        if self.transform:
            image = self.transform(image)
            
        return {"image": image, "target": {"boxes": boxes, "labels": labels}}
