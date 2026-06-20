# 05. DATA FACTORY VÀ BỘ ĐỌC DỮ LIỆU (ADAPTERS)

## 1. Data Factory (`engine/data_factory.py`)

Trong file YAML, bạn cấu hình khối Dữ liệu qua các tham số:
```yaml
dataset_name: "coco"
dataset_root: "./data/coco"
batch_size: 16
img_size: 512
num_workers: 4
```

Khi chạy `main.py`, hàm `build_dataloaders_from_yaml` sẽ được kích hoạt.
1. **Transform Factory:** Đầu tiên nó sẽ nạp một chuỗi ống nước `transforms.Compose`. Cắt ảnh về `512x512`, trừ đi Mean và chia Std theo hệ số của ImageNet.
2. **Dataset Routing:** Tùy vào chữ `"coco"` hay `"imagenet"`, nó sẽ `import` class Adapter tương ứng từ thư mục `engine/adapters/`.
3. **Dataloader Assembling:** Bước quan trọng nhất!
   - Nó kiểm tra xem `torch.distributed.is_initialized()` có = True hay không?
   - Nếu True (Tức là máy bạn đang bật chế độ DDP nhiều Card). Thay vì load ảnh tuần tự, nó gán vào `DistributedSampler`. Sampler này dùng Toán rời rạc để đảm bảo Card 0 đọc ảnh chẵn, Card 1 đọc ảnh lẻ. Cực kỳ tối ưu và không bao giờ xảy ra Race Condition.
   - Trả về `train_loader` và `val_loader`.

---

## 2. Thư mục Trạm Chuyển Đổi `engine/adapters/`

Đây là kiệt tác kiến trúc của hệ thống đọc dữ liệu. Tại sao lại cần Adapter?

Giả sử `BaseTrainer` của chúng ta là ổ cắm điện 220V. Nó yêu cầu Format dữ liệu truyền vào bắt buộc phải là một Dictionary:
```python
# Format của CV-Nets
return {
    "image": Tensor[3, 224, 224], 
    "target": Label_Hay_Bbox
}
```

Tuy nhiên, `CocoDataset` của PyTorch lại nôn ra 1 cái List rườm rà. ImageNet lại nôn ra 1 cái Tuple. Để `BaseTrainer` không bị chập mạch, chúng ta xây dựng các Adapter (Cục sạc chuyển đổi 110V sang 220V).

### A. BaseDatasetAdapter (`engine/adapters/base_adapter.py`)
Class móng nhà. Bắt buộc mọi Adapter con phải có hàm `__getitem__` trả về Format Dictionary chuẩn của CV-Nets.

### B. CocoDetectionAdapter (`engine/adapters/coco.py`)
Mổ xẻ file JSON của COCO.
```python
class CocoDetectionAdapter(BaseDatasetAdapter):
    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        
        # COCO trả về một List các Dictionary. Ta dùng Vòng lặp For để bóc tách.
        boxes = []
        labels = []
        for obj in target:
            boxes.append(obj['bbox']) # Tọa độ x, y, width, height
            labels.append(obj['category_id']) # Nhãn (VD: 1 là Người, 2 là Xe)
            
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        # Ép khung trả về chuẩn của CV-Nets
        return {"image": image, "target": {"boxes": boxes, "labels": labels}}
```

### C. Cách mở rộng dữ liệu cho riêng bạn (Custom Adapter)
Nếu công ty bạn làm dữ liệu Y Tế dạng file `XML`. Bạn tạo file `medical.py`:
1. Class `MedicalAdapter(BaseDatasetAdapter)`.
2. Dùng thư viện `xml.etree` để đọc file.
3. Nhét vào Dictionary trả về.
4. Mở file `data_factory.py`, thêm 2 dòng `elif dataset_name == 'medical':` để import class vừa tạo. XONG!
Mọi thao tác DDP, Transform, Huấn luyện phía sau tự động tương thích 100%. Mức độ Decoupled (Tách rời) tuyệt đỉnh!
