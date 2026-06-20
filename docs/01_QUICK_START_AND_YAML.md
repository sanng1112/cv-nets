# 01. KHỞI ĐẦU NHANH & CƠ CHẾ YAML TOÀN TẬP

## 1. Cơ chế hoạt động từ Python đến YAML

Trong CV-Nets, chúng ta không khởi tạo hàm bằng cách truyền từng tham số dài ngoằng như `Model(layers=50, classes=10, drop=0.5)`. Thay vào đó, toàn bộ dữ liệu đi qua một cơ chế gọi là **Opts Injection**.

Trong `main.py`, khi bạn truyền file `config.yaml`, nó sẽ được đọc và biến đổi nhờ hàm:
```python
def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d
```
Hàm này đệ quy biến một Dictionary thành một Object (Ký hiệu là biến `opts`). Thay vì phải gọi `cfg['model']['name']`, bạn có thể gọi cực kỳ thanh lịch: `opts.model.name`. Mọi module trong hệ thống (từ Trainer đến Layer bé nhất) đều nhận duy nhất một biến `opts` này và tự động lấy ra những tham số nó cần.

---

## 2. Danh Sách Toàn Bộ Các Tham Số YAML Hiện Có

Dưới đây là từ điển liệt kê 100% các biến YAML đang được hỗ trợ trong phiên bản hiện tại, được ánh xạ (mapping) trực tiếp tới các file Python nào.

### A. Cấu Hình Bài Toán (Routing config)
- **`task_type`**: (String) Bắt buộc. Quyết định `CVNetModel` sẽ gọi file Python nào trong thư mục `models/heads/`.
  - Các giá trị hợp lệ: `"classification"`, `"detection"`, `"segmentation"`, `"metric_learning"`, `"keypoint"`, `"instance_segmentation"`, `"super_resolution"`, `"depth"`, `"optical_flow"`.
- **`model_name`**: (String) Bắt buộc. Tên Backbone để truy vấn vào thư viện `timm`.
  - Ví dụ: `"resnet18"`, `"convnext_tiny"`, `"mobilenetv3_large_100"`.
- **`pretrained`**: (Bool) Mặc định: `True`. Yêu cầu `timm` tải trọng số ImageNet.
- **`in_chans`**: (Int) Mặc định: `3`. Cấu hình số kênh vào (1 cho MNIST, 3 cho RGB).

### B. Cấu Hình Đầu Phân loại (Head Configs)
Khi `CVNetModel` gọi các Head, nó sẽ giải nén biến `head_kwargs`. Các tham số này bạn có thể đặt trực tiếp vào YAML:
- **`num_classes`**: (Int). Số nhãn đầu ra. (Dùng cho `ClassificationHead`, `DecoupledDetectionHead`, `FCNHead`, `ArcFaceHead`).
- **Ngoài ra**: Mỗi Head còn nhận các tham số siêu chi tiết (sẽ được trình bày kỹ ở file `03_MODEL_BUILDER_AND_HEADS.md`).

### C. Cấu Hình Dữ Liệu (Data Factory)
Nằm trong file `engine/data_factory.py`:
- **`dataset_name`**: (String). Gọi Adapter. Hợp lệ: `"mnist"`, `"cifar10"`, `"cifar100"`, `"image_folder"`, `"coco"`, `"imagenet"`.
- **`dataset_root`**: (String). Đường dẫn gốc tới thư mục Dataset (Dùng cho `image_folder`, `coco`, `imagenet`).
- **`batch_size`**: (Int). Mặc định `64`.
- **`img_size`**: (Int). Mặc định `224`. Kích thước biến đổi (Resize).
- **`num_workers`**: (Int). Mặc định `2`. Số luồng nạp Data vào RAM.

### D. Cấu Hình Huấn Luyện (Trainer Engine)
Nằm trong file `engine/trainer.py` và `main.py`:
- **`loss_name`**: (String). Hợp lệ: `"cross_entropy"`. 
- **`optim_name`**: (String). Hợp lệ: `"adamw"`, `"sgd"`.
- **`lr`**: (Float). Mặc định `0.001`.
- **`epochs`**: (Int). Mặc định `10`.
- **`seed`**: (Int). Mặc định `42`. Dùng cho `utils.seed_everything` khóa tính ngẫu nhiên.
- **`accumulation_steps`**: (Int). Mặc định `1`. Cộng dồn Loss trước khi Backward.
- **`epoch_sleep_sec`**: (Float). Mặc định `0.0`. Chống quá nhiệt (Thermal Throttling) sau mỗi vòng lặp.

---

## 3. Cách Gọi Lệnh Chạy (CLI)
Mã nguồn: `main.py`
```bash
python main.py --config path/to/your_config.yaml
```
Khi chạy lệnh này:
1. `argparse` sẽ nhận cờ `--config`.
2. Trạm điều phối gọi `seed_everything()`.
3. Trạm điều phối gọi `build_dataloaders_from_yaml(opts)`.
4. Trạm điều phối gọi `CVNetModel.build_from_yaml(opts)`.
5. Đẩy Optimizer, Criterion và Model vào `BaseTrainer(opts)`.
6. Gọi `trainer.train()` để kích hoạt vòng lặp!
