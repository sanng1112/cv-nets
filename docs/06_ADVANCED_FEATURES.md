# 06. TÍNH NĂNG CAO CẤP (ADVANCED FEATURES)

Thư mục `engine/` chứa 4 vũ khí tuyệt mật giúp Framework này đạt đẳng cấp Production (Triển khai công nghiệp). Dưới đây là phân tích chi tiết Mã nguồn.

---

## 1. Cắt Tỉa Nơ-ron (`engine/pruning.py`)

Chứa class `ModelPruner`. Sử dụng module `torch.nn.utils.prune`.
```python
@staticmethod
def prune_unstructured(model, amount=0.2):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Lệnh l1_unstructured tìm các trọng số (weights) có trị tuyệt đối
            # gần số 0 nhất, và ép chúng thành 0 vĩnh viễn.
            prune.l1_unstructured(module, name='weight', amount=amount)
            # Tháo gỡ các liên kết ảo, làm nhẹ file Model
            prune.remove(module, 'weight')
    return model
```
**Cách dùng:** Chạy trước khi Lưu (Save) hoặc sau khi Train. Giảm khối lượng tính toán mà không làm giảm Loss đáng kể. Cực kì quan trọng để nhúng AI vào Camera AI.

---

## 2. Ép Kiểu Lượng Tử Hóa (`engine/quantization.py`)

Chứa class `Quantizer`. Biến Model từ số thực 32-bit (FP32) khổng lồ xuống số nguyên 8-bit (INT8) siêu nhẹ. Thường giảm 4 lần dung lượng RAM tiêu thụ.

### Kỹ thuật PTQ (Post-Training Quantization - Lượng tử hóa Động)
```python
@staticmethod
def to_int8_dynamic(model):
    # Dynamic Quantization chỉ tác dụng mạnh nhất trên lớp Linear.
    # Thích hợp cho mạng NLP hoặc mạng phân loại dày đặc.
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model
```
**Ưu điểm:** Ép rất nhanh (3 giây) sau khi Train xong.
**Nhược điểm:** Mất mát độ chính xác cao.

### Kỹ thuật QAT (Quantization-Aware Training - Ép trong lúc học)
```python
@staticmethod
def prepare_qat(model):
    # Cắm các nốt ruồi Cảm biến sai số (FakeQuantize) vào toàn bộ Graph
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)
    return model

@staticmethod
def convert_qat(model):
    # Sau khi chạy qua Trainer vài Epoch, gọi hàm này để kết liễu
    # Biến đổi vĩnh viễn sang Model INT8
    torch.quantization.convert(model.eval(), inplace=True)
    return model
```
**Ưu điểm:** Nhờ học được sai số trong quá trình train, mô hình xuất ra giữ được 99% độ chính xác của bản gốc.

---

## 3. Xuất Xưởng Đa Nền Tảng (`engine/exporter.py`)

Class `Exporter`. Chứa lệnh xuất `.pt` sang `.onnx`.
```python
@staticmethod
def export_onnx(model, input_size=(1, 3, 224, 224), save_path="model.onnx"):
    model.eval()
    dummy_input = torch.randn(*input_size) # Nhét 1 ảnh giả vào
    torch.onnx.export(
        model, dummy_input, save_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True, # Tối ưu hóa, dồn các hằng số lại
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
```
**Công dụng:** Biến model PyTorch thành chuẩn đồ thị chung ONNX. Cờ `dynamic_axes` rất thông minh, giúp bạn xuất 1 cái model, nhưng khi chạy thực tế (Inference) bạn có thể nhét 1 ảnh, hoặc 10 ảnh vào cùng lúc đều không bị lỗi.

---

## 4. Cảm Biến Chống Cháy Nổ (`engine/sanity_check.py`)

Chứa hàm `run_sanity_check(model, input_size, device)`.
- **Luồng chạy:** Nó sinh ra ảnh ngẫu nhiên, bơm qua model lấy `output`, tạo một Loss ngẫu nhiên `loss = output.sum()`, và gọi `loss.backward()`.
- **Tính năng:**
  1. Kiểm tra Lỗi văng RAM (Out of Memory).
  2. Bắt lỗi Chia cho 0 (NaN/Inf Loss).
  3. Bắt lỗi "Thiếu Gradient" (Feature Collapse) nếu có Layer nào bạn lỡ set `requires_grad=False` mà không biết.

Nó là tấm khiên bảo vệ tuyệt đối thời gian bạc vàng của bạn khỏi những sơ suất cấu hình ngớ ngẩn.
