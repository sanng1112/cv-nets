# 04. THƯ VIỆN LINH KIỆN ĐIỆN TỬ (LAYERS & BLOCKS)

Thư mục `layers/` chứa những nguyên tử nhỏ nhất cấu tạo nên một mạng Nơ-ron. Tại sao CV-Nets lại cấm việc gọi trực tiếp `torch.nn` mà bắt buộc phải wrap lại?

## 1. Base Layer (`layers/base_layer.py`)
Mọi class trong `layers/` đều kế thừa từ `BaseLayer`.
```python
class BaseLayer(nn.Module):
    def update_opts(self, opts): ...
```
Tính năng này giúp mọi lớp ẩn sâu 10 tầng dưới lòng đất (Deep in Graph) vẫn có thể thò tay lấy cấu hình từ file YAML thông qua biến `opts` mà không cần phải truyền biến trung gian qua 10 hàm `__init__`.

---

## 2. Toàn Tập Các Layers CV-Nets

### 1. Custom Conv2d (`layers/conv_layer.py`)
Wrap của `nn.Conv2d`.
- **Sự khác biệt:** Khi khởi tạo, nó tự động gọi hàm khởi tạo trọng số `Kaiming Normal` (Dành cho ReLU) theo He et al. Nếu bạn dùng chuẩn cũ, nó gọi `Xavier Normal`. Bạn không cần bao giờ lo về hiện tượng "Vanishing Gradient" (mất não học) do tạ quá nhỏ.

### 2. Custom Linear (`layers/linear_layer.py`)
Wrap của `nn.Linear` (Lớp kết nối chéo Toàn cục).
- **Sự khác biệt:** Cung cấp tham số `linear_init="kaiming_normal"`. Hỗ trợ ép cứng (Freeze) lớp Linear nếu bạn đang mổ xẻ Finetuning mô hình.

### 3. ETF Classifier (`layers/etf_classifier.py`)
ETF (Equiangular Tight Frame) là một kỹ thuật Research tối tân (2023).
- **Nguyên lý:** Thay vì để khối Linear cuối cùng tự học trọng số ngẫu nhiên (dễ bị thiên vị - Imbalanced Bias), ETF "cắm sẵn" các điểm neo (Anchor) tạo thành một khối đa giác đều hoàn hảo trong không gian N-chiều. Lớp này không có trọng số học được (Requires_grad = False). Ép mạng Nơ-ron phải đẩy các Features về đúng các góc của đa giác. Giúp cực kỳ tối ưu cho Dữ liệu mất cân bằng (Imbalanced Data).

### 4. GCN Layer - Đồ Thị (`layers/gcn_layer.py`)
Graph Convolutional Network. Dành cho bài toán nhận diện Phân tử Hóa học, hoặc Mạng xã hội.
- **Tính năng:** Nó nhân ma trận Đặc trưng `X` với Ma trận kề `A` (Adjacency Matrix) để các Node kế cận truyền thông tin cho nhau. Tích hợp sẵn chuẩn hóa Ma trận kề bằng $D^{-1/2}AD^{-1/2}$.

### 5. Information Bottleneck (`layers/information_bottleneck.py`)
VIB (Variational Information Bottleneck). Kỹ thuật Research hẹp dành cho Regularization.
- **Tính năng:** Ép đặc trưng đi qua một cái cổ chai hình Chuông (Gaussian Noise) bằng cách ép học $\mu$ (Mean) và $\sigma$ (Variance) qua cơ chế Reparameterization Trick. Khiến Model không thể học vẹt được ảnh gốc.

### 6. Activation & Normalization
Trong `layers/activation/` và `layers/normalization/`.
- Hỗ trợ gọi nhanh qua chuỗi String trong YAML.
- Hỗ trợ: `relu`, `gelu`, `silu`, `swish`...
- Hỗ trợ Normalization: `batch_norm`, `layer_norm`, `instance_norm`. Tự động áp dụng chuẩn hóa số chiều theo Cảnh báo của PyTorch.

### 7. Dropout Thông Minh (`layers/dropout.py`)
Ngoài Dropout truyền thống (tắt ngẫu nhiên Nơ-ron), hỗ trợ **DropPath** (Tắt ngẫu nhiên nguyên một nhánh ResNet) - Rất cần thiết cho các mạng họ Transformer và ConvNeXt.
