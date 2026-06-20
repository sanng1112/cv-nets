# 03. MODEL BUILDER & CUSTOM HEADS TỪ A ĐẾN Z

Đây là bản hướng dẫn toàn diện nhất về 9 loại "Đầu quyết định" (Custom Heads) đang có sẵn trong CV-Nets.

## 1. Cơ Chế Tháo Lắp `CVNetModel` (Dummy Forward)
`CVNetModel` nằm tại `models/builder.py`. Nhiệm vụ của nó là nạp mạng Backbone từ thư viện `timm`, chặt đứt đầu của mạng đó, và cắm cái Custom Head bạn chọn vào.

**Làm sao nó biết đầu ra của ConvNeXt là 768 kênh, còn của ResNet là 512 kênh?**
Thay vì hard-code, nó tự động nội suy bằng cách nhét một "Bức ảnh rác" (Dummy Tensor) vào mạng:
```python
dummy_input = torch.randn(1, in_chans, 224, 224)
dummy_out = self.backbone(dummy_input)
in_channels = dummy_out.shape[1]
```
Nhờ `in_channels` đo được này, nó sẽ bơm vào cho các Custom Head phía dưới cấu hình khởi tạo.

---

## 2. Toàn Tập Về 9 Custom Heads

Tất cả nằm trong thư mục `models/heads/`. Để kích hoạt Head nào, bạn truyền tham số `task_type` tương ứng vào file YAML.

### 1. Classification Head (`task_type: "classification"`)
Dùng để Phân loại ảnh (Chó/Mèo).
- **Cấu trúc:** Nhận `in_features`, chạy qua Dropout (nếu có), đẩy vào một lớp Linear duy nhất xuất ra ma trận `(Batch, num_classes)`.
- **YAML Args:**
  - `num_classes`: Số lượng nhãn (Bắt buộc).
  - `dropout_rate`: (Tùy chọn) Tỷ lệ tắt Nơ-ron chống quá khớp.

### 2. Decoupled Detection Head (`task_type: "detection"`)
Dùng cho Nhận diện vật thể (Object Detection) với Bounding Box. Khái niệm "Decoupled" (Tách rời) được lấy từ siêu phẩm YOLO-X.
- **Cấu trúc:** Tách nhánh đặc trưng làm 2 luồng: 
  - Luồng 1 (Box Branch): Dự đoán tọa độ `[x, y, w, h]`.
  - Luồng 2 (Class Branch): Dự đoán Nhãn (Chó, Mèo).
- **YAML Args:**
  - `num_classes`: Số loại vật thể.
  - `num_anchors`: Số khung mỏ neo trên mỗi điểm ảnh (Mặc định: 3).

### 3. FCN Head - Segmentation (`task_type: "segmentation"`)
Fully Convolutional Network. Chuyên trị bài toán Phân vùng ảnh Y Yế hoặc Semantic Segmentation.
- **Cấu trúc:** Không dùng lớp Linear! Dùng `Conv2d` kết hợp `Dropout` để giữ nguyên chiều không gian 2D, xuất ra mặt nạ ảnh `(Batch, num_classes, H, W)`.
- **YAML Args:** `num_classes`.

### 4. ArcFace Head (`task_type: "metric_learning"`)
Chuyên dành cho Face Recognition (Chấm công bằng khuôn mặt). Thuật toán ArcFace ép khoảng cách Cosine giữa các mặt người giống nhau lại gần nhau.
- **Cấu trúc:** Tính Margin Cosine Penalty. Lớp tính Loss được thiết kế tích hợp ngay trong Head!
- **YAML Args:**
  - `num_classes`: Số nhân viên trong công ty.
  - `s`: Khẩu độ Scale (Mặc định: 64.0).
  - `m`: Angular Margin (Mặc định: 0.5).

### 5. Heatmap Keypoint Head (`task_type: "keypoint"`)
Dùng trong Pose Estimation (Ước lượng dáng người - Điểm khớp xương).
- **Cấu trúc:** Xuất ra bản đồ nhiệt (Heatmap) `(Batch, num_keypoints, H/4, W/4)`. Nơi nào sáng nhất trên bản đồ là vị trí của khớp xương.
- **YAML Args:**
  - `num_keypoints`: Số điểm khớp (COCO là 17 điểm).

### 6. Proto Mask Head (`task_type: "instance_segmentation"`)
Học hỏi từ kiến trúc YOLACT. Dùng khi bạn không chỉ muốn khoanh vùng cái xe, mà phải phân biệt được Xe số 1 và Xe số 2.
- **Cấu trúc:** Sinh ra k-Prototype Masks (Mặt nạ nguyên mẫu), sau đó nhân chập với Mask Coefficients từ nhánh Detection.
- **YAML Args:**
  - `num_prototypes`: Số mặt nạ cơ sở (Thường là 32).

### 7. Pixel Shuffle Head (`task_type: "super_resolution"`)
Dùng để Tăng nét ảnh (Zoom không vỡ) từ mờ lên 4K.
- **Cấu trúc:** Ứng dụng kỹ thuật `nn.PixelShuffle`. Giúp nới rộng chiều rộng/cao của ảnh (Upscale) bằng cách bòn rút số Channels.
- **YAML Args:**
  - `upscale_factor`: Hệ số phóng đại (VD: `4` -> Ảnh 64x64 biến thành 256x256).

### 8. Depth Estimation Head (`task_type: "depth"`)
Dự đoán Độ Sâu 3D (Z-axis) từ ảnh 2D camera thường. (Dùng cho Xe tự lái).
- **Cấu trúc:** Dùng chuỗi Conv2d + Upsample nội suy xuất ra một bức ảnh đen trắng 1 kênh. Màu trắng là gần, đen là xa.
- **YAML Args:** (Không cần, mặc định xuất 1 channel).

### 9. Optical Flow Head (`task_type: "optical_flow"`)
Đo đếm hướng di chuyển của từng điểm ảnh giữa 2 khung hình liên tiếp (Video).
- **Cấu trúc:** Xuất ra Tensor `(Batch, 2, H, W)` tương ứng với Vector vận tốc `(dx, dy)`.
- **YAML Args:** (Không cần, tự động cấu hình 2 channels).
