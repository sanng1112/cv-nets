# MỤC LỤC TÀI LIỆU CV-NETS TOÀN TẬP (MASTER DOCUMENTATION)

Bộ tài liệu này được biên soạn ở cấp độ cực kỳ chi tiết (Exhaustive Level). Trong này liệt kê **TOÀN BỘ** tất cả các hàm, class, layer, head, parameter hiện đang có mặt trong thư mục mã nguồn của CV-Nets. Không có một chi tiết nào bị bỏ sót.

Bạn có thể sử dụng bộ tài liệu này như một quyển Bách khoa toàn thư để đối chiếu mã nguồn Python với tham số YAML.

## Danh mục bài viết

1. **[01_QUICK_START_AND_YAML.md](./01_QUICK_START_AND_YAML.md):** 
   - Tổng quan về kiến trúc Config-Driven.
   - Hướng dẫn cơ chế Parse YAML sang `SimpleNamespace` (dot-notation).
   - Danh sách toàn tập các tham số cấu hình.

2. **[02_CORE_ENGINE_TRAINER.md](./02_CORE_ENGINE_TRAINER.md):** 
   - Khám phá lõi `BaseTrainer`.
   - Cơ chế hoạt động của AMP, Gradient Accumulation, DDP (Multi-GPU).
   - Thermal Throttling, EMA (Exponential Moving Average) và Logger.

3. **[03_MODEL_BUILDER_AND_HEADS.md](./03_MODEL_BUILDER_AND_HEADS.md):** 
   - Cơ chế Dummy Forward để quét kích thước tự động.
   - **Tài liệu toàn tập về 9 Custom Heads**: Classification, Detection (Decoupled), FCN (Segmentation), ArcFace, HeatmapKeypoint, ProtoMask (Instance Seg), PixelShuffle (Super Res), DepthEstimation, OpticalFlow. Tham số YAML tương ứng cho mỗi Head.

4. **[04_LAYERS_AND_BLOCKS.md](./04_LAYERS_AND_BLOCKS.md):** 
   - Giải phẫu thư mục `layers/`.
   - Cơ chế Opts Injection vào `BaseLayer`.
   - Tài liệu về `Conv2d`, `LinearLayer`, `Dropout`, `ETFClassifier`, `GCNLayer` (Graph Conv), `InformationBottleneck`, và các mô-đun Normalize/Activation/Pooling.

5. **[05_DATA_FACTORY_ADAPTERS.md](./05_DATA_FACTORY_ADAPTERS.md):** 
   - Cấu trúc thư mục `engine/adapters/`.
   - Phân tích chi tiết `CocoDetectionAdapter`, `ImageNetAdapter`.
   - Cơ chế Transform Factory và DistributedSampler dành cho DDP.

6. **[06_ADVANCED_FEATURES.md](./06_ADVANCED_FEATURES.md):** 
   - Mổ xẻ chi tiết thư mục `engine/` bao gồm:
   - `pruning.py`: Kỹ thuật cắt tỉa Weights Unstructured.
   - `quantization.py`: Phân biệt mã nguồn PTQ (Dynamic) và QAT (Prepare/Convert).
   - `exporter.py`: Code xuất ONNX chuẩn công nghiệp.
   - `sanity_check.py`: Mạch chống cháy nổ Model.
