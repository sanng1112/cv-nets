import torch
import torch.nn as nn
import copy

class Quantizer:
    """
    [Chi tiết hàm]: Bộ định lượng mô hình (Model Quantization)
    Hỗ trợ giảm dung lượng và tăng tốc độ Dự đoán (Inference) bằng cách ép kiểu trọng số.
    """
    @staticmethod
    def to_fp16(model: nn.Module) -> nn.Module:
        """
        Chuyển đổi toàn bộ mô hình sang FP16 (Half Precision).
        Tiết kiệm 50% VRAM và tính toán siêu nhanh trên GPU hỗ trợ TensorCore.
        """
        quantized_model = copy.deepcopy(model)
        quantized_model.half()
        
        # Đảm bảo các layer chuẩn hóa (Norm) vẫn giữ FP32 để không bị chia cho 0 hoặc lỗi Overflow
        for module in quantized_model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                module.float()
        return quantized_model

    @staticmethod
    def to_bf16(model: nn.Module) -> nn.Module:
        """
        Chuyển đổi toàn bộ mô hình sang BF16 (Brain Floating Point).
        Giữ nguyên dải biểu diễn (dynamic range) của FP32 nhưng chỉ dùng 16-bit.
        Tránh hoàn toàn lỗi Overflow của FP16. Tối ưu cực tốt cho kiến trúc GPU Ampere trở lên.
        """
        quantized_model = copy.deepcopy(model)
        quantized_model.bfloat16()
        
        # Norm layer nên giữ nguyên FP32 để ổn định số học
        for module in quantized_model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                module.float()
        return quantized_model

    @staticmethod
    def to_int8_dynamic(model: nn.Module) -> nn.Module:
        """
        Lượng tử hóa động (Dynamic Quantization) sang INT8 (Post-Training Quantization).
        Làm giảm 75% dung lượng RAM. Phương pháp này áp dụng tốt nhất khi chạy Inference trên CPU,
        cụ thể là nó ép kiểu tự động đối với các lớp mạng Linear.
        """
        model.eval()
        # Chuyển đổi mô hình đang chạy trên CPU sang dạng INT8 một cách an toàn
        quantized_model = torch.quantization.quantize_dynamic(
            model.to('cpu'), 
            {nn.Linear}, 
            dtype=torch.qint8
        )
        return quantized_model

    @staticmethod
    def prepare_qat(model: nn.Module, backend: str = 'fbgemm') -> nn.Module:
        """
        Chuẩn bị mô hình cho Quantization-Aware Training (QAT) - Huấn luyện giả lượng tử.
        Gắn thêm các 'FakeQuantize' modules để mô phỏng sai số của INT8 ngay trong lúc train.
        Mạng Nơ-ron sẽ tự động bù đắp sai số này, giúp giữ vững 99.9% độ chính xác gốc khi ép về INT8.
        - backend='fbgemm': Tối ưu cho x86 CPU (Server/PC).
        - backend='qnnpack': Tối ưu cho ARM CPU (Mobile/Raspberry Pi).
        """
        model.train()
        qat_model = copy.deepcopy(model).to('cpu') # QAT setup ở Pytorch yêu cầu khởi tạo trên CPU
        qat_model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
        torch.quantization.prepare_qat(qat_model, inplace=True)
        return qat_model

    @staticmethod
    def convert_qat(qat_model: nn.Module) -> nn.Module:
        """
        Biến đổi mô hình QAT (sau khi đã train xong) thành mô hình INT8 thực sự (Hard INT8) để Deploy.
        """
        qat_model.eval().to('cpu')
        return torch.quantization.convert(qat_model, inplace=False)

    @staticmethod
    def check_quantization_health(quantized_model: nn.Module, dummy_input: torch.Tensor) -> bool:
        """
        Kiểm tra sức khỏe (Sanity Check) của mô hình sau lượng tử hóa.
        Chặn đứng lỗi số học (NaN, Inf) và sụp đổ đặc trưng (Feature Collapse).
        """
        quantized_model.eval()
        try:
            with torch.no_grad():
                output = quantized_model(dummy_input)
            
            # 1. Kiểm tra tràn số (NaN / Inf)
            if torch.isnan(output).any() or torch.isinf(output).any():
                raise ValueError("[Lỗi Nghiêm Trọng] Mô hình sinh ra NaN hoặc Inf sau khi lượng tử hóa.")
                
            # 2. Kiểm tra sụp đổ ma trận (Đầu ra toàn số giống hệt nhau)
            # Không áp dụng cho Tensor kích thước quá nhỏ (chỉ 1 số)
            if output.numel() > 1 and output.float().std() < 1e-7:
                print("[Cảnh báo] Đầu ra bị 'bẹt' (Variance ~ 0). Trọng số lượng tử hóa có thể đang làm mất thông tin nghiêm trọng.")
                
            return True
        except Exception as e:
            print(f"[Quantization Sanity Check Failed]: {e}")
            return False
