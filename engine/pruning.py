import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class ModelPruner:
    """
    [Chi tiết hàm]: Bộ Tỉa Mạng Nơ-ron (Network Pruning).
    Đóng vai trò như một bác sĩ phẫu thuật: Tìm và cắt bỏ những nơ-ron "lười biếng"
    (trọng số gần bằng 0) không đóng góp vào kết quả.
    - Kết hợp Pruning + Quantization (INT8) sẽ tạo ra siêu mô hình cực nhẹ.
    """
    
    @staticmethod
    def prune_unstructured(model: nn.Module, amount: float = 0.3) -> nn.Module:
        """
        Cắt tỉa không cấu trúc (Unstructured L1 Pruning).
        - Cắt bỏ `amount` (VD: 0.3 = 30%) số lượng trọng số (weights) nhỏ nhất trong mạng.
        - Khuyết điểm: Mạng bị "rỗ", dung lượng đĩa giảm mạnh khi nén zip nhưng RAM lúc chạy giảm ít.
        """
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Lập mặt nạ (mask) che đi các trọng số yếu nhất
                prune.l1_unstructured(module, name='weight', amount=amount)
                # Xóa vĩnh viễn chúng khỏi RAM, biến thành số 0 cứng
                prune.remove(module, 'weight')
                
        print(f"[Pruning] Đã dọn dẹp vĩnh viễn {amount*100}% trọng số rác (Unstructured).")
        return model

    @staticmethod
    def prune_structured(model: nn.Module, amount: float = 0.2) -> nn.Module:
        """
        Cắt tỉa có cấu trúc (Structured L2 Pruning).
        - Trực tiếp cắt bỏ toàn bộ một kênh/bộ lọc (Filter) trong mạng CNN.
        - Ưu điểm: Làm giảm số lượng tính toán FLOPs thực tế, mô hình chạy nhanh hơn rõ rệt.
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Cắt bỏ các Filter (dim=0) có chuẩn L2 nhỏ nhất
                prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
                prune.remove(module, 'weight')
                
        print(f"[Pruning] Đã gọt bỏ vĩnh viễn {amount*100}% kênh Filter kém cỏi (Structured).")
        return model
