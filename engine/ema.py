import math
import copy
import torch
import torch.nn as nn

class ModelEMA:
    """
    [Chi tiết hàm]: Mô hình Exponential Moving Average (EMA)
    Duy trì một bản sao lưu các trọng số mô hình (Shadow Model) được cập nhật liên tục 
    bằng trung bình cộng có trọng số hàm mũ. Giúp tăng độ ổn định và accuracy khi validation.
    Thường được dùng trong YOLO, Swin, ViT để cải thiện kết quả.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999, tau: int = 2000, updates: int = 0):
        # Tạo bản sao hoàn chỉnh nhưng không lưu gradient
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
            
        self.decay = decay
        self.tau = tau
        self.updates = updates

    def update(self, model: nn.Module):
        self.updates += 1
        # Giảm dần decay ở giai đoạn đầu để khởi động nhanh, sau đó tăng dần đến decay thực
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        
        # Lấy trạng thái của model đang train (hỗ trợ DDP/DataParallel)
        msd = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        
        with torch.no_grad():
            for name, param in self.ema.state_dict().items():
                if param.dtype.is_floating_point:
                    # Cập nhật mượt mà (Smoothing)
                    param.copy_(param * d + (1.0 - d) * msd[name].detach())
                    
    def update_attr(self, model: nn.Module, include=(), exclude=('process_group', 'reducer')):
        # Đồng bộ các cấu hình không phải tensor (ví dụ configs) từ model đang train sang EMA
        import inspect
        for k, v in model.__dict__.items():
            if (len(include) and k not in include) or k in exclude:
                continue
            if not k.startswith('_') and not inspect.ismethod(v):
                setattr(self.ema, k, v)
