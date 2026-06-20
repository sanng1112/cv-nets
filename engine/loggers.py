import os
from typing import Dict, Any

class WandbLogger:
    """
    [Chi tiết hàm]: Trình ghi nhận (Logger) tích hợp Weights & Biases (W&B)
    Theo dõi Loss, Metrics, Learning Rate qua Dashboard Đám mây theo thời gian thực.
    Bạn có thể chia sẻ link Dashboard cho đồng nghiệp.
    """
    def __init__(self, project: str, name: str, config: Dict[str, Any] = None):
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            raise ImportError("Thiếu thư viện wandb. Hãy chạy: uv add wandb")
            
        # Khởi tạo phiên làm việc (run)
        # Chỉ chạy log khi ở Main Process (nếu có dùng Multi-GPU DDP)
        rank = int(os.environ.get('RANK', 0))
        if rank == 0:
            self.run = self.wandb.init(project=project, name=name, config=config)
        else:
            self.run = None
        
    def log(self, metrics: Dict[str, float], step: int = None):
        if self.run is not None:
            self.wandb.log(metrics, step=step)
            
    def finish(self):
        if self.run is not None:
            self.wandb.finish()
