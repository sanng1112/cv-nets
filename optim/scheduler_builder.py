import argparse
import torch.optim.lr_scheduler as lr_scheduler

def add_scheduler_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Chi tiết hàm: `add_scheduler_args`
    - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
    - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
    """
    group = parser.add_argument_group("Scheduler arguments")
    group.add_argument("--scheduler-name", type=str, default="cosine", choices=["cosine", "step", "exponential"])
    group.add_argument("--epochs", type=int, default=100, help="Total number of epochs")
    group.add_argument("--warmup-epochs", type=int, default=5, help="Number of warmup epochs")
    group.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate for Cosine")
    group.add_argument("--step-size", type=int, default=30, help="Step size for StepLR")
    group.add_argument("--gamma", type=float, default=0.1, help="Gamma for StepLR/ExponentialLR")
    return parser

def build_scheduler(opts, optimizer):
    """
    Chi tiết hàm: `build_scheduler`
    - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
    - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
    """
    scheduler_name = getattr(opts, "scheduler_name", "cosine").lower()
    epochs = getattr(opts, "epochs", 100)
    warmup_epochs = getattr(opts, "warmup_epochs", 5)
    
    if scheduler_name == "cosine":
        min_lr = getattr(opts, "min_lr", 1e-6)
        main_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=min_lr)
    elif scheduler_name == "step":
        step_size = getattr(opts, "step_size", 30)
        gamma = getattr(opts, "gamma", 0.1)
        main_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == "exponential":
        gamma = getattr(opts, "gamma", 0.95)
        main_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
        
    if warmup_epochs > 0:
        warmup_scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
        return lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
    return main_scheduler
