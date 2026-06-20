import argparse
import torch.optim as optim

def add_optimizer_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Chi tiết hàm: `add_optimizer_args`
    - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
    - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
    """
    group = parser.add_argument_group("Optimizer arguments")
    group.add_argument("--optim-name", type=str, default="adamw", choices=["sgd", "adam", "adamw", "rmsprop"])
    group.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    group.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    group.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD/RMSprop")
    return parser

def build_optimizer(opts, model):
    """
    Chi tiết hàm: `build_optimizer`
    - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
    - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
    """
    optim_name = getattr(opts, "optim_name", "adamw").lower()
    lr = getattr(opts, "lr", 1e-3)
    weight_decay = getattr(opts, "weight_decay", 1e-4)
    momentum = getattr(opts, "momentum", 0.9)
    
    # Advanced: Separate parameters with and without weight decay
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
            
    param_groups = [
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': weight_decay}
    ]
    
    if optim_name == "sgd":
        return optim.SGD(param_groups, lr=lr, momentum=momentum)
    elif optim_name == "adam":
        return optim.Adam(param_groups, lr=lr)
    elif optim_name == "adamw":
        return optim.AdamW(param_groups, lr=lr)
    elif optim_name == "rmsprop":
        return optim.RMSprop(param_groups, lr=lr, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer: {optim_name}")
