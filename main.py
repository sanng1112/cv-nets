import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from types import SimpleNamespace

from utils.seed import seed_everything
from models.builder import CVNetModel
from engine.trainer import BaseTrainer

def dict_to_namespace(d):
    """
    Chuyển đổi Dictionary thành SimpleNamespace để truy cập qua dấu chấm (opts.model.name)
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{
            k: dict_to_namespace(v)
            for k, v in d.items()
        })
    return d

def main():
    parser = argparse.ArgumentParser(description="CV-Nets: Trạm Điều Phối Huấn Luyện Tự Động (CLI Dispatcher)")
    parser.add_argument("--config", type=str, required=True, help="Đường dẫn tới file cấu hình YAML")
    args = parser.parse_args()

    # 1. Đọc YAML -> Opts
    print(f"[*] Đang nạp cấu hình từ: {args.config}")
    with open(args.config, "r", encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    opts = dict_to_namespace(cfg)
    
    # 2. Cố định hạt giống (Reproducibility)
    seed = getattr(opts, "seed", 42)
    seed_everything(seed)
    
    # 3. Tự động Build Model từ YAML
    model = CVNetModel.build_from_yaml(opts)
    
    # 4. Tự động Build Loss (Criterion)
    try:
        from loss_fn import build_loss_fn
        criterion = build_loss_fn(opts)
    except Exception as e:
        print(f"[*] Không thể build loss từ config, dùng mặc định CrossEntropyLoss: {e}")
        criterion = nn.CrossEntropyLoss()
        
    # 5. Tự động Build Optimizer
    from optim.optimizer_builder import build_optimizer
    optimizer = build_optimizer(opts, model)
        
    # 5.5 Tự động Build Scheduler
    from optim.scheduler_builder import build_scheduler
    scheduler = build_scheduler(opts, optimizer)
        
    # 6. Tự động Build Dataloader từ YAML
    from engine.data_factory import build_dataloaders_from_yaml
    train_loader, val_loader = build_dataloaders_from_yaml(opts)
    
    # 7. Đẩy tất cả vào BaseTrainer
    if train_loader is not None:
        trainer = BaseTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler, # Đã tích hợp Scheduler
            train_loader=train_loader,
            val_loader=val_loader,
            opts=opts
        )
        print("[*] Sẵn sàng huấn luyện!")
        trainer.train()
    else:
        print("[*] Mô hình đã được khởi tạo thành công từ YAML! (Chưa có Dataloader để train)")

if __name__ == "__main__":
    main()
