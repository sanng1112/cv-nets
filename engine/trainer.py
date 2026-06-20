import os
import json
import yaml
import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from engine.metrics_modules.builder import build_metrics

class BaseTrainer:
    """
    Robust BaseTrainer for Deep Learning.
    Supports:
    - Standard Training Loop
    - Automatic Mixed Precision (AMP) to accelerate GPU training
    - Gradient Accumulation to support large batch sizes on limited VRAM
    """
    def __init__(
        self, 
        model, 
        criterion, 
        optimizer, 
        scheduler, 
        train_loader, 
        val_loader, 
        opts
    ):
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.device = getattr(opts, "device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # --- [TORCH COMPILE SUPPORT] ---
        self.use_compile = getattr(opts, "use_compile", False)
        if self.use_compile and hasattr(torch, "compile") and self.device != "cpu":
            import logging
            import warnings
            
            # Tắt toàn bộ cảnh báo (Warning) gây rối mắt của trình biên dịch C++ (Triton)
            warnings.filterwarnings("ignore", category=UserWarning)
            logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
            logging.getLogger("torch._inductor").setLevel(logging.ERROR)
            
            print("[*] Áp dụng torch.compile (JIT) để tăng tốc độ thực thi...")
            self.model = torch.compile(self.model)
        
        # --- [DDP SUPPORT] ---
        import torch.distributed as dist
        self.is_ddp = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_ddp else 0
        
        if self.is_ddp and self.device != "cpu":
            # Wrap mô hình bằng DDP để chạy Multi-GPU
            self.model = nn.parallel.DistributedDataParallel(
                self.model, 
                device_ids=[torch.distributed.get_rank()]
            )
        
        self.epochs = getattr(opts, "epochs", 100)
        self.use_amp = getattr(opts, "use_amp", torch.cuda.is_available())
        self.accumulation_steps = getattr(opts, "accumulation_steps", 1)
        self.clip_grad_norm = getattr(opts, "clip_grad_norm", 1.0)
        self.eval_interval = getattr(opts, "eval_interval", 1)
        self.save_dir = getattr(opts, "save_dir", "./checkpoints")
        
        # Thermal Throttling / Anti-Overheat
        self.batch_sleep_sec = getattr(opts, "batch_sleep_sec", 0.0)
        self.epoch_sleep_sec = getattr(opts, "epoch_sleep_sec", 0.0)
        
        self.opts_dict = vars(opts) if hasattr(opts, '__dict__') else opts
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Dump configuration to yaml
        with open(os.path.join(self.save_dir, "config.yaml"), "w") as f:
            yaml.dump(self.opts_dict, f)
            
        # Initialize Metrics Engine
        self.task_type = getattr(opts, "task_type", "classification")
        num_classes = getattr(opts, "num_classes", 10)
        self.train_metrics = build_metrics(self.task_type, num_classes)
        self.val_metrics = build_metrics(self.task_type, num_classes)
        if self.train_metrics:
            self.train_metrics = self.train_metrics.to(self.device)
            self.val_metrics = self.val_metrics.to(self.device)
            
        self.history = []
        
        # AMP Scaler for mixed precision training
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        
        self.start_epoch = 0
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch):
        """
        Chi tiết hàm: `train_epoch`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        self.model.train()
        total_loss = 0.0
        
        # We use tqdm for a nice progress bar only on Master Node (rank 0)
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]") if self.rank == 0 else self.train_loader
        
        if self.train_metrics:
            self.train_metrics.reset()
            
        self.optimizer.zero_grad()
        
        for batch_idx, data in enumerate(pbar):
            # Generalize unpacking depending on dataset format (assumes image, target)
            if isinstance(data, (list, tuple)):
                inputs, targets = data[0], data[1]
            else:
                inputs, targets = data['image'], data['target']
                
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass with AMP
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(inputs)
                # Compute loss and scale it for gradient accumulation
                loss = self.criterion(outputs, targets) / self.accumulation_steps
            
            # Backward pass with Scaler
            self.scaler.scale(loss).backward()
            
            # Compute Metrics
            if self.train_metrics and self.task_type != "detection":
                self.train_metrics.update(outputs, targets)
            
            # Step optimizer every accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                if self.clip_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            # Re-multiply loss for accurate logging
            loss_item = loss.item() * self.accumulation_steps
            total_loss += loss_item
            
            pbar_dict = {"Loss": f"{loss_item:.4f}", "LR": f"{self.optimizer.param_groups[0]['lr']:.6f}"}
            if self.rank == 0:
                pbar.set_postfix(pbar_dict)
            import time
            if self.batch_sleep_sec > 0:
                time.sleep(self.batch_sleep_sec)
                
        epoch_loss = total_loss / len(self.train_loader)
        metrics_res = self.train_metrics.compute() if self.train_metrics else {}
        return epoch_loss, {k: v.item() for k, v in metrics_res.items()}

    @torch.no_grad()
    def validate_epoch(self, epoch):
        """
        Chi tiết hàm: `validate_epoch`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        if self.val_loader is None:
            return 0.0
            
        self.model.eval()
        total_loss = 0.0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.epochs} [Val]") if self.rank == 0 else self.val_loader
        
        if self.val_metrics:
            self.val_metrics.reset()
            
        for data in pbar:
            if isinstance(data, (list, tuple)):
                inputs, targets = data[0], data[1]
            else:
                inputs, targets = data['image'], data['target']
                
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
            if self.val_metrics and self.task_type != "detection":
                self.val_metrics.update(outputs, targets)
                
            total_loss += loss.item()
            if self.rank == 0:
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(self.val_loader)
        metrics_res = self.val_metrics.compute() if self.val_metrics else {}
        metrics_res_cpu = {k: v.item() for k, v in metrics_res.items()}
        if self.rank == 0:
            print(f"--> Val Loss: {avg_loss:.4f} | Metrics: {metrics_res_cpu}")
        return avg_loss, metrics_res_cpu

    def save_checkpoint(self, epoch, is_best=False):
        """
        Chi tiết hàm: `save_checkpoint`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        model_state = self.model.module.state_dict() if self.is_ddp else self.model.state_dict()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        path = os.path.join(self.save_dir, f"checkpoint_last.pt")
        torch.save(checkpoint, path)
        if is_best:
            best_path = os.path.join(self.save_dir, f"checkpoint_best.pt")
            torch.save(checkpoint, best_path)

    def train(self):
        """
        Chi tiết hàm: `train`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        print(f"Starting Training on device: {self.device} | AMP: {self.use_amp} | Accumulation: {self.accumulation_steps}")
        for epoch in range(self.start_epoch, self.epochs):
            train_loss, train_metrics = self.train_epoch(epoch)
            
            val_loss, val_metrics = 0.0, {}
            # Chỉ chạy validate theo chu kỳ eval_interval hoặc ở epoch cuối cùng
            if self.val_loader is not None and (epoch % self.eval_interval == 0 or epoch == self.epochs - 1):
                val_loss, val_metrics = self.validate_epoch(epoch)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log History
            epoch_history = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr
            }
            # Add prefix for metrics
            epoch_history.update({f"train_{k}": v for k, v in train_metrics.items()})
            epoch_history.update({f"val_{k}": v for k, v in val_metrics.items()})
            self.history.append(epoch_history)
            
            if self.scheduler is not None:
                self.scheduler.step()
                
            # Master Node (rank 0) đảm nhiệm việc Save Checkpoint và Log file
            if self.rank == 0:
                # Save history to CSV and JSON
                pd.DataFrame(self.history).to_csv(os.path.join(self.save_dir, "history.csv"), index=False)
                with open(os.path.join(self.save_dir, "history.json"), "w") as f:
                    json.dump(self.history, f, indent=4)
                    
                # Checkpoint saving
                is_best = False
                if self.val_loader is not None and val_metrics and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    is_best = True
                    
                self.save_checkpoint(epoch, is_best=is_best)
            
            import time
            if self.epoch_sleep_sec > 0 and epoch < self.epochs - 1:
                print(f"[Anti-Overheat] Đang nghỉ {self.epoch_sleep_sec} giây để hạ nhiệt GPU...")
                time.sleep(self.epoch_sleep_sec)
