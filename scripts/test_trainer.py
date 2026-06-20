import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from engine.trainer import BaseTrainer
from engine.sanity_check import run_sanity_check
from optim.optimizer_builder import build_optimizer
from optim.scheduler_builder import build_scheduler

def test_trainer():
    # 1. Setup mock data
    x_train = torch.randn(128, 10)
    y_train = torch.randn(128, 1) # Regression task
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    val_loader = DataLoader(TensorDataset(x_train[:32], y_train[:32]), batch_size=32)
    
    # 2. Setup mock model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.GELU(),
        nn.Linear(64, 1)
    )
    
    # 3. Setup configurations
    parser = argparse.ArgumentParser()
    opts = parser.parse_args([])
    opts.device = "cpu" # Force CPU for local quick testing
    opts.use_amp = False # Disable AMP for CPU testing
    opts.accumulation_steps = 2
    opts.epochs = 3
    opts.clip_grad_norm = 1.0
    opts.save_dir = "./temp_checkpoints"
    opts.task_type = "regression"
    opts.num_classes = 1
    
    opts.optim_name = "adamw"
    opts.lr = 1e-3
    opts.weight_decay = 1e-4
    opts.momentum = 0.9
    
    opts.scheduler_name = "cosine"
    opts.warmup_epochs = 1
    opts.min_lr = 1e-5
    
    # 4. Build components
    criterion = nn.MSELoss()
    optimizer = build_optimizer(opts, model)
    scheduler = build_scheduler(opts, optimizer)
    
    # 5. Initialize Trainer
    trainer = BaseTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        opts=opts
    )
    
    print("Running Sanity Check...")
    success = run_sanity_check(trainer)
    assert success, "Sanity Check Failed!"
    
    print("Testing Trainer Loop (Forward, Backward, Accumulation)...")
    trainer.train()
    print("Trainer Loop OK!")

if __name__ == "__main__":
    test_trainer()
