import argparse
import torch
import torch.nn as nn
from optim import build_optimizer, build_scheduler

def test_optim():
    parser = argparse.ArgumentParser()
    opts = parser.parse_args([])
    opts.optim_name = "adamw"
    opts.lr = 0.1
    opts.weight_decay = 1e-4
    opts.momentum = 0.9
    
    opts.scheduler_name = "cosine"
    opts.epochs = 10
    opts.warmup_epochs = 3
    opts.min_lr = 1e-5
    
    # Dummy model
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.LayerNorm(10)
    )
    
    print("Testing Optimizer Builder...")
    optimizer = build_optimizer(opts, model)
    # Check weight decay grouping
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]['weight_decay'] == 0.0 # Bias and Norm
    assert optimizer.param_groups[1]['weight_decay'] == 1e-4 # Weights
    print("Optimizer OK.")

    print("Testing Scheduler Builder...")
    scheduler = build_scheduler(opts, optimizer)
    
    # Test Warmup
    lrs = []
    for epoch in range(opts.epochs):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
    
    # Check warmup phase (lr should increase)
    assert lrs[0] < lrs[1] < lrs[2]
    # Check cosine phase (lr should decrease)
    assert lrs[3] > lrs[4] > lrs[-1]
    print(f"Learning rates simulated: {[round(lr, 4) for lr in lrs]}")
    print("Scheduler OK.")

if __name__ == "__main__":
    test_optim()
