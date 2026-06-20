import torch

def run_sanity_check(trainer):
    """
    Dry-run / Sanity Check mechanism to test OOM, shapes, and gradient flow 
    before starting a multi-hour training session.
    """
    print("\n" + "="*55)
    print("RUNNING PRE-TRAINING SANITY CHECK")
    print("="*55)
    
    # 1. Parameter Configuration Check
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,} (Trainable: {trainable_params:,})")
    print(f"Device: {trainer.device} | AMP: {trainer.use_amp} | Accumulation: {trainer.accumulation_steps}")
    
    # 2. Dataloader & Shape Check
    print("Testing DataLoader extraction...")
    try:
        data = next(iter(trainer.train_loader))
        if isinstance(data, (list, tuple)):
            inputs, targets = data[0], data[1]
        else:
            inputs, targets = data['image'], data['target']
            
        print(f"[OK] DataLoader: Input shape: {inputs.shape}, Target shape: {targets.shape}")
        
        # Check for NaNs
        if torch.isnan(inputs).any() or torch.isnan(targets).any():
            print("[WARNING] NaNs detected in input data!")
            return False
            
        # Check Input Normalization (Values > 10 usually means not normalized)
        if inputs.float().max() > 10.0 or inputs.float().min() < -10.0:
            print(f"[WARNING] Input min={inputs.min():.2f}, max={inputs.max():.2f}. Did you forget to normalize images (e.g. / 255.0)?")
            
    except Exception as e:
        print(f"[ERROR] DataLoader failed to yield a batch. Error: {str(e)}")
        return False
        
    # 2.5 Optimizer Configuration Check
    try:
        for param_group in trainer.optimizer.param_groups:
            lr = param_group['lr']
            opt_name = type(trainer.optimizer).__name__
            if lr > 0.1 and opt_name in ['Adam', 'AdamW']:
                print(f"[WARNING] Learning rate {lr} is extremely high for {opt_name}. Loss might diverge (NaN) immediately!")
    except Exception:
        pass
        
    # 3. OOM & Forward/Backward Check
    print("Simulating Forward/Backward pass (OOM & Gradient Check)...")
    try:
        trainer.model.train()
        trainer.optimizer.zero_grad()
        
        inputs = inputs.to(trainer.device)
        targets = targets.to(trainer.device)
        
        with torch.amp.autocast('cuda', enabled=trainer.use_amp):
            outputs = trainer.model(inputs)
            loss = trainer.criterion(outputs, targets)
            
        if torch.isnan(loss):
            print("[WARNING] Loss is NaN in the first forward pass. Check your data normalization or Loss setup!")
            return False
            
        print(f"[OK] Forward Pass: Output shape: {outputs.shape}, Initial Loss: {loss.item():.4f}")
        
        # Backward Pass Check
        trainer.scaler.scale(loss).backward()
        
        # Gradient Flow Check
        has_grad = False
        for name, p in trainer.model.named_parameters():
            if p.grad is not None and torch.sum(torch.abs(p.grad)).item() > 0:
                has_grad = True
                break
                
        if not has_grad:
            print("[WARNING] No gradients flowing! Your model might be detached from the Loss function.")
            return False
        else:
            print("[OK] Backward Pass & Gradient Flow.")
            
        # Clean up memory
        trainer.optimizer.zero_grad()
        del inputs, targets, outputs, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("[OK] OOM Check Passed! Batch size is safe for your VRAM.")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[ERROR] OOM: Not enough VRAM for batch_size={inputs.shape[0]}.")
            print("Suggestion: Reduce `batch_size` and increase `accumulation_steps` proportionally in config.")
            return False
        else:
            print(f"[ERROR] Forward/Backward failed. Error: {str(e)}")
            return False
            
    print("="*55)
    print("ALL SANITY CHECKS PASSED. READY TO TRAIN.")
    print("="*55 + "\n")
    return True
