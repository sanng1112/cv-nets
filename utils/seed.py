import random
import os
import numpy as np
import torch

def seed_everything(seed: int = 42):
    """
    Cố định toàn bộ các hạt giống ngẫu nhiên (Random Seeds) 
    để đảm bảo khả năng tái lập 100% (Reproducibility) trong các thí nghiệm khoa học.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # Dành cho multi-GPU
        
    # Đảm bảo các thuật toán Convolution của CUDNN chạy cố định
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[Seed] Đã thiết lập Global Seed = {seed} cho toàn bộ Framework.")
