import torch
from torch import Tensor


def compute_class_weights(
    target: Tensor, n_classes: int, norm_val: float = 1.1
) -> Tensor:
    class_hist = torch.histc(target.float(), bins=n_classes, min=0, max=n_classes - 1)
    print(class_hist)
    mask_indices = class_hist == 0
    norm_hist = torch.div(class_hist, class_hist.sum())
    print(norm_hist)
    norm_hist = torch.add(norm_hist, norm_val)
    class_wts = torch.div(torch.ones_like(class_hist), torch.log(norm_hist))
    class_wts[mask_indices] = 0.0
    return class_wts.to(device=target.device)
