import torch
from torch import nn, Tensor
from typing import Optional, Any
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt, sns = None, None
    print("Warning: matplotlib and seaborn are required for visualization. Please install them.")


def visualize_weight_distribution(module: nn.Module, layer_name: Optional[str] = None) -> None:
    """
    Vẽ phân phối trọng số (histogram) của layer/block này.
    """
    if plt is None or sns is None:
        print("Cần cài đặt matplotlib và seaborn: pip install matplotlib seaborn")
        return

    layer_name = layer_name or module.__class__.__name__
    weights = []
    for name, param in module.named_parameters():
        if "weight" in name and param.requires_grad:
            weights.append(param.detach().cpu().numpy().flatten())
    
    if not weights:
        print(f"Không tìm thấy tham số weight nào trong {layer_name}")
        return
        
    weights = np.concatenate(weights)
    plt.figure(figsize=(8, 5))
    sns.histplot(weights, bins=50, kde=True)
    plt.title(f"Phân phối tham số - {layer_name}")
    plt.xlabel("Giá trị trọng số")
    plt.ylabel("Tần suất")
    plt.grid(True)
    plt.show()


def visualize_weight_heatmap(module: nn.Module, layer_name: Optional[str] = None) -> None:
    """
    Vẽ heatmap thể hiện cường độ của trọng số cho các convolutional / linear layers.
    """
    if plt is None or sns is None:
        print("Cần cài đặt matplotlib và seaborn: pip install matplotlib seaborn")
        return

    layer_name = layer_name or module.__class__.__name__
    found = False
    for name, param in module.named_parameters():
        if "weight" in name and param.requires_grad:
            found = True
            weight_tensor = param.detach().cpu().numpy()
            
            # Nếu là Conv2d (C_out, C_in, H, W) -> tính trung bình không gian
            if len(weight_tensor.shape) == 4:
                weight_tensor = weight_tensor.mean(axis=(2, 3))
            
            # Nếu là mảng 1D (ví dụ bias hoặc BatchNorm) -> chuyển về 2D (1, N)
            if len(weight_tensor.shape) == 1:
                weight_tensor = np.expand_dims(weight_tensor, 0)
                
            plt.figure(figsize=(10, 8))
            sns.heatmap(weight_tensor, cmap="coolwarm", center=0)
            plt.title(f"Heatmap tham số - {layer_name} ({name})")
            plt.xlabel("Input Channels")
            plt.ylabel("Output Channels / Feature")
            plt.show()
            
    if not found:
        print(f"Không tìm thấy tham số weight nào để vẽ heatmap trong {layer_name}")


class FeatureMapHook:
    """
    Đăng ký hook để lấy feature map (đầu ra) của bất kỳ module nào sau forward pass.
    """
    def __init__(self, module: nn.Module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.feature_map: Optional[Tensor] = None
        self.module_name = module.__class__.__name__
        
    def hook_fn(self, module, input, output):
        if isinstance(output, Tensor):
            self.feature_map = output.detach().cpu()
        elif isinstance(output, tuple) or isinstance(output, list):
            self.feature_map = output[0].detach().cpu()
            
    def remove(self):
        """Hủy bỏ hook để tránh tốn memory."""
        self.hook.remove()
        
    def visualize(self, max_channels: int = 16) -> None:
        """
        Trực quan hóa Feature Maps (kênh màu) dưới dạng hình ảnh.
        """
        if plt is None:
            print("Cần cài đặt matplotlib: pip install matplotlib")
            return
            
        if self.feature_map is None:
            print("Chưa có feature map. Hãy chạy forward pass qua module trước khi visualize.")
            return
            
        # Feature map thường có dạng (Batch, C, H, W)
        if len(self.feature_map.shape) != 4:
            print(f"Feature map của {self.module_name} không có dạng 4D (B, C, H, W). Shape: {self.feature_map.shape}. Bỏ qua.")
            return
            
        fmap = self.feature_map[0] # Lấy ảnh đầu tiên trong batch
        channels = min(fmap.shape[0], max_channels)
        cols = 4
        rows = (channels + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows * cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        for i in range(channels):
            ax = axes[i]
            img = fmap[i].numpy()
            ax.imshow(img, cmap="viridis")
            ax.axis('off')
            ax.set_title(f"Channel {i}")
            
        for j in range(channels, len(axes)):
            axes[j].axis('off')
            
        fig.suptitle(f"Feature Map - {self.module_name} (Max {max_channels} channels)")
        plt.tight_layout()
        plt.show()


def enable_research_visualization(module: nn.Module) -> nn.Module:
    """
    Tiêm (monkey-patch) các hàm trực quan hóa vào bất kỳ module nào (Layer hoặc Block).
    """
    # Gắn hàm như một phương thức của đối tượng
    import types
    
    module.visualize_weight_distribution = types.MethodType(visualize_weight_distribution, module)
    module.visualize_weight_heatmap = types.MethodType(visualize_weight_heatmap, module)
    
    # Gắn tiện ích lấy feature map
    def register_feature_map_hook(self):
        return FeatureMapHook(self)
        
    module.register_feature_map_hook = types.MethodType(register_feature_map_hook, module)
    
    return module
