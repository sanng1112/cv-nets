import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(root_path)
print("Root path added to sys.path:", root_path)

import torch
from torch import Tensor, nn
import yaml
from types import SimpleNamespace
from typing import Optional, Any

from layers import *
from layers.activation import *
# Giả sử bạn đã có: Conv2d, LinearLayer, Dropout, build_activation_layer


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{
            k: dict_to_namespace(v)
            for k, v in d.items()
        })
    return d


class CNN(nn.Module):
    def __init__(
        self, 
        opts: Any, 
        input_size: Optional[list] = None, 
        output_dim: Optional[int] = None
    ) -> None:
        super(CNN, self).__init__()

        self.opts = opts
        
        # Lấy thông tin cấu hình cơ bản
        self.input_size = input_size or getattr(self.opts.model, "input_size", [1, 28, 28])
        self.output_dim = output_dim or getattr(self.opts.model, "output_dim", 10)

        # Trích xuất config của các layers từ YAML
        layers_cfg = self.opts.model.layers

        # --- Block 1 ---
        self.conv1 = Conv2d(opts=layers_cfg.conv1)
        self.act1 = build_activation_layer(opts=layers_cfg.act1)

        # --- Block 2 ---
        self.conv2 = Conv2d(opts=layers_cfg.conv2)
        self.act2 = build_activation_layer(opts=layers_cfg.act2)

        # --- Block 3 ---
        self.conv3 = Conv2d(opts=layers_cfg.conv3)
        self.act3 = build_activation_layer(opts=layers_cfg.act3)

        # --- Classifier ---
        self.flatten = nn.Flatten() # Duỗi [B, C, H, W] thành [B, C*H*W]
        self.dropout = Dropout(opts=self.opts.model.train.dropout)
        self.fc1 = LinearLayer(opts=layers_cfg.fc1)

        # Đóng gói thành các khối (Sequential) để dễ quản lý và in ấn
        self.feature_extractor = nn.Sequential(
            self.conv1, self.act1,
            self.conv2, self.act2,
            self.conv3, self.act3
        )

        self.classifier = nn.Sequential(
            self.flatten,
            self.dropout,
            self.fc1
        )

    def forward(self, x: Tensor) -> Tensor:
        # Bước 1: Trích xuất đặc trưng qua các lớp Conv
        x = self.feature_extractor(x)
        
        # Bước 2: Phân loại qua lớp Fully Connected
        x = self.classifier(x)
        
        return x

    def save(self, path: Optional[str] = None) -> None:
        save_path = path or "models/cnn_checkpoint.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'output_dim': self.output_dim,
            'config_opts': self.opts.__dict__ if hasattr(self.opts, '__dict__') else self.opts 
        }

        torch.save(checkpoint, save_path)
        print(f"Model saved successfully at: {save_path}")

    @classmethod
    def load(cls, path: str) -> 'CNN':
        checkpoint = torch.load(path)
        opts = dict_to_namespace(checkpoint['config_opts']) 
        
        model = cls(
            opts=opts,
            input_size=checkpoint.get('input_size'),
            output_dim=checkpoint.get('output_dim')
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def __repr__(self) -> str:
        return (f"CNN(\n"
                f"  (feature_extractor): {self.feature_extractor}\n"
                f"  (classifier): {self.classifier}\n"
                f")")