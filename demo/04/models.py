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
from layers.normalization import *


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{
            k: dict_to_namespace(v)
            for k, v in d.items()
        })
    return d

def namespace_to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    elif isinstance(ns, dict):
        return {k: namespace_to_dict(v) for k, v in ns.items()}
    elif isinstance(ns, list):
        return [namespace_to_dict(v) for v in ns]
    else:
        return ns

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

        layers_cfg = self.opts.model.layers

        self.conv1 = Conv2d(opts=layers_cfg.conv1)
        self.bnorm1 = build_normalization_layer(opts=layers_cfg.bnorm1)
        self.act1 = build_activation_layer(opts=layers_cfg.act1)
        self.pool1 = build_pooling_layer(opts = layers_cfg.pool1)

        self.conv2 = Conv2d(opts=layers_cfg.conv2)
        self.bnorm2 = build_normalization_layer(opts=layers_cfg.bnorm2)
        self.act2 = build_activation_layer(opts=layers_cfg.act2)
        self.pool2 = build_pooling_layer(opts = layers_cfg.pool2)
        
        self.conv3 = Conv2d(opts=layers_cfg.conv3)
        self.bnorm3 = build_normalization_layer(opts=layers_cfg.bnorm3)
        self.act3 = build_activation_layer(opts=layers_cfg.act3)
        self.pool3 = build_pooling_layer(opts = layers_cfg.pool3)

        self.conv4 = Conv2d(opts=layers_cfg.conv4)
        self.bnorm4 = build_normalization_layer(opts=layers_cfg.bnorm4)
        self.act4 = build_activation_layer(opts=layers_cfg.act4)
        self.pool4 = build_pooling_layer(opts = layers_cfg.pool4)


        self.conv5 = Conv2d(opts=layers_cfg.conv5)
        self.bnorm5 = build_normalization_layer(opts=layers_cfg.bnorm5)
        self.act5 = build_activation_layer(opts=layers_cfg.act5)

        self.flatten = Flatten() 
        self.dropout = Dropout(opts=self.opts.model.train.dropout)
        self.fc1 = LinearLayer(opts=layers_cfg.fc1)
        self.fc2 = LinearLayer(opts=layers_cfg.fc2)
        self.fc3 = LinearLayer(opts=layers_cfg.fc3)
        
        self.feature_extractor = nn.Sequential(
            self.conv1, self.bnorm1, self.act1, self.pool1,
            self.conv2, self.bnorm2, self.act2, self.pool2,
            self.conv3, self.bnorm3, self.act3, self.pool3,
            self.conv4, self.bnorm4, self.act4, self.pool4,
            self.conv5, self.bnorm5, self.act5
        )

        self.classifier = nn.Sequential(
            self.flatten,
            self.fc1,
            self.fc2,
            self.dropout,
            self.fc3
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

    def save(self, save_dir: str = "models", model_name: str = "cnn_model") -> None:
        os.makedirs(save_dir, exist_ok=True)
        
        yaml_path = os.path.join(save_dir, f"{model_name}.yaml")
        pth_path = os.path.join(save_dir, f"{model_name}.pth")
        
        torch.save(self.state_dict(), pth_path)
        print(self.opts)
        
        config_dict = namespace_to_dict(self.opts)
        print(config_dict)
        
        config_dict["model"]["name"] = model_name
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            
        print(f"Model saved successfully!\n- Weights: {pth_path}\n- Config: {yaml_path}")

    @classmethod
    def load(cls, yaml_path: str, pth_path: str) -> 'CNN':
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            
        opts = dict_to_namespace(config_dict) 
        
        model = cls(
            opts=opts,
            input_size=getattr(opts.model, "input_size", None),
            output_dim=getattr(opts.model, "output_dim", None)
        )
        state_dict = torch.load(pth_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        
        print(f"Model loaded successfully from config: {yaml_path}")
        return model

    def __repr__(self) -> str:
        return (f"CNN(\n"
                f"  (feature_extractor): {self.feature_extractor}\n"
                f"  (classifier): {self.classifier}\n"
                f")")