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
from blocks import *

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{
            k: dict_to_namespace(v)
            for k, v in d.items()
        })
    elif isinstance(d, list):  
        return [dict_to_namespace(v) for v in d]
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
        self.input_size = input_size or getattr(self.opts.model, "input_size", [1, 48, 48])
        self.output_dim = output_dim or getattr(self.opts.model, "output_dim", 7)

        feature_layers = []
        classifier_layers = []

        layers_cfg = self.opts.model.layers

        for cfg in layers_cfg:
            layer_type = cfg.type.lower()

            if layer_type == "convbnact":
                block = ConvBNAct(opts=cfg)
                feature_layers.append(block)

            elif layer_type == "fc":
                if not any(isinstance(l, nn.Flatten) for l in classifier_layers):
                    classifier_layers.append(nn.Flatten())
            
                classifier_layers.append(LinearLayer(opts=cfg))
            elif layer_type == "act":
                act_fn = build_activation_layer(opts = cfg)
                classifier_layers.append(act_fn)
        self.feature_extractor = nn.Sequential(*feature_layers)
        
        train_cfg = getattr(self.opts, "train", getattr(self.opts.model, "train", None))
        if train_cfg and hasattr(train_cfg, "dropout"):
            dropout_cfg = train_cfg.dropout
            if len(classifier_layers) > 1:
                classifier_layers.insert(-1, Dropout(opts=dropout_cfg))

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

    def save(self, save_dir: str = "models", model_name: str = "cnn_model") -> None:
        os.makedirs(save_dir, exist_ok=True)
        
        yaml_path = os.path.join(save_dir, f"{model_name}.yaml")
        pth_path = os.path.join(save_dir, f"{model_name}.pth")
        
        torch.save(self.state_dict(), pth_path)
        
        config_dict = namespace_to_dict(self.opts)
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


if __name__ == "__main__":
    CONFIG = 'config.yaml'

    with open(CONFIG, "r") as f:
        cfg = yaml.safe_load(f)

    opts = dict_to_namespace(cfg)
    
    model = CNN(opts=opts)
    print("\n=== CẤU TRÚC MÔ HÌNH SAU KHI THIẾT KẾ LẠI HỢP NHẤT ===")
    print(model)