import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(root_path)
print("Root path added to sys.path:", root_path)
from layers import *
from layers.activation import *
import yaml
from types import SimpleNamespace


import torch
from torch import Tensor, nn


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{
            k: dict_to_namespace(v)
            for k, v in d.items()
        })
    return d

class MLP(nn.Module):
    def __init__(self, opts, 
                    input_size: int = None, 
                    hidden_size: int = None, 
                    output_size: int = None) -> None:
        super(MLP, self).__init__()

        self.input_size = input_size 
        self.hidden_size = hidden_size 
        self.output_size = output_size     
        self.opts = opts

        self.fc1 = LinearLayer(input_size, hidden_size, opts=self.opts.model.fc1)
        self.act1 = build_activation_layer(opts=self.opts.model.act1)
        self.fc2 = LinearLayer(hidden_size, output_size, opts=self.opts.model.fc2)
        self.dropout = Dropout(opts = self.opts.model.train.dropout)

        self._forward = nn.Sequential(
            self.fc1,
            self.dropout,
            self.act1,
            self.fc2
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._forward(x)
            

    def save(self, path: Optional[str] = None) -> None:
        save_path = path or "models/mlp_checkpoint.pth"
        

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_size,
            'hidden_dim': self.hidden_size,
            'output_dim': self.output_size,
            'config_opts': self.opts.__dict__ if hasattr(self.opts, '__dict__') else self.opts 
        }

        torch.save(checkpoint, save_path)
        print(f"Model saved successfully at: {save_path}")

    @classmethod
    def load(cls, path) -> 'MLP':
        checkpoint = torch.load(path)
        opts = dict_to_namespace(checkpoint['config_opts']) 
        
        model = cls(
            opts=opts,
            input_size=checkpoint['input_dim'],
            hidden_size=checkpoint['hidden_dim'],
            output_size=checkpoint['output_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


    def __repr__(self) -> str:
        return (f"MLP(\n"
                f"  (fc1): {self.fc1}\n"
                f"  (dropout): {self.dropout}\n"
                f"  (act1): {self.act1}\n"
                f"  (fc2): {self.fc2}\n"
                f")")

