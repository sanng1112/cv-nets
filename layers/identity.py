from torch import Tensor

from layers.base_layer import BaseLayer

class Identity(BaseLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, x:Tensor) -> Tensor:
        return x