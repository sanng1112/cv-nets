from typing import Optional
from torch import Tensor, nn

from . import register_act_fn

'''
\text{ReLU}(x) = \begin{cases}
                x, & \text{if } x > 0 \\                
                0, & \text{otherwise}                
                \end{cases}
'''


@register_act_fn('relu')
class ReLU(nn.ReLU):
    def __init__(self, inplace: bool = False, neg_slope: float = 0.1, *arg, **kwargs):
        super().__init__(inplace= inplace)
        self.inplance = inplace 
        
    def __repr__(self):
        return f'ReLU : inplace { self.inplance }'
    
    

