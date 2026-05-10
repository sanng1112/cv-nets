from typing import Optional
from torch import Tensor, nn
from . import register_act_fn


'''
\text{LeakyRelu}(x) = \begin{cases}
                x, & \text{if } x > 0 \\                
                \alpha x, & \text{otherwise}                
                \end{cases} 
                \qquad \alpha \text{ is} \text{negative slope}
'''



@register_act_fn(name="leaky_relu")
class LeakyReLU(nn.LeakyReLU):
    def __init__(
        self, inplace: Optional[bool] = False, negative_slope: Optional[float] = 1e-2, *args, **kwargs
    ) -> None:
        super().__init__(inplace=inplace, negative_slope=negative_slope)
    
    def __repr__(self) -> str:
        return f"LeakyReLU(inplace={self.inplace}, negative_slope={self.negative_slope})"
    
