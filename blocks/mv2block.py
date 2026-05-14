import argparse
import torch
from torch import nn, Tensor
from typing import Optional, Union, Tuple, Any
from layers import (Conv2d, build_activation_layer)


class MV2Block(nn.Conv2d):
    def __init__(
        self,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        kernel_size: Optional[Union[int, Tuple[int, int]]] = None,
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Optional[Union[int, Tuple[int, int]]] = None,
        dilation: Optional[Union[int, Tuple[int, int]]] = None,
        groups: Optional[int] = None,
        bias: Optional[bool] = None,
        padding_mode: Optional[str] = None,
        use_norm: Optional[bool] = None,
        use_act: Optional[bool] = None,
        act_name: Optional[str] = None,
        opts: Optional[Any] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        pass