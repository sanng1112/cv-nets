import argparse
import torch
from torch import nn, Tensor
from typing import Optional, Union, Tuple, Any
from layers import (Conv2d, build_activation_layer, build_normalization_layer)

class MV2Block(nn.Module):
    def __init__(
        self,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        kernel_size: Optional[Union[int, Tuple[int, int]]] = 3,
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        expansion_ratio: Optional[int] = None,
        dilation: Optional[Union[int, Tuple[int, int]]] = 1,
        act_name: Optional[str] = None,
        opts: Optional[Any] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        opts = opts or kwargs.get("opts", None)
        
        def get_param(explicit_val, attr_name, default_val):
            if explicit_val is not None:
                return explicit_val
            return getattr(opts, attr_name, default_val)
        _in_channels = get_param(in_channels, "in_channels", None)
        _out_channels = get_param(out_channels, "out_channels", None)
        _stride = get_param(stride, "stride", 1)
        _expansion_ratio = get_param(expansion_ratio, "expansion_ratio", 6)
        _act_name = get_param(act_name, "act.type", "relu6")
        
        if _in_channels is None or _out_channels is None:
            raise ValueError("`in_channels` and `out_channels` must be provided directly or via `opts`.")

        hidden_dim = int(round(_in_channels * _expansion_ratio))
        self.use_res_connect = _stride == 1 and _in_channels == _out_channels

        layers = []
        
        if _expansion_ratio != 1:
            layers.append(
                Conv2d(in_channels=_in_channels, out_channels=hidden_dim, kernel_size=1, bias=False, opts=opts)
            )
            layers.append(build_normalization_layer(opts, num_features=hidden_dim))
            layers.append(build_activation_layer(opts, act_name=_act_name))

        layers.append(
            Conv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                stride=_stride,
                padding=int((kernel_size - 1) // 2 * dilation),
                dilation=dilation,
                groups=hidden_dim, 
                bias=False,
                opts=opts
            )
        )

        layers.append(build_normalization_layer(opts, num_features=hidden_dim))
        layers.append(build_activation_layer(opts, act_name=_act_name))

        layers.append(
            Conv2d(in_channels=hidden_dim, out_channels=_out_channels, kernel_size=1, bias=False, opts=opts)
        )
        layers.append(build_normalization_layer(opts, num_features=_out_channels))
        
        self.block = nn.Sequential(*layers)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group("MV2Block Arguments")
        group.add_argument("--expansion-ratio", type=int, default=6, help="Hệ số mở rộng kênh")
        group.add_argument("--act-name", type=str, default="relu6", help="Tên hàm kích hoạt sử dụng")
        return parser

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        return self.block(x)