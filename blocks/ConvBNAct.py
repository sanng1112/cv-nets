import argparse
import torch
from torch import nn, Tensor
from typing import Optional, Union, Tuple, Any
from layers import Conv2d, build_activation_layer, build_normalization_layer, arguments_activation_fn,  arguments_norm_layers
from utils.config_helper import get_param

class ConvBNAct(nn.Module):
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
        opts: Optional[Any] = None,
        *args, 
        **kwargs):
        super().__init__() 
        
        self.block = nn.Sequential()
        
        opts_conv = get_param(opts, None, 'conv', None)
        _in_channels = get_param(opts_conv , in_channels, 'in_channels', None)
        _out_channels = get_param(opts_conv, out_channels, "out_channels", None)
        _kernel_size = get_param(opts_conv, kernel_size, "kernel_size", 3)
        _stride = get_param(opts_conv, stride, "stride", 1)
        _padding = get_param(opts_conv, padding, "padding", 1)
        _dilation = get_param(opts_conv, dilation, "dilation", 1)
        _groups = get_param(opts_conv, groups, "groups", 1)
        _padding_mode = get_param(opts_conv, padding_mode, "padding_mode", "zeros")
        _bias = get_param(opts_conv, bias, "bias", False)

        if _in_channels is None or _out_channels is None:
            raise ValueError("`in_channels` and `out_channels` must be provided directly or via `opts`.")
        
        self.conv = Conv2d(
            in_channels=_in_channels,
            out_channels=_out_channels,
            kernel_size=_kernel_size,
            stride=_stride,
            padding=_padding,
            dilation=_dilation,
            groups=_groups,
            padding_mode=_padding_mode,
            bias=_bias
        )
        self.block.add_module(name="conv", module=self.conv)
        
        self.batchnorm = build_normalization_layer(
            opts=get_param(opts, None, 'norm', None),
            num_features=_out_channels
        )
        if self.batchnorm is not None:
            self.block.add_module(name="norm", module=self.batchnorm)
        
        self.act = build_activation_layer(opts=get_param(opts, None, 'act', None))
        if self.act is not None:
            self.block.add_module(name="act", module=self.act)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if not any(isinstance(group, argparse._ArgumentGroup) and group.title == f"Arguments for {cls.__name__}" for group in parser._action_groups):
            # 1. Nhóm cho ConvBNAct
            group_main = parser.add_argument_group(f"Arguments for {cls.__name__}")
            group_main.add_argument("--block-name", type=str, help="Tên của block này")

            parser = Conv2d.add_arguments(parser)

            group_act = parser.add_argument_group("Arguments for Activation")
            arguments_activation_fn(group_act) 

            group_norm = parser.add_argument_group("Arguments for Normalization")
            arguments_norm_layers(group_norm)

        return parser
if __name__ == "__main__":
    formatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=60, width=240)
    parser = argparse.ArgumentParser(
        description="Kiểm tra thông số của lớp ConvBNAct",
        formatter_class=formatter,
        conflict_handler='resolve'  
    )
    ConvBNAct.add_arguments(parser)
    args = parser.parse_args()