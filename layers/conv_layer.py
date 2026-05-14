import argparse
import torch
from torch import nn, Tensor
from typing import Optional, Union, Tuple, Any
from utils.config_helper import get_param



class Conv2d(nn.Conv2d):
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
        *args: Any,
        **kwargs: Any,
    ) -> None:
        

        if in_channels is not None and not isinstance(in_channels, int):
            opts = in_channels
            in_channels = None

        opts = opts or kwargs.get("opts", None)


        _in_channels = get_param(opts , in_channels, 'in_channels', None)
        _out_channels = get_param(opts, out_channels, "out_channels", None)
        _kernel_size = get_param(opts, kernel_size, "kernel_size", 3)
        _stride = get_param(opts, stride, "stride", 1)
        _padding = get_param(opts, padding, "padding", 1)
        _dilation = get_param(opts, dilation, "dilation", 1)
        _groups = get_param(opts, groups, "groups", 1)
        print(_groups)
        _padding_mode = get_param(opts, padding_mode, "padding_mode", "zeros")
        


        _bias = get_param(opts, bias, "bias", False)

        if _in_channels is None or _out_channels is None:
            raise ValueError("`in_channels` and `out_channels` must be provided directly or via `opts`.")

        super().__init__(
            in_channels=_in_channels,
            out_channels=_out_channels,
            kernel_size=_kernel_size,
            stride=_stride,
            padding=_padding,
            dilation=_dilation,
            groups=_groups,
            bias=_bias,
            padding_mode=_padding_mode,
        )


    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(f"Arguments for {cls.__name__}")

        group.add_argument(
            "--in-channels", 
            type=int, 
            metavar="N",
            help="Số lượng kênh đầu vào (in_channels)"
        )
        group.add_argument(
            "--out-channels", 
            type=int, 
            metavar="N",
            help="Số lượng kênh đầu ra (out_channels)"
        )

        group.add_argument(
            "--kernel-size", 
            type=int, 
            default=3, 
            metavar="N",
            help="Kích thước nhân (default: 3)"
        )
        group.add_argument(
            "--stride", 
            type=int, 
            default=1, 
            metavar="N",
            help="Bước nhảy (default: 1)"
        )
        group.add_argument(
            "--padding", 
            type=int, 
            default=1, 
            metavar="N",
            help="Đệm viền (default: 1)"
        )
        group.add_argument(
            "--dilation", 
            type=int, 
            default=1, 
            metavar="N",
            help="Độ giãn nở nhân (default: 1)"
        )

        # 3. groups
        group.add_argument(
            "--groups", 
            type=int, 
            default=1, 
            metavar="N",
            help="Số lượng nhóm (cho Grouped Conv, default: 1)"
        )

        # 4. padding_mode
        group.add_argument(
            "--padding-mode", 
            type=str, 
            default="zeros", 
            choices=["zeros", "reflect", "replicate", "circular"],
            help="Chế độ đệm (default: zeros)"
        )

        group.add_argument(
            "--bias", 
            action="store_true", 
            help="Sử dụng bias cho lớp Convolution"
        )

        return parser
    
    
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


if __name__ == "__main__":
    formatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=60, width=240)
    parser = argparse.ArgumentParser(
        description="Kiểm tra thông số của lớp Conv2d",
        formatter_class = formatter)
    Conv2d.add_arguments(parser)
    args = parser.parse_args()