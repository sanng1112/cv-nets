import argparse
import torch
from torch import nn, Tensor
from typing import Optional, Union, Tuple, Any

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
        use_norm: Optional[bool] = None,
        use_act: Optional[bool] = None,
        act_name: Optional[str] = None,
        opts: Optional[Any] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        

        if in_channels is not None and not isinstance(in_channels, int):
            opts = in_channels
            in_channels = None

        opts = opts or kwargs.get("opts", None)

        def get_param(explicit_val, attr_name, default_val):
            if explicit_val is not None:
                return explicit_val
            return getattr(opts, attr_name, default_val)

        _in_channels = get_param(in_channels, "in_channels", None)
        _out_channels = get_param(out_channels, "out_channels", None)
        _kernel_size = get_param(kernel_size, "kernel_size", 3)
        _stride = get_param(stride, "stride", 1)
        _padding = get_param(padding, "padding", 0)
        _dilation = get_param(dilation, "dilation", 1)
        _groups = get_param(groups, "groups", 1)
        _padding_mode = get_param(padding_mode, "padding_mode", "zeros")
        


        _bias = get_param(bias, "bias", False)

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
        """Thêm các argument cấu hình Conv2d cho Command Line."""
        group = parser.add_argument_group("Conv2d Arguments")
        return parser

    def forward(self, x: Tensor) -> Tensor:

        return super().forward(x)
       