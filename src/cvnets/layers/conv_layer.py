from typing import Optional, Union, Tuple, Any
from torch import Tensor, nn


class Conv2d(nn.Conv2d):
    """2D convolution layer with explicit parameter support.

    **Note**: This acts exactly like `torch.nn.Conv2d` but forces explicit keyword
    arguments `in_channels` and `out_channels` to make configuration more robust.

    **Example Usage**:
    ```python
    import torch
    from cvnets.layers import Conv2d

    layer = Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    x = torch.randn(2, 3, 32, 32) # Batch 2, 3 channels, 32x32
    out = layer(x)
    print(out.shape) # Output: torch.Size([2, 16, 32, 32])
    ```

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple
        Kernel size (default ``3``).
    stride : int or tuple
        Stride (default ``1``).
    padding : int or tuple
        Padding (default ``1``).
    dilation : int or tuple
        Dilation (default ``1``).
    groups : int
        Number of groups (default ``1``).
    bias : bool
        Whether to use a bias (default ``False``).
    """

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
        *args: Any,
        **kwargs: Any,
    ) -> None:
        _in_channels = in_channels
        _out_channels = out_channels
        _kernel_size = kernel_size if kernel_size is not None else 3
        _stride = stride if stride is not None else 1
        _padding = padding if padding is not None else 1
        _dilation = dilation if dilation is not None else 1
        _groups = groups if groups is not None else 1
        _bias = bias if bias is not None else False

        if _in_channels is None or _out_channels is None:
            raise ValueError(
                "`in_channels` and `out_channels` must be provided."
            )

        super().__init__(
            in_channels=_in_channels,
            out_channels=_out_channels,
            kernel_size=_kernel_size,
            stride=_stride,
            padding=_padding,
            dilation=_dilation,
            groups=_groups,
            bias=_bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)
