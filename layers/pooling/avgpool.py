from . import register_pooling_fn
from torch import nn


@register_pooling_fn("avgpool")
class AvgPool2d(nn.AvgPool2d):
    def __init__(self, kernel_size=None, stride=None, padding=None, opts=None, **kwargs):
        _kernel_size = kernel_size if kernel_size is not None else getattr(opts, 'kernel_size', (2, 2))
        _stride = stride if stride is not None else getattr(opts, 'stride', _kernel_size)
        _padding = padding if padding is not None else getattr(opts, 'padding', 0)
        
        # Sửa lỗi logic các params đặc thù của AvgPool
        _ceil_mode = getattr(opts, 'ceil_mode', False)
        _count_include_pad = getattr(opts, 'count_include_pad', True)
        
        super().__init__(
            kernel_size=_kernel_size, 
            stride=_stride, 
            padding=_padding, 
            ceil_mode=_ceil_mode, 
            count_include_pad=_count_include_pad
        )

    def __repr__(self):
        # Đã sửa từ upscale_factor thành kernel_size
        return f"{self.__class__.__name__}(kernel_size={self.kernel_size}, stride={self.stride})"