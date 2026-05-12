from . import register_pooling_fn
from torch import nn


@register_pooling_fn("maxpool")
class MaxPool2d(nn.MaxPool2d):
    def __init__(self, kernel_size=None, stride=None, padding=None, opts=None, **kwargs):
        # Ưu tiên: Tham số trực tiếp > Tham số trong opts > Mặc định (2)
        _kernel_size = kernel_size if kernel_size is not None else getattr(opts, 'kernel_size', 2)
        _stride = stride if stride is not None else getattr(opts, 'stride', _kernel_size)
        _padding = padding if padding is not None else getattr(opts, 'padding', 0)
        
        super().__init__(kernel_size=_kernel_size, stride=_stride, padding=_padding)

    def __repr__(self):
        return f"{self.__class__.__name__}(kernel_size={self.kernel_size}, stride={self.stride})"