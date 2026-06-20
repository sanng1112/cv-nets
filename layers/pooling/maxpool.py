from . import register_pooling_fn
from torch import nn


@register_pooling_fn("maxpool")
class MaxPool2d(nn.MaxPool2d):
    def __init__(self, kernel_size=None, stride=None, padding=None, opts=None, **kwargs):
        # Ưu tiên: Tham số trực tiếp > Tham số trong opts > Mặc định (2)
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        _kernel_size = kernel_size if kernel_size is not None else getattr(opts, 'kernel_size', 2)
        _stride = stride if stride is not None else getattr(opts, 'stride', _kernel_size)
        _padding = padding if padding is not None else getattr(opts, 'padding', 0)
        
        super().__init__(kernel_size=_kernel_size, stride=_stride, padding=_padding)

    def __repr__(self):
        """
        Chi tiết hàm: `__repr__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return f"{self.__class__.__name__}(kernel_size={self.kernel_size}, stride={self.stride})"