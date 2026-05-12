from . import register_pooling_fn
from torch import nn


@register_pooling_fn("adaptive_avg")
class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size=None, opts=None, **kwargs):
        # Nếu không truyền size, mặc định là 1 (Global Average Pooling)
        _output_size = output_size if output_size is not None else getattr(opts, 'output_size', 1)
        super().__init__(output_size=_output_size)