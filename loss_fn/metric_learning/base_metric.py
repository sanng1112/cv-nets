import argparse
from typing import Any, Dict, Union

from torch import Tensor

from loss_fn import LOSS_REGISTRY, BaseCriteria
from utils import logger

@LOSS_REGISTRY.register(name="__base__", type="metric_learning")
class BaseMetricCriteria(BaseCriteria):
    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__(opts, *args, **kwargs)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Chi tiết hàm: `add_arguments`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        if cls != BaseMetricCriteria:
            return parser

        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--loss.metric-learning.name",
            type=str,
            default=None,
            help=f"Name of the loss function in {cls.__name__}. Defaults to None.",
        )
        return parser

    def _compute_loss(self, *args, **kwargs) -> Tensor:
        """
        Chi tiết hàm: `_compute_loss`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        raise NotImplementedError

    def forward(
        self,
        input_sample: Any,
        prediction: Union[Dict[str, Tensor], Tensor, Any],
        target: Any = None,
        *args,
        **kwargs,
    ) -> Tensor:
        # In Metric learning, prediction can be a tuple of (anchor, positive, negative)
        # or dictionary depending on the pipeline
        """
        Chi tiết hàm: `forward`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        if isinstance(prediction, Tensor) and target is not None:
             return self._compute_loss(prediction, *args, target=target, **kwargs)
        elif isinstance(prediction, dict):
            return self._compute_loss(*args, target=target, **prediction, **kwargs)
        elif isinstance(prediction, (tuple, list)):
            return self._compute_loss(*prediction, *args, target=target, **kwargs)
        else:
            return self._compute_loss(prediction, *args, target=target, **kwargs)
