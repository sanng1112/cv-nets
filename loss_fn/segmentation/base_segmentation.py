import argparse
from typing import Any, Dict, Union

from torch import Tensor

from loss_fn import LOSS_REGISTRY, BaseCriteria
from utils import logger

@LOSS_REGISTRY.register(name="__base__", type="segmentation")
class BaseSegmentationCriteria(BaseCriteria):
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
        if cls != BaseSegmentationCriteria:
            return parser

        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--loss.segmentation.name",
            type=str,
            default=None,
            help=f"Name of the loss function in {cls.__name__}. Defaults to None.",
        )
        return parser

    def _compute_loss(self, prediction: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """
        Chi tiết hàm: `_compute_loss`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        raise NotImplementedError

    def forward(
        self,
        input_sample: Any,
        prediction: Union[Dict[str, Tensor], Tensor],
        target: Tensor,
        *args,
        **kwargs,
    ) -> Tensor:
        """
        Chi tiết hàm: `forward`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        if isinstance(prediction, Tensor):
            return self._compute_loss(prediction=prediction, target=target, *args, **kwargs)
        elif isinstance(prediction, Dict):
            if "out" not in prediction:
                logger.error(f"'out' is a required key in {self.__class__.__name__} when prediction type is dictionary.")
            return self._compute_loss(prediction=prediction["out"], target=target, *args, **kwargs)
        else:
            logger.error("Prediction should be either a Tensor or Dictionary.")
