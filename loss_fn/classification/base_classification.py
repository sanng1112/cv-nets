import argparse
from typing import Any, Dict, Union

from torch import Tensor

from loss_fn import LOSS_REGISTRY, BaseCriteria
from utils import logger



@LOSS_REGISTRY.register(name="__base__", type="classification")
class BaseClassificationCriteria(BaseCriteria):
    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != BaseClassificationCriteria:
            return parser

        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--loss.classification.name",
            type=str,
            default=None,
            help=f"Name of the loss function in {cls.__name__}. Defaults to None.",
        )
        return parser

    def _compute_loss(
        self, prediction: Tensor, target: Tensor, *args, **kwargs
    ) -> Tensor:
        raise NotImplementedError

    def forward(
        self,
        input_sample: Any,
        prediction: Union[Dict[str, Tensor], Tensor],
        target: Tensor,
        *args,
        **kwargs,
    ) -> Tensor:
        if isinstance(prediction, Tensor):
            return self._compute_loss(
                prediction=prediction, target=target, *args, **kwargs
            )
        elif isinstance(prediction, Dict):
            if "logits" not in prediction:
                logger.error(
                    f"logits is a required key in {self.__class__.__name__} when prediction type"
                    f"is dictionary. Got keys: {prediction.keys()}"
                )

            predicted_logits = prediction["logits"]
            if predicted_logits is None:
                logger.error("Predicted logits can not be None.")

            ce_loss = self._compute_loss(
                prediction=predicted_logits, target=target, *args, **kwargs
            )
            return ce_loss
        else:
            logger.error(
                f"Prediction should be either a Tensor or Dictionary[str, Tensor]. Got: {type(prediction)}"
            )
