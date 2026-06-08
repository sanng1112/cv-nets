import abc
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor, nn


class BaseModel(nn.Module, abc.ABC):
    """Abstract base with standardized serialization (weights + config)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self._config = config or {}

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @abc.abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        ...

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        import cvnets
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self._config,
            "cvnets_version": cvnets.__version__,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseModel":
        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        model = cls(config=config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
