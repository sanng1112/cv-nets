"""
Training pipeline for cv-nets.

Provides ``Trainer``, metric classes, and callback implementations.
"""

from cvnets.trainer.callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    MetricsLogger,
    ModelCheckpoint,
    ProgressBar,
)
from cvnets.trainer.metrics import Accuracy, AverageLoss, MetricsTracker
from cvnets.trainer.trainer import Trainer

__all__ = [
    "Trainer",
    "Callback",
    "CallbackList",
    "EarlyStopping",
    "MetricsLogger",
    "ModelCheckpoint",
    "ProgressBar",
    "MetricsTracker",
    "Accuracy",
    "AverageLoss",
]

