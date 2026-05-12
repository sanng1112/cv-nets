import argparse
import importlib
import os
from typing import Optional

import torch

from layers.identity import Identity
from utils import logger

SUPPORTED_NORM_FNS = []
NORM_LAYER_REGISTRY = {}
NORM_LAYER_CLS = []


def register_norm_fn(name):
    def register_fn(cls):
        if name in SUPPORTED_NORM_FNS:
            raise ValueError(
                "Cannot register duplicate normalization function ({})".format(name)
            )
        SUPPORTED_NORM_FNS.append(name)
        NORM_LAYER_REGISTRY[name] = cls
        NORM_LAYER_CLS.append(cls)
        return cls

    return register_fn

def build_normalization_layer(
    opts: argparse.Namespace,
    num_features: int = None,
    norm_type: Optional[str] = None,
    num_groups: Optional[int] = None,
    momentum: Optional[float] = None,
) -> torch.nn.Module:
    if norm_type is None:
        norm_type = getattr(opts, "type")
    if num_groups is None:
        num_groups = getattr(opts, "groups", 1)
    if momentum is None:
        momentum = getattr(opts, "momentum", 0.1)

    num_features = num_features if num_features else getattr(opts, 'num_features')

    norm_layer = None
    norm_type = norm_type.lower()

    if norm_type in NORM_LAYER_REGISTRY:
        if (
            "cuda" not in str(getattr(opts, "dev.device", "cpu"))
            and "sync_batch" in norm_type
        ):
            # for a CPU-device, Sync-batch norm does not work. So, change to batch norm
            norm_type = norm_type.replace("sync_", "")
        norm_layer = NORM_LAYER_REGISTRY[norm_type](
            normalized_shape=num_features,
            num_features=num_features,
            momentum=momentum,
            num_groups=num_groups,
        )
    elif norm_type == "identity":
        norm_layer = Identity()
    else:
        logger.error(
            "Supported normalization layer arguments are: {}. Got: {}".format(
                SUPPORTED_NORM_FNS, norm_type
            )
        )
    return norm_layer



def arguments_norm_layers(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        title="Normalization layers", description="Normalization layers"
    )

    group.add_argument(
        "--name",
        default="batch_norm",
        type=str,
        help="Normalization layer. Defaults to 'batch_norm'.",
    )
    group.add_argument(
        "--groups",
        default=1,
        type=str,
        help="Number of groups in group normalization layer. Defaults to 1.",
    )
    group.add_argument(
        "--momentum",
        default=0.1,
        type=float,
        help="Momentum in normalization layers. Defaults to 0.1",
    )

    group.add_argument(
        "--enable",
        action="store_true",
        help="Adjust momentum in batch normalization layers",
    )
    group.add_argument(
        "--anneal-type",
        default="cosine",
        type=str,
        help="Method for annealing momentum in Batch normalization layer",
    )
    group.add_argument(
        "--final-momentum-value",
        default=1e-6,
        type=float,
        help="Min. momentum in batch normalization layer",
    )

    return parser


norm_dir = os.path.dirname(__file__)
for file in os.listdir(norm_dir):
    path = os.path.join(norm_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("layers.normalization." + model_name)
