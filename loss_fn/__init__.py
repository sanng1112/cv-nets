#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Optional

from loss_fn.base_criteria import BaseCriteria
from utils import logger
from utils.registry import Registry


LOSS_REGISTRY = Registry(
    registry_name="loss_functions",
    base_class=BaseCriteria,
    lazy_load_dirs=["loss_fn"],
    internal_dirs=["internal", "internal/projects/*"],
)


def build_loss_fn(
    opts: argparse.Namespace, category: Optional[str] = "", *args, **kwargs
) -> BaseCriteria:
    if not category:
        category = getattr(opts, "loss.category")

    if category is None:
        logger.error(
            "Please specify loss name using --loss.category. For composite loss function, see configuration"
            "example in `loss_fns.composite_loss.CompositeLoss`. Got None"
        )

    if hasattr(opts, f"loss.{category}.name"):
        loss_fn_name = getattr(opts, f"loss.{category}.name")
    else:
        loss_fn_name = category
    if loss_fn_name == "__base__":
        logger.error("__base__ can't be used as a loss function name. Please check.")

    loss_fn = LOSS_REGISTRY[loss_fn_name, category](opts, *args, **kwargs)
    return loss_fn


def add_loss_fn_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = BaseCriteria.add_arguments(parser=parser)

    parser = LOSS_REGISTRY.all_arguments(parser)
    return parser
