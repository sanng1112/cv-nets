import argparse
import importlib
import os
from typing import Optional, Union, Any
from types import SimpleNamespace
import torch.nn as nn

from utils import logger


SUPPORTED_ACT_FNS = []
ACT_FN_MODULES = {}


def register_act_fn(name: str):
    def register_fn(cls):
        if name in SUPPORTED_ACT_FNS:
            raise ValueError(f"Cannot register duplicate activation function ({name})")
        SUPPORTED_ACT_FNS.append(name)
        ACT_FN_MODULES[name] = cls
        return cls
    return register_fn

def arguments_activation_fn(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        title="Non-linear functions", description="Non-linear functions"
    )
    group.add_argument(
        "--type",
        default="relu",
        type=str,
        help="Non-linear function name",
    )
    group.add_argument(
        "--inplace",
        action="store_true",
        help="Use non-linear functions inplace",
    )
    group.add_argument(
        "--neg-slope",
        default=0.1,
        type=float,
        help="Negative slope in leaky relu function",
    )
    return parser


def get_config_prop(opts: Any, prop_path: str, default: Any = None) -> Any:
    try:
        parts = prop_path.split('.')
        for part in parts:
            # Hỗ trợ cả dict và object (Namespace)
            if isinstance(opts, dict):
                opts = opts.get(part)
            else:
                opts = getattr(opts, part)
        return opts if opts is not None else default
    except AttributeError:
        return default

def build_activation_layer(
    opts: Union[argparse.Namespace, SimpleNamespace],
    act_type: Optional[str] = None,
    inplace: Optional[bool] = None,
    negative_slope: Optional[float] = None,
    num_parameters: Optional[int] = 1,
) -> Optional[nn.Module]:
    
    if act_type is None:
        act_type = getattr(opts, "type", None) 
    
    if not act_type:
        return None

    if inplace is None:
        inplace = getattr(opts, "inplace", False)
        
    if negative_slope is None:
        negative_slope = getattr(opts, "neg_slope", getattr(opts, "neg-slope", 0.1))
    
    act_type = act_type.lower()

    if act_type in SUPPORTED_ACT_FNS:
        return ACT_FN_MODULES[act_type](
            inplace=inplace, 
            negative_slope=negative_slope, 
            num_parameters=num_parameters
        )
    
    logger.error(
        f"Supported activation layers: {SUPPORTED_ACT_FNS}. Supplied: {act_type}"
    )
    raise NotImplementedError(f"Activation function '{act_type}' is not supported/registered.")


act_dir = os.path.dirname(__file__)
for file in os.listdir(act_dir):
    path = os.path.join(act_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        try:
            importlib.import_module("layers.activation." + model_name)
        except Exception as e:
            logger.warning(f"Failed to auto-import module '{model_name}': {e}")
