import argparse
import importlib
import os
from typing import Optional, Union, Any
from types import SimpleNamespace
import torch.nn as nn

from utils import logger


SUPPORTED_POOLING_LAYERS = []
POOLING_LAYER_REGISTRY = {}


def register_pooling_fn(name: str):
    def register_fn(cls):
        if name in SUPPORTED_POOLING_LAYERS:
            raise ValueError(f"Cannot register duplicate pooling function ({name})")
        SUPPORTED_POOLING_LAYERS.append(name)
        POOLING_LAYER_REGISTRY[name] = cls
        return cls
    return register_fn

def arguments_pooling_fn(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        title="Pooling layer arguments", 
        description="Parameters for Max, Avg, and Adaptive Pooling"
    )
    group.add_argument(
        "--model.layer.pooling.type",
        default="max",
        type=str,
        help="Type of pooling (e.g., max, avg, adaptive_avg, global_pool)",
    )
    group.add_argument(
        "--model.layer.pooling.kernel-size",
        default=2,
        type=int, # Hoặc dùng str để parse tuple
        help="Kernel size for pooling",
    )
    group.add_argument(
        "--model.layer.pooling.stride",
        default=2,
        type=int,
        help="Stride for pooling",
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

def build_pooling_layer(
    opts: Any,
    pool_type: Optional[str] = None,
    kernel_size: Optional[Any] = None,
    stride: Optional[Any] = None,
    padding: Optional[Any] = None,
    **kwargs
) -> Optional[nn.Module]:
    
    # Ưu tiên pool_type truyền trực tiếp, sau đó đến cấu hình trong opts
    if pool_type is None:
        pool_type = getattr(opts, "type", None)
        # Nếu opts là một cấu hình lồng nhau (nested), bạn có thể dùng hàm get_config_prop bạn đã viết
    
    if not pool_type:
        logger.error("Pooling type must be specified.")
        return None

    pool_type = pool_type.lower()

    if pool_type in SUPPORTED_POOLING_LAYERS:
        # Khởi tạo layer từ Registry, truyền opts để layer tự lấy thêm các params khác
        return POOLING_LAYER_REGISTRY[pool_type](
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            opts=opts,
            **kwargs
        )
    
    logger.error(
        f"Supported pooling layers: {SUPPORTED_POOLING_LAYERS}. Supplied: {pool_type}"
    )
    raise NotImplementedError(f"Pooling type '{pool_type}' is not supported.")

pooling_dir = os.path.dirname(__file__)
for file in os.listdir(pooling_dir):
    path = os.path.join(pooling_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        try:
            importlib.import_module("layers.pooling." + model_name)
        except Exception as e:
            logger.warning(f"Failed to auto-import module '{model_name}': {e}")
