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
    """
    Chi tiết hàm: `register_act_fn`
    - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
    - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
    """
    def register_fn(cls):
        """
        Chi tiết hàm: `register_fn`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        if name in SUPPORTED_ACT_FNS:
            raise ValueError(f"Cannot register duplicate activation function ({name})")
        SUPPORTED_ACT_FNS.append(name)
        ACT_FN_MODULES[name] = cls
        return cls
    return register_fn

def arguments_activation_fn(parser: argparse.ArgumentParser):
    """
    Chi tiết hàm: `arguments_activation_fn`
    - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
    - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
    """
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
    """
    Chi tiết hàm: `get_config_prop`
    - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
    - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
    """
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
    opts: Union[argparse.Namespace, SimpleNamespace, dict, None] = None,
    act_type: Optional[str] = None,
    inplace: Optional[bool] = None,
    negative_slope: Optional[float] = None,
    num_parameters: Optional[int] = None,
    **kwargs,
) -> Optional[nn.Module]:
    # 1. Resolve act_type: explicit > opts > default
    """
    Chi tiết hàm: `build_activation_layer`
    - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
    - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
    """
    if act_type is None:
        if isinstance(opts, dict):
            act_type = opts.get("type")
        else:
            act_type = getattr(opts, "type", None) if opts is not None else None
    if not act_type:
        return None

    # 2. Resolve common params (check both underscore and hyphenated keys)
    if inplace is None:
        inplace = opts.get("inplace", False) if isinstance(opts, dict) else getattr(opts, "inplace", False)
    if negative_slope is None:
        negative_slope = (
            opts.get("neg_slope") or opts.get("neg-slope") or 0.1
        ) if isinstance(opts, dict) else getattr(opts, "neg_slope", getattr(opts, "neg-slope", 0.1))
    if num_parameters is None:
        num_parameters = opts.get("num_parameters", 1) if isinstance(opts, dict) else getattr(opts, "num_parameters", 1)

    act_type = act_type.lower()

    if act_type not in SUPPORTED_ACT_FNS:
        logger.error(
            f"Supported activation layers: {SUPPORTED_ACT_FNS}. Supplied: {act_type}"
        )
        raise NotImplementedError(f"Activation function '{act_type}' is not supported/registered.")

    act_class = ACT_FN_MODULES[act_type]

    # 3. Collect all candidate params
    raw_args = {
        "inplace": inplace,
        "negative_slope": negative_slope,
        "num_parameters": num_parameters,
    }
    raw_args.update(kwargs)

    # 4. Filter to match the class constructor signature
    import inspect
    sig = inspect.signature(act_class.__init__)
    allowed = sig.parameters

    filtered = {}
    for pname, _ in allowed.items():
        if pname in ("self", "args", "kwargs"):
            continue
        if pname in raw_args:
            filtered[pname] = raw_args[pname]

    # If class accepts **kwargs, pass everything
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in allowed.values()):
        filtered.update(raw_args)

    return act_class(**filtered)


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
