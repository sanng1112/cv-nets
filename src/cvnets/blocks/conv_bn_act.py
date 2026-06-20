"""
ConvBNAct — convolutional block with optional normalisation and activation.

Builds the sequence ``Conv2d → [BatchNorm] → [Activation]`` where the
convolution, normalisation and activation layers are each configured
via ``ConfigResolver``-compatible dictionaries passed as ``conv``,
``norm`` and ``act`` keys respectively.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from torch import Tensor, nn

from cvnets.config.resolver import ConfigResolver
from cvnets.core.base_block import BaseBlock
from cvnets.core.registry import BLOCK_REGISTRY

# Layer builders from the new cvnets.layers package
from cvnets.layers.activation import build_activation_layer
from cvnets.layers.conv_layer import Conv2d
from cvnets.layers.normalization import build_normalization_layer


@BLOCK_REGISTRY.register("conv_bn_act")
@BLOCK_REGISTRY.register("ConvBNAct")
class ConvBNAct(BaseBlock):
    """Convolution + optional BatchNorm + optional Activation.

    Accepts either direct keyword arguments or a nested config dictionary
    with ``conv``, ``norm`` and ``act`` sub-sections.

    Parameters
    ----------
    conv : dict or None
        Configuration for the ``Conv2d`` layer (e.g.
        ``{"in_channels": 3, "out_channels": 64, "kernel_size": 3}``).
    norm : dict or None
        Configuration for the normalisation layer (e.g.
        ``{"type": "batch_norm", "num_features": 64}``).
        Pass ``None`` or an empty dict to skip normalisation.
    act : dict or None
        Configuration for the activation layer (e.g.
        ``{"type": "relu"}``).  Pass ``None`` or an empty dict to skip
        activation.
    **kwargs
        Additional keyword arguments that are forwarded to the
        ``ConfigResolver``.  Useful for per-block overrides.
    """

    def __init__(
        self,
        conv: Optional[Dict[str, Any]] = None,
        norm: Optional[Dict[str, Any]] = None,
        act: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Resolve configs — allow plain dicts or ConfigResolver objects
        conv_cfg = (
            ConfigResolver(conv).to_dict() if isinstance(conv, dict) else (conv or {})
        )
        norm_cfg = (
            ConfigResolver(norm).to_dict() if isinstance(norm, dict) else (norm or {})
        )
        act_cfg = (
            ConfigResolver(act).to_dict() if isinstance(act, dict) else (act or {})
        )

        # Merge extra kwargs into conv config
        conv_cfg.update(kwargs)

        # ------------------------------------------------------------------
        # 1. Convolution
        # ------------------------------------------------------------------
        self.conv = Conv2d(
            in_channels=conv_cfg.get("in_channels"),
            out_channels=conv_cfg.get("out_channels"),
            kernel_size=conv_cfg.get("kernel_size", 3),
            stride=conv_cfg.get("stride", 1),
            padding=conv_cfg.get("padding", 1),
            dilation=conv_cfg.get("dilation", 1),
            groups=conv_cfg.get("groups", 1),
            bias=conv_cfg.get("bias", False),
        )

        # ------------------------------------------------------------------
        # 2. Normalisation (optional)
        # ------------------------------------------------------------------
        if norm_cfg and norm_cfg.get("type"):
            # Pass num_features from conv output channels if not explicit
            if "num_features" not in norm_cfg:
                norm_cfg["num_features"] = conv_cfg.get("out_channels")
            self.norm = build_normalization_layer(
                opts=norm_cfg,
                num_features=norm_cfg.get("num_features"),
            )
        else:
            self.norm = nn.Identity()

        # ------------------------------------------------------------------
        # 3. Activation (optional)
        # ------------------------------------------------------------------
        if act_cfg and act_cfg.get("type"):
            self.act = build_activation_layer(opts=act_cfg)
        else:
            self.act = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
