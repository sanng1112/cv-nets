"""
ModelFactory — build full models from configuration dictionaries.

The factory validates the configuration, walks the ``model.layers`` list,
dispatches each layer to the appropriate builder (block registry, pooling,
activation, or fully-connected), and returns a ``_ComposedModel`` instance
with separate ``feature_extractor`` and ``classifier`` sub-modules.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor, nn

from cvnets.config.schema import ConfigSchema
from cvnets.core.base_model import BaseModel
from cvnets.core.registry import BLOCK_REGISTRY

# Layer builders from the new cvnets.layers package
from cvnets.layers.activation import build_activation_layer
from cvnets.layers.flatten import Flatten
from cvnets.layers.linear_layer import LinearLayer
from cvnets.layers.multi_head_attention import MultiHeadSelfAttention
from cvnets.layers.patch_embedding import PatchEmbedding
from cvnets.layers.pooling import build_pooling_layer


# ===================================================================
# _ComposedModel
# ===================================================================


class _ComposedModel(BaseModel):
    """A simple composed model with a feature extractor and a classifier.

    The forward pass runs the input through ``feature_extractor``, applies
    a global adaptive average pool (if the output is a 4-D feature map) to
    reduce spatial dimensions to 1×1, flattens, and finally runs through
    ``classifier``.

    Parameters
    ----------
    feature_extractor : nn.Sequential
        Layers that process spatial / feature map data (convs, pools, etc.).
    classifier : nn.Sequential
        Layers that produce the final output (flatten, linear, activations).
    config : dict or None
        The full model configuration (stored for serialisation).
    """

    def __init__(
        self,
        feature_extractor: nn.Sequential,
        classifier: nn.Sequential,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config=config)
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, x: Tensor) -> Tensor:
        x = self.feature_extractor(x)
        # Global adaptive average pool to 1×1 before classifier
        if x.dim() == 4:
            x = torch.mean(x, dim=[-2, -1], keepdim=False)
        # For 3-D token-sequence output (B, N, C), pool over sequence dim
        elif x.dim() == 3:
            x = torch.mean(x, dim=1, keepdim=False)
        # Flatten if still multi-dimensional (e.g. (B, C, 1, 1))
        if x.dim() > 2:
            x = x.flatten(1)
        x = self.classifier(x)
        return x


# ===================================================================
# ModelFactory
# ===================================================================


class ModelFactory:
    """Static factory that builds models from configuration dictionaries.

    Usage
    -----
    >>> config = {"model": {"name": "MyNet", "layers": [...]}}
    >>> model = ModelFactory.build(config)
    """

    _POOLING_TYPES = frozenset({"avgpool", "maxpool", "adaptive_avg"})

    @classmethod
    def build(cls, config: Dict[str, Any]) -> _ComposedModel:
        """Build a model from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Must contain a ``"model"`` key with a ``"layers"`` list (see
            :meth:`ConfigSchema.validate_model_config`).

        Returns
        -------
        _ComposedModel
            The constructed model.

        Raises
        ------
        ConfigValidationError
            If the configuration is structurally invalid.
        ValueError
            If a layer type is unknown.
        """
        # Ensure all blocks are registered by importing the blocks package
        import cvnets.blocks  # noqa: F401

        # 1. Validate
        ConfigSchema.validate_model_config(config)

        model_cfg = config["model"]
        layers_cfg: List[Dict[str, Any]] = model_cfg["layers"]

        feature_layers: List[nn.Module] = []
        classifier_layers: List[nn.Module] = []

        for layer_cfg in layers_cfg:
            raw_type = layer_cfg.get("type", "")
            layer_type = raw_type.lower()
            # Copy and remove type so remaining keys are forwarded
            cfg = {k: v for k, v in layer_cfg.items() if k != "type"}

            # -- Block registry -------------------------------------------------
            # Try the raw type first (CamelCase), then lowercased
            block_key = raw_type if BLOCK_REGISTRY.contains(
                raw_type, category=""
            ) else (
                layer_type if BLOCK_REGISTRY.contains(layer_type, category="")
                else None
            )
            if block_key is not None:
                module = BLOCK_REGISTRY.build(block_key, **cfg)
                feature_layers.append(module)

            # -- Fully-connected (flatten + linear) -----------------------------
            elif layer_type == "fc":
                classifier_layers.append(Flatten())
                classifier_layers.append(
                    LinearLayer(
                        in_features=cfg.get("in_features"),
                        out_features=cfg.get("out_features"),
                        bias=cfg.get("bias", False),
                    )
                )

            # -- Activation layer -----------------------------------------------
            elif layer_type == "act":
                act = build_activation_layer(
                    opts=layer_cfg,
                    act_type=cfg.get("act_type") or cfg.get("name"),
                )
                if act is not None:
                    classifier_layers.append(act)

            # -- Pooling layers -------------------------------------------------
            elif layer_type in cls._POOLING_TYPES:
                pool = build_pooling_layer(
                    opts=layer_cfg,
                    pool_type=layer_type,
                    kernel_size=cfg.get("kernel_size"),
                    stride=cfg.get("stride"),
                    padding=cfg.get("padding"),
                )
                if pool is not None:
                    feature_layers.append(pool)

            # -- Multi-Head Self-Attention ----------------------------------------
            elif layer_type == "multi_head_attention":
                module = MultiHeadSelfAttention(
                    embed_dim=cfg.get("embed_dim"),
                    num_heads=cfg.get("num_heads"),
                    dropout=cfg.get("dropout", 0.0),
                    bias=cfg.get("bias", False),
                )
                feature_layers.append(module)

            # -- Patch Embedding --------------------------------------------------
            elif layer_type == "patch_embedding":
                module = PatchEmbedding(
                    img_size=cfg.get("img_size"),
                    patch_size=cfg.get("patch_size"),
                    in_channels=cfg.get("in_channels"),
                    embed_dim=cfg.get("embed_dim"),
                )
                feature_layers.append(module)

            # -- Unknown ---------------------------------------------------------
            else:
                raise ValueError(
                    f"Unknown layer type {layer_type!r}. "
                    f"Supported types: blocks in BLOCK_REGISTRY, "
                    f"'fc', 'act', avgpool, maxpool, adaptive_avg, "
                    f"'multi_head_attention', 'patch_embedding'. "
                    f"Available blocks: {BLOCK_REGISTRY.keys()}"
                )

        return _ComposedModel(
            feature_extractor=nn.Sequential(*feature_layers),
            classifier=nn.Sequential(*classifier_layers),
            config=config,
        )
