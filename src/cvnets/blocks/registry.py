"""
Block registry — factory dispatch via ``BLOCK_REGISTRY``.

Usage
-----
>>> from cvnets.blocks.registry import build_block
>>> block = build_block({"type": "conv_bn_act", "conv": {...}, ...}, some_extra=42)
"""

from __future__ import annotations

from typing import Any, Dict

from cvnets.config.resolver import ConfigResolver
from cvnets.core.registry import BLOCK_REGISTRY


def build_block(block_config: Dict[str, Any], **kwargs: Any) -> Any:
    """Build a block from a configuration dictionary.

    Parameters
    ----------
    block_config : dict
        Must contain a ``"type"`` key.  Additional keys are passed as
        keyword arguments to the registered block constructor.
    **kwargs
        Extra keyword arguments forwarded to the block constructor.

    Returns
    -------
    Any
        An instance of the registered block class.

    Raises
    ------
    ValueError
        If *block_config* is missing the ``"type"`` field.
    KeyError
        If the block type is not registered in ``BLOCK_REGISTRY``.
    """
    config = ConfigResolver(block_config)
    block_type = config.get("type")
    if not block_type:
        raise ValueError("Block config must contain a 'type' field.")

    # Merge config (minus type) with extra kwargs
    resolved = block_config.copy()
    resolved.pop("type", None)
    resolved.update(kwargs)

    return BLOCK_REGISTRY.build(block_type.lower(), **resolved)
