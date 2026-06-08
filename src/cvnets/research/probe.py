"""LayerProbe — lightweight forward/backward hooks for layer introspection."""

from __future__ import annotations

from typing import Any, List, Optional
import torch
from torch import Tensor, nn


class LayerProbe:
    """Capture forward activations and backward gradients from nn.Module hooks.

    Usage
    -----
    >>> model = nn.Linear(4, 2)
    >>> probe = LayerProbe()
    >>> probe.attach(model)
    >>> x = torch.randn(3, 4)
    >>> out = model(x)
    >>> out.sum().backward()
    >>> print(len(probe.activations))   # 1
    >>> print(len(probe.gradients))     # 1
    >>> probe.clear()
    """

    def __init__(self) -> None:
        self._handles: List[Any] = []
        self.activations: List[Tensor] = []
        self.gradients: List[Tensor] = []

    def attach(self, module: nn.Module) -> None:
        """Register forward and backward hooks on *module*."""
        fwd_handle = module.register_forward_hook(self._forward_hook)
        bwd_handle = module.register_full_backward_hook(self._backward_hook)
        self._handles.append(fwd_handle)
        self._handles.append(bwd_handle)

    def detach(self) -> None:
        """Remove hooks from the most recently attached module."""
        if self._handles:
            self._handles.pop().remove()
        if self._handles:
            self._handles.pop().remove()

    def detach_all(self) -> None:
        """Remove all registered hooks."""
        while self._handles:
            self._handles.pop().remove()

    def clear(self) -> None:
        """Empty all recorded buffers."""
        self.activations.clear()
        self.gradients.clear()

    def _forward_hook(self, module: nn.Module, inp: Any, out: Any) -> None:
        if isinstance(out, Tensor):
            self.activations.append(out.detach().clone())
        elif isinstance(out, (tuple, list)):
            for o in out:
                if isinstance(o, Tensor):
                    self.activations.append(o.detach().clone())

    def _backward_hook(self, module: nn.Module, grad_in: Any, grad_out: Any) -> None:
        if isinstance(grad_out[0], Tensor):
            self.gradients.append(grad_out[0].detach().clone())

    def __enter__(self) -> "LayerProbe":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.detach_all()
