"""
TorchScript export utilities for cv-nets.

Provides ``export_to_torchscript()`` to convert a PyTorch model to
TorchScript via either tracing or scripting.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor, nn

from cvnets.utils.logger import info


def export_to_torchscript(
    model: nn.Module,
    input_sample: Tensor,
    output_path: str,
    method: str = "trace",
    verbose: bool = True,
    **kwargs: Any,
) -> str:
    """Export a PyTorch model to TorchScript format.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to export.  It will be moved to CPU for export.
    input_sample : Tensor
        A sample input tensor used for tracing.  Required when *method* is
        ``"trace"``; ignored for ``"script"`` (but still recommended for
        shape validation).
    output_path : str
        Filesystem path for the output ``.pt`` file.
    method : str
        Either ``"trace"`` (``torch.jit.trace``) or ``"script"``
        (``torch.jit.script``).  Default is ``"trace"``.
    verbose : bool
        If ``True``, log progress via ``cvnets.utils.info()``.
    **kwargs
        Additional keyword arguments forwarded to ``torch.jit.trace()`` or
        ``torch.jit.script()``.

    Returns
    -------
    str
        The absolute path to the exported TorchScript file.

    Raises
    ------
    ValueError
        If *method* is not ``"trace"`` or ``"script"``.
    RuntimeError
        If the export fails.
    FileNotFoundError
        If the exported file is not found after export.
    """
    method = method.lower().strip()
    if method not in ("trace", "script"):
        raise ValueError(
            f"Unknown TorchScript export method {method!r}. "
            f"Expected 'trace' or 'script'."
        )

    # Move model to CPU for export
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    if str(device) != "cpu":
        if verbose:
            info(f"Moving model from {device} to CPU for TorchScript export")
        model_cpu = model.cpu()
    else:
        model_cpu = model

    output_path = str(Path(output_path).resolve())

    if verbose:
        info(f"Exporting model to TorchScript via {method!r} ...")

    try:
        with warnings.catch_warnings():
            # Suppress deprecation warnings for torch.jit APIs (still widely used)
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.jit")
            if method == "trace":
                input_sample_cpu = input_sample.cpu()
                traced_script_module = torch.jit.trace(
                    model_cpu, input_sample_cpu, **kwargs
                )
                traced_script_module.save(output_path)
            else:
                scripted_module = torch.jit.script(model_cpu, **kwargs)
                scripted_module.save(output_path)
    except Exception as exc:
        raise RuntimeError(f"TorchScript export failed: {exc}") from exc

    # Restore model state
    if was_training:
        model.train()
    if str(device) != "cpu":
        model.to(device)

    # Validate output file
    out_file = Path(output_path)
    if not out_file.is_file():
        raise FileNotFoundError(
            f"TorchScript export file not found at {output_path}"
        )

    file_size_mb = out_file.stat().st_size / (1024 * 1024)
    if verbose:
        info(
            f"TorchScript export ({method}) successful → "
            f"{output_path} ({file_size_mb:.2f} MB)"
        )

    return output_path
