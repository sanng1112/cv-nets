"""
ONNX export utilities for cv-nets.

Provides ``export_to_onnx()`` to convert a PyTorch model to ONNX format
with optional dynamic batching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor, nn

from cvnets.utils.logger import info


def export_to_onnx(
    model: nn.Module,
    input_sample: Tensor,
    output_path: str,
    opset_version: int = 17,
    dynamic_batch: bool = True,
    input_names: Optional[list[str]] = None,
    output_names: Optional[list[str]] = None,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    verbose: bool = True,
    **kwargs: Any,
) -> str:
    """Export a PyTorch model to ONNX format.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to export. It will be moved to CPU for export.
    input_sample : Tensor
        A sample input tensor that matches the model's expected input shape.
    output_path : str
        Filesystem path for the output ``.onnx`` file.
    opset_version : int
        ONNX opset version to target (default 17).
    dynamic_batch : bool
        If ``True``, the first dimension (batch) is marked as dynamic so that
        the exported ONNX model accepts variable batch sizes.  Ignored if
        *dynamic_axes* is also provided.
    input_names : list of str, optional
        Names for the input tensors.  Defaults to ``["input"]``.
    output_names : list of str, optional
        Names for the output tensors.  Defaults to ``["output"]``.
    dynamic_axes : dict, optional
        Fine-grained dynamic axis specification.  If provided, *dynamic_batch*
        is ignored.
    verbose : bool
        If ``True``, log progress via ``cvnets.utils.info()``.
    **kwargs
        Additional keyword arguments forwarded to ``torch.onnx.export()``.

    Returns
    -------
    str
        The absolute path to the exported ONNX file.

    Raises
    ------
    RuntimeError
        If the export fails.
    FileNotFoundError
        If the exported file is not found after export.
    """
    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]

    # Move model to CPU for export
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    if str(device) != "cpu":
        if verbose:
            info(f"Moving model from {device} to CPU for ONNX export")
        model_cpu = model.cpu()
    else:
        model_cpu = model

    # Ensure input sample is on CPU
    input_sample_cpu = input_sample.cpu()

    # Build dynamic axes if not provided
    if dynamic_axes is None and dynamic_batch:
        dynamic_axes = {
            input_names[0]: {0: "batch_size"},
            output_names[0]: {0: "batch_size"},
        }

    output_path = str(Path(output_path).resolve())

    if verbose:
        info(
            f"Exporting model to ONNX (opset={opset_version}, "
            f"dynamic_batch={dynamic_batch})..."
        )

    try:
        torch.onnx.export(
            model_cpu,
            input_sample_cpu,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            **kwargs,
        )
    except Exception as exc:
        raise RuntimeError(f"ONNX export failed: {exc}") from exc

    # Restore model state
    if was_training:
        model.train()
    if str(device) != "cpu":
        model.to(device)

    # Validate output file
    out_file = Path(output_path)
    if not out_file.is_file():
        raise FileNotFoundError(f"ONNX export file not found at {output_path}")

    file_size_mb = out_file.stat().st_size / (1024 * 1024)
    if verbose:
        info(
            f"ONNX export successful → {output_path} "
            f"({file_size_mb:.2f} MB)"
        )

    return output_path
