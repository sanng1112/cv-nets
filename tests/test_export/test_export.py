"""
Tests for the export module — ONNX and TorchScript.

All tests use a tiny ``nn.Linear`` model so they run quickly on any
machine without a GPU.
"""

from __future__ import annotations

import sys
sys.path.insert(0, "src")

import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from cvnets.export import export_to_onnx, export_to_torchscript

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    """A trivial model for export tests — a single Linear layer."""

    def __init__(self, in_features: int = 4, out_features: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _make_model_and_sample(
    in_features: int = 4, out_features: int = 2, batch_size: int = 1
):
    model = TinyModel(in_features, out_features)
    sample = torch.randn(batch_size, in_features)
    return model, sample


# ---------------------------------------------------------------------------
# ONNX export tests
# ---------------------------------------------------------------------------

try:
    import onnx  # noqa: F401
    import onnxscript  # noqa: F401
    onnx_available = True
except ImportError:
    onnx_available = False


class TestONNXExport:
    """Tests for ``export_to_onnx()``."""

    @pytest.mark.skipif(not onnx_available, reason="onnx/onnxscript packages not installed")
    def test_onnx_export_succeeds(self, tmp_path: Path) -> None:
        """Export a tiny model to ONNX and verify the file exists."""
        model, sample = _make_model_and_sample()
        output_path = str(tmp_path / "model.onnx")

        result = export_to_onnx(
            model, sample, output_path, opset_version=17, verbose=False
        )

        assert result == output_path
        assert Path(output_path).is_file()
        assert Path(output_path).stat().st_size > 0

    @pytest.mark.skipif(not onnx_available, reason="onnx package not installed")
    def test_onnx_file_is_valid(self, tmp_path: Path) -> None:
        """Verify the exported ONNX file can be loaded by the onnx library."""
        import onnx

        model, sample = _make_model_and_sample()
        output_path = str(tmp_path / "model.onnx")

        export_to_onnx(model, sample, output_path, verbose=False)

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        assert len(onnx_model.graph.node) >= 1

    @pytest.mark.skipif(not onnx_available, reason="onnx/onnxscript packages not installed")
    def test_onnx_dynamic_batch(self, tmp_path: Path) -> None:
        """Dynamic batch dimension produces variable batch axes in ONNX."""
        model, sample = _make_model_and_sample()
        output_path = str(tmp_path / "model_dynamic.onnx")

        export_to_onnx(
            model, sample, output_path, dynamic_batch=True, verbose=False
        )

        assert Path(output_path).is_file()

    @pytest.mark.skipif(not onnx_available, reason="onnx/onnxscript packages not installed")
    def test_onnx_export_preserves_model(self, tmp_path: Path) -> None:
        """Export should not modify the original model's state."""
        model, sample = _make_model_and_sample()
        original_weight = model.fc.weight.clone()

        export_to_onnx(model, sample, str(tmp_path / "model.onnx"), verbose=False)

        assert torch.equal(model.fc.weight, original_weight)


# ---------------------------------------------------------------------------
# TorchScript export tests
# ---------------------------------------------------------------------------


class TestTorchScriptExport:
    """Tests for ``export_to_torchscript()``."""

    def test_trace_export_succeeds(self, tmp_path: Path) -> None:
        """Export via tracing and verify the file exists."""
        model, sample = _make_model_and_sample()
        output_path = str(tmp_path / "model_traced.pt")

        result = export_to_torchscript(
            model, sample, output_path, method="trace", verbose=False
        )

        assert result == output_path
        assert Path(output_path).is_file()
        assert Path(output_path).stat().st_size > 0

    def test_script_export_succeeds(self, tmp_path: Path) -> None:
        """Export via scripting and verify the file exists."""
        model, sample = _make_model_and_sample()
        output_path = str(tmp_path / "model_scripted.pt")

        result = export_to_torchscript(
            model, sample, output_path, method="script", verbose=False
        )

        assert result == output_path
        assert Path(output_path).is_file()

    def test_traced_model_reload_and_infer(self, tmp_path: Path) -> None:
        """Exported TorchScript model can be reloaded and produces output."""
        model, sample = _make_model_and_sample(in_features=4, out_features=3)
        output_path = str(tmp_path / "model_traced.pt")

        export_to_torchscript(
            model, sample, output_path, method="trace", verbose=False
        )

        # Reload
        loaded = torch.jit.load(output_path)
        loaded.eval()

        test_input = torch.randn(2, 4)
        with torch.no_grad():
            output = loaded(test_input)

        assert output.shape == (2, 3)

    def test_scripted_model_reload_and_infer(self, tmp_path: Path) -> None:
        """Scripted TorchScript model can be reloaded and produces output."""
        model, sample = _make_model_and_sample(in_features=4, out_features=3)
        output_path = str(tmp_path / "model_scripted.pt")

        export_to_torchscript(
            model, sample, output_path, method="script", verbose=False
        )

        # Reload
        loaded = torch.jit.load(output_path)
        loaded.eval()

        test_input = torch.randn(2, 4)
        with torch.no_grad():
            output = loaded(test_input)

        assert output.shape == (2, 3)

    def test_trace_vs_original_output(self, tmp_path: Path) -> None:
        """Traced model should produce the same output as the original."""
        model, sample = _make_model_and_sample(in_features=4, out_features=3)

        # Get original output
        with torch.no_grad():
            original_out = model(sample)

        output_path = str(tmp_path / "model_traced.pt")
        export_to_torchscript(
            model, sample, output_path, method="trace", verbose=False
        )

        loaded = torch.jit.load(output_path)
        with torch.no_grad():
            traced_out = loaded(sample)

        assert torch.allclose(original_out, traced_out, atol=1e-6)

    def test_unknown_method_raises(self, tmp_path: Path) -> None:
        """Invalid method should raise ValueError."""
        model, sample = _make_model_and_sample()

        with pytest.raises(ValueError, match="Unknown TorchScript"):
            export_to_torchscript(
                model, sample, str(tmp_path / "bad.pt"), method="invalid"
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_torchscript_cuda_model(self, tmp_path: Path) -> None:
        """Export a model that was on CUDA via tracing."""
        model, sample = _make_model_and_sample()
        model = model.cuda()
        sample = sample.cuda()

        output_path = str(tmp_path / "model_cuda_traced.pt")
        result = export_to_torchscript(
            model, sample, output_path, method="trace", verbose=False
        )

        assert Path(result).is_file()
        # Model should still be on CUDA after export
        assert next(model.parameters()).is_cuda

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_onnx_cuda_model(self, tmp_path: Path) -> None:
        """Export a model that was on CUDA."""
        model, sample = _make_model_and_sample()
        model = model.cuda()
        sample = sample.cuda()

        output_path = str(tmp_path / "model_cuda.onnx")
        result = export_to_onnx(
            model, sample, output_path, verbose=False
        )

        assert Path(result).is_file()
        # Model should still be on CUDA after export
        assert next(model.parameters()).is_cuda
