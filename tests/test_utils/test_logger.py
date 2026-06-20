"""
Tests for the logger module.

Verifies that ``error()`` raises ``LoggerError`` (instead of ``sys.exit``)
and that regular log functions print to stdout.
"""

from __future__ import annotations

import pytest

from cvnets.utils.logger import LoggerError, error, log


class TestLogger:
    """Test suite for the logger module."""

    def test_error_raises_exception(self) -> None:
        """Calling ``error()`` must raise ``LoggerError``."""
        with pytest.raises(LoggerError):
            error("This is a test error")

    def test_log_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``log()`` should print a message to stdout."""
        test_msg = "Hello from log"
        log(test_msg)
        captured = capsys.readouterr()
        assert test_msg in captured.out
        assert "LOGS" in captured.out or "logs" in captured.out
