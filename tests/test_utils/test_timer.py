"""Tests for timer utilities."""

import time

from cvnets.utils import Timer, format_duration


class TestTimer:
    def test_timer_measures_elapsed(self):
        with Timer() as t:
            time.sleep(0.01)
        assert t.elapsed >= 0.01

    def test_timer_reset(self):
        t = Timer()
        with t:
            time.sleep(0.005)
        assert t.elapsed >= 0.005
        t.reset()
        assert t.elapsed == 0.0

    def test_timer_verbose(self):
        """Verbose timer should not raise."""
        with Timer(verbose=True):
            pass  # no error

    def test_format_duration_ms(self):
        assert "ms" in format_duration(0.05)

    def test_format_duration_us(self):
        assert "us" in format_duration(0.000001)

    def test_format_duration_seconds(self):
        result = format_duration(5.5)
        assert "s" in result
