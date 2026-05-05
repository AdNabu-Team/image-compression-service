"""Tests for Settings / config.py behaviour."""

from unittest.mock import patch


def test_compression_semaphore_floor_on_single_cpu():
    """When os.cpu_count() returns 1, semaphore must be at least 2."""
    with patch("os.cpu_count", return_value=1):
        from config import Settings

        s = Settings()

    assert (
        s.compression_semaphore_size >= 2
    ), f"Expected semaphore >= 2 on 1-cpu host, got {s.compression_semaphore_size}"
