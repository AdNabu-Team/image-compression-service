import os
import sys

import pytest

from bench.runner.report.json_writer import _available_cpu_count


def test_available_cpu_count_positive():
    assert _available_cpu_count() >= 1


@pytest.mark.skipif(sys.platform != "linux", reason="sched_getaffinity is Linux-only")
def test_available_cpu_count_matches_sched_getaffinity_on_linux():
    assert _available_cpu_count() == len(os.sched_getaffinity(0))
