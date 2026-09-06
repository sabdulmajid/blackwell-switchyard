"""Shared test controls, including the unattended GPU collision guard."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def pytest_configure(config):
    """Start before test collection can initialize a CUDA context."""
    target_uuid = os.environ.get("SWITCHYARD_TARGET_GPU_UUID")
    if not target_uuid:
        return
    harness = Path(__file__).resolve().parents[1] / "bench" / "harness.py"
    spec = importlib.util.spec_from_file_location("switchyard_test_harness", harness)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the GPU process monitor")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monitor = module.GPUProcessMonitor(target_uuid, abort_on_collision=True)
    monitor.start()
    config._switchyard_gpu_monitor = monitor


def pytest_unconfigure(config):
    monitor = getattr(config, "_switchyard_gpu_monitor", None)
    if monitor is not None:
        report = monitor.stop()
        if report["collision_detected"] or report["probe_errors"]:
            os._exit(75)
