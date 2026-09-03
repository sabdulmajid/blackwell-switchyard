"""CPU-only checks for the installed package contract."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib import resources
from pathlib import Path

import switchyard


def test_root_exports_reference_api():
    assert callable(switchyard.block_attn_res_reference)
    assert callable(switchyard.block_attn_res_oracle)
    assert switchyard.BlockAttnRes.__module__ == "switchyard.reference"


def test_root_import_does_not_require_triton():
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src")
    code = "import sys, switchyard; assert 'triton' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_optional_cuda_source_is_packaged():
    source = resources.files("switchyard").joinpath("csrc/shared_backward.cu")
    assert source.is_file()
