"""CPU-only contract checks for the exact-work Liger-style comparator."""

from __future__ import annotations

from pathlib import Path

import pytest

from switchyard._liger_exact import _launch_configuration

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("n", "d", "expected"),
    [
        (1, 257, (4, 512, 4)),
        (9, 2049, (16, 4096, 8)),
        (17, 4097, (32, 8192, 16)),
        (32, 8192, (32, 8192, 16)),
    ],
)
def test_liger_exact_launch_matches_the_disclosed_liger_geometry(n, d, expected):
    assert _launch_configuration(n, d) == expected


def test_liger_exact_rejects_more_than_32_sources():
    with pytest.raises(ValueError, match="at most 32 sources"):
        _launch_configuration(33, 2048)


def test_liger_derivative_keeps_required_bsd_attribution():
    source = (REPO / "src" / "switchyard" / "_liger_exact.py").read_text()
    notice = (REPO / "NOTICE").read_text()
    license_text = (REPO / "licenses" / "BSD-2-Clause-Liger-Kernel.txt").read_text()
    assert "Copyright 2024 LinkedIn Corporation" in source
    assert "Liger-Kernel" in notice
    assert "Copyright 2024 LinkedIn Corporation" in license_text
