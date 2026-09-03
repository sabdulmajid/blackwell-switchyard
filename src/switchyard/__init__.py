"""Block Attention Residuals reference code and optimized GPU operators."""

from __future__ import annotations

from .reference import BlockAttnRes, block_attn_res_oracle, block_attn_res_reference

__version__ = "0.0.1"

__all__ = [
    "__version__",
    "BlockAttnRes",
    "BlockAttnResTriton",
    "block_attn_res_batched",
    "block_attn_res_oracle",
    "block_attn_res_reference",
    "block_attn_res_triton",
]


def __getattr__(name: str):
    """Load Triton only when a GPU API is requested.

    The reference API therefore remains importable in CPU-only environments
    that do not install Triton.
    """
    if name in {"BlockAttnResTriton", "block_attn_res_batched", "block_attn_res_triton"}:
        from .triton_op import (
            BlockAttnResTriton,
            block_attn_res_batched,
            block_attn_res_triton,
        )

        return {
            "BlockAttnResTriton": BlockAttnResTriton,
            "block_attn_res_batched": block_attn_res_batched,
            "block_attn_res_triton": block_attn_res_triton,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
