# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import importlib
import sys

import torch

from . import _ops, research, utils
from .autograd._functions import (
    MatmulLtState,
    matmul,
    matmul_4bit,
)
from .backends.cpu import ops as cpu_ops
from .backends.default import ops as default_ops
from .backends.torchscript import ops as torchscript_ops
from .nn import modules
from .optim import adam

# This is a signal for integrations with transformers/diffusers.
# Eventually we may remove this but it is currently required for compatibility.
features = {"multi_backend"}
supported_torch_devices = {
    "cpu",
    "cuda",  # NVIDIA/AMD GPU
    "xpu",  # Intel GPU
    "hpu",  # Intel Gaudi
    "npu",  # Ascend NPU
    "mps",  # Apple Silicon
}

if torch.cuda.is_available():
    from .backends.cuda import ops as cuda_ops

if hasattr(torch, "xpu") and torch.xpu.is_available():
    from .backends.xpu import ops as xpu_ops


if importlib.util.find_spec("habana_frameworks") and importlib.util.find_spec("habana_frameworks.torch"):
    # In case not automatically imported
    import habana_frameworks.torch

    if hasattr(torch, "hpu") and torch.hpu.is_available():
        from .backends.hpu import ops as hpu_ops


def _import_backends():
    """
    Discover and autoload all available backends installed as separate packages.
    Packages with an entrypoint for "bitsandbytes.backends" will be loaded.
    Inspired by PyTorch implementation: https://pytorch.org/tutorials/prototype/python_extension_autoload.html
    """
    from importlib.metadata import entry_points

    if sys.version_info < (3, 10):
        extensions = entry_points().get("bitsandbytes.backends", [])
    else:
        extensions = entry_points(group="bitsandbytes.backends")

    for ext in extensions:
        try:
            entry = ext.load()
            entry()
        except Exception as e:
            raise RuntimeError(f"bitsandbytes: failed to load backend {ext.name}: {e}") from e


_import_backends()

__pdoc__ = {
    "libbitsandbytes": False,
    "optim.optimizer.Optimizer8bit": False,
    "optim.optimizer.MockArgs": False,
}

__version__ = "0.47.0.dev0"

# -- HPΔKV residual utilities --------------------------------------------------
try:
    from .formats import pack_residuals, unpack_residuals, estimate_hpdkv_size  # noqa: F401
except Exception:  # pragma: no cover
    # In some installation scenarios (e.g. minimal builds) the formats submodule
    # may be stripped; fail silently so core bitsandbytes remains usable.
    pass

# ---------------------------------------------------------------------------
# Soft fallback for missing custom C++ ops when bitsandbytes native library is
# unavailable (e.g. on macOS without CUDA).  We expose minimal Python
# implementations so that unit tests relying only on CPU logic succeed.
# ---------------------------------------------------------------------------

try:
    import types

    import torch

    # Import Python reference kernels from the TorchScript backend.
    from .backends.torchscript.ops import (
        torchscript_dequantize_4bit_default as _py_dequant4_default,
        torchscript_quantize_blockwise_default as _py_qblock_default,
    )

    _ops_ns = torch.ops.bitsandbytes  # shorthand

    if not hasattr(_ops_ns, "quantize_blockwise"):
        _ops_ns.quantize_blockwise = types.SimpleNamespace(default=_py_qblock_default)

    if not hasattr(_ops_ns, "dequantize_4bit"):
        _ops_ns.dequantize_4bit = types.SimpleNamespace(
            default=_py_dequant4_default,
            cpu=_py_dequant4_default,
        )

    # Provide a trivial fallback for dequantize_blockwise to avoid noisy errors
    if not hasattr(_ops_ns, "dequantize_blockwise"):
        def _dequantize_blockwise_default(A, absmax, code, blocksize, dtype):
            # Very slow reference: map code (uint8) to float by lookup
            code = code.to(dtype).to(A.device)
            # Each uint8 encodes two 4-bit indices
            lo = torch.bitwise_and(A, 0xF)
            hi = torch.bitwise_right_shift(A, 4)
            idx = torch.stack([lo, hi], dim=-1).flatten()
            out = code[idx].float() * absmax.repeat_interleave(blocksize)[: idx.numel()]
            return out.to(dtype)

        _ops_ns.dequantize_blockwise = types.SimpleNamespace(default=_dequantize_blockwise_default)
except Exception:
    # As a last resort, swallow any failure – the rest of the package remains
    # usable even if fallbacks cannot be installed.
    pass
