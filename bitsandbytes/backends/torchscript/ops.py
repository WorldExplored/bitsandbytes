# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from typing import Optional, List

import torch

from ..._ops import register_kernel


# Lookup tables for supported 4-bit formats
# NF4 lookup table (same as defined in functional.py)
NF4_CODEBOOK = torch.tensor([
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
], dtype=torch.float32)

# Uniform FP4 lookup table (linear mapping between -1 and 1)
# FP4 as implemented in the CUDA backend uses 16 linearly spaced values in [-1, 1].
FP4_CODEBOOK = torch.linspace(-1.0, 1.0, 16, dtype=torch.float32)


# Shared helper that unpacks one quantization block and performs the LUT lookup.
# Having this as a separate scripted function avoids recompiling the nibble-unpacking
# logic for every data-type variant (NF4/FP4).
@torch.jit.script
def _dequant_block_torchscript(
    quantized_block: torch.Tensor,
    scale: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    """
    Dequantize a single block of NF4 quantized data.
    
    Args:
        quantized_block: uint8 tensor with packed 4-bit values
        scale: scalar tensor with block scale
        codebook: 16-element NF4 lookup table
        
    Returns:
        Dequantized float tensor
    """
    # Unpack 4-bit nibbles from uint8
    # Each uint8 contains two 4-bit values: higher nibble and lower nibble
    flat_quantized = quantized_block.flatten()
    n_packed = flat_quantized.numel()
    
    # Create output tensor for unpacked values (2x the packed size)
    unpacked = torch.zeros(n_packed * 2, dtype=torch.int32, device=quantized_block.device)
    
    # Extract nibbles using bitwise operations
    # Lower nibble: value & 0xF
    # Higher nibble: (value >> 4) & 0xF
    unpacked[0::2] = torch.bitwise_and(flat_quantized.int(), 0xF)  # Lower nibbles
    unpacked[1::2] = torch.bitwise_and(torch.bitwise_right_shift(flat_quantized.int(), 4), 0xF)  # Higher nibbles
    
    # Lookup in codebook and scale
    dequantized = codebook[unpacked] * scale
    
    return dequantized


@torch.jit.script
def dequant_nf4_torchscript(
    quantized: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    shape: List[int],
    dtype: torch.dtype,
    codebook: torch.Tensor,
) -> torch.Tensor:
    """
    TorchScript implementation of NF4 dequantization with manual chunking.
    
    Args:
        quantized: Packed uint8 tensor with 4-bit quantized weights
        absmax: Per-block scaling factors
        blocksize: Number of elements per quantization block
        shape: Target output shape
        dtype: Target output dtype
        codebook: 16-element NF4 lookup table
        
    Returns:
        Dequantized tensor with target shape and dtype
    """
    # Ensure codebook is on the right device
    codebook = codebook.to(quantized.device)
    
    # Calculate total elements and blocks
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    
    blocks_count = absmax.numel()
    elements_per_packed = 2  # Each uint8 holds 2 nibbles
    
    # Process each block manually (TorchScript doesn't support list.append Tensor? -> ensure list[Tensor])
    dequantized_blocks: List[torch.Tensor] = []
    
    quantized_flat = quantized.flatten()
    
    for block_idx in range(blocks_count):
        start_element = block_idx * blocksize
        end_element = min(start_element + blocksize, total_elements)
        actual_block_size = end_element - start_element
        
        start_packed = start_element // elements_per_packed
        end_packed = (end_element + elements_per_packed - 1) // elements_per_packed
        
        block_packed = quantized_flat[start_packed:end_packed]
        block_scale = absmax[block_idx]
        
        block_dequantized = _dequant_block_torchscript(block_packed, block_scale, codebook)
        
        if actual_block_size < blocksize:
            block_dequantized = block_dequantized[:actual_block_size]
        else:
            block_dequantized = block_dequantized[:blocksize]
            
        dequantized_blocks.append(block_dequantized)
    
    result = torch.cat(dequantized_blocks, dim=0)
    result = result[:total_elements]
    return result.reshape(torch.Size(shape)).to(dtype)


@register_kernel("bitsandbytes::dequantize_4bit", "cpu")
def torchscript_dequantize_4bit(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    TorchScript backend for 4-bit dequantization.
    
    Currently supports NF4 format only.
    """
    torch._check_is_size(blocksize)
    torch._check(quant_type in ["nf4", "fp4"], lambda: f"TorchScript backend only supports NF4/FP4, got {quant_type}")
    torch._check(
        dtype in [torch.bfloat16, torch.float16, torch.float32],
        lambda: f"TorchScript 4bit dequantization only supports 16/32-bit floats, but got {dtype}",
    )
    
    # Ensure input is uint8
    if A.dtype != torch.uint8:
        A = A.view(torch.uint8)
    
    # Convert shape sequence to list for TorchScript compatibility
    target_shape = list(shape)
    
    # Select appropriate codebook
    if quant_type == "nf4":
        codebook = NF4_CODEBOOK.to(dtype).to(A.device)
        return dequant_nf4_torchscript(A, absmax, blocksize, target_shape, dtype, codebook)
    else:  # "fp4"
        codebook = FP4_CODEBOOK.to(dtype).to(A.device)
        return dequant_fp4_torchscript(A, absmax, blocksize, target_shape, dtype, codebook)


@register_kernel("bitsandbytes::dequantize_4bit.out", "cpu")
def torchscript_dequantize_4bit_out(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
    out: torch.Tensor,
) -> None:
    """
    TorchScript backend for 4-bit dequantization with pre-allocated output.
    """
    torch._check(out.shape == tuple(shape), lambda: f"Expected out.shape == {tuple(shape)}, got {out.shape}")
    torch._check(out.dtype == dtype, lambda: f"Expected out.dtype == {dtype}, got {out.dtype}")
    
    result = torchscript_dequantize_4bit(A, absmax, blocksize, quant_type, shape, dtype)
    out.copy_(result)


@torch.jit.script
def dequant_fp4_torchscript(
    quantized: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    shape: List[int],
    dtype: torch.dtype,
    codebook: torch.Tensor,
) -> torch.Tensor:
    return dequant_nf4_torchscript(quantized, absmax, blocksize, shape, dtype, codebook)


@register_kernel("bitsandbytes::dequantize_4bit", "mps")
def torchscript_dequantize_4bit_mps(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    # Re-use the CPU implementation – data is already on the correct (MPS) device
    return torchscript_dequantize_4bit(A, absmax, blocksize, quant_type, shape, dtype)


@register_kernel("bitsandbytes::dequantize_4bit.out", "mps")
def torchscript_dequantize_4bit_out_mps(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
    out: torch.Tensor,
) -> None:
    torch._check(out.shape == tuple(shape), lambda: f"Expected out.shape == {tuple(shape)}, got {out.shape}")
    torch._check(out.dtype == dtype, lambda: f"Expected out.dtype == {dtype}, got {out.dtype}")

    result = torchscript_dequantize_4bit_mps(A, absmax, blocksize, quant_type, shape, dtype)
    out.copy_(result)


# -----------------------------------------------------------------------------
# 4-bit Quantization (FP4 / NF4) – TorchScript CPU/MPS implementation
# -----------------------------------------------------------------------------

# We register this under the *default* dispatch key (empty string) so it is the
# fallback implementation for any non-CUDA device (CPU, MPS, etc.).


@register_kernel("bitsandbytes::quantize_4bit", "")
def torchscript_quantize_4bit(
    A: torch.Tensor,
    blocksize: int,
    quant_type: str,
    quant_storage: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorised 4-bit block quantiser usable from pure Python / TorchScript."""

    torch._check_is_size(blocksize)
    torch._check(quant_type in ("nf4", "fp4"), lambda: f"quant_type must be nf4 or fp4, got {quant_type}")
    torch._check(
        A.dtype in [torch.bfloat16, torch.float16, torch.float32],
        lambda: f"Blockwise 4bit quantization only supports 16/32-bit floats, but got {A.dtype}",
    )

    n = A.numel()
    full_blocks = n // blocksize
    rem = n % blocksize
    blocks = full_blocks + (1 if rem else 0)

    absmax = torch.empty(blocks, device=A.device, dtype=torch.float32)

    A_flat = A.reshape(n)

    # Compute absmax per full block
    if full_blocks:
        A_full = A_flat[: n - rem].reshape(full_blocks, blocksize)
        absmax[:full_blocks] = torch.abs(A_full).amax(dim=1)
    # Remainder block
    if rem:
        absmax[-1] = torch.abs(A_flat[-rem:]).amax()

    # Scale to [-1, 1]
    scale_full = absmax[:full_blocks].view(-1, 1)
    scaled_full = torch.clamp((A_full / scale_full), -1, 1) if full_blocks else torch.empty(0, device=A.device)

    if rem:
        scaled_rem = torch.clamp((A_flat[-rem:] / absmax[-1]), -1, 1)
        scaled = torch.cat([scaled_full.reshape(-1), scaled_rem], dim=0)
    else:
        scaled = scaled_full.reshape(-1)

    # Codebook
    codebook = (NF4_CODEBOOK if quant_type == "nf4" else FP4_CODEBOOK).to(scaled.dtype).to(A.device)

    # Vectorised nearest-neighbour quantisation
    idx = torch.argmin(torch.abs(scaled.unsqueeze(1) - codebook), dim=1).to(torch.uint8)

    # Pack two indices per byte
    packed = idx[::2] << 4
    if idx.numel() % 2 == 1:
        # pad with zero for odd length
        idx = torch.cat([idx, torch.zeros(1, dtype=idx.dtype, device=idx.device)])
    packed |= idx[1::2]

    packed = packed.to(quant_storage)

    return packed, absmax.float()


# ---------------------------------------------------------------------------
# Compatibility helpers – some torch builds (e.g. macOS CPU-only wheels) do
# not expose the internal `_check` / `_check_is_size` APIs used in the backend
# reference implementation.  Provide light-weight shims so the scripted code
# runs everywhere.
# ---------------------------------------------------------------------------

if not hasattr(torch, "_check_is_size"):
    def _check_is_size(val):
        if not (isinstance(val, int) and val >= 0):
            raise ValueError("Expected a non-negative int for blocksize, got " + str(val))

    torch._check_is_size = _check_is_size  # type: ignore[attr-defined]

if not hasattr(torch, "_check"):
    def _check(cond, msg_fn=lambda: ""):
        if not cond:
            raise AssertionError(msg_fn() if callable(msg_fn) else msg_fn)

    torch._check = _check  # type: ignore[attr-defined]


# ------------------------------------------------------------------
# Ensure legacy operator schemas exist *before* we register fallback kernels.
# ------------------------------------------------------------------

# Make sure the schemas exist (idempotent safe-define)
def _safe_define(op, schema):
    try:
        torch.library.define(op, schema)
    except RuntimeError as err:
        if "already" not in str(err):
            raise

_safe_define(
    "bitsandbytes::dequantize_4bit",
    "(Tensor A, Tensor absmax, int blocksize, str quant_type, int[] shape, ScalarType dtype) -> Tensor",
)

_safe_define(
    "bitsandbytes::quantize_blockwise",
    "(Tensor A, Tensor code, int blocksize) -> (Tensor, Tensor)",
)


@register_kernel("bitsandbytes::dequantize_4bit", "")
def torchscript_dequantize_4bit_default(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
):
    return torchscript_dequantize_4bit(A, absmax, blocksize, quant_type, shape, dtype)


@register_kernel("bitsandbytes::quantize_blockwise", "")
def torchscript_quantize_blockwise_default(A: torch.Tensor, code: torch.Tensor, blocksize: int):
    # Naive Python fallback mirroring backend/default implementation
    torch._check_is_size(blocksize)
    n = A.numel()
    rem = n % blocksize
    blocks = n // blocksize + (1 if rem else 0)
    absmax = torch.empty(blocks, dtype=torch.float32, device=A.device)

    A_flat = A.reshape(n).float()

    for i in range(blocks):
        start = i * blocksize
        end = min(start + blocksize, n)
        block = A_flat[start:end]
        absmax[i] = torch.max(block.abs())
        scale = 1.0 / (absmax[i] + 1e-8)
        A_flat[start:end] = torch.clamp(block * scale, -1.0, 1.0)

    # Quantize by lookup
    diff = (A_flat.unsqueeze(-1) - code.to(A.device)) .abs()
    q = torch.argmin(diff, dim=-1).to(torch.uint8)
    return q.view(A.shape), absmax


# ---------------------------------------------------------------------------
# Register direct implementations bypassing the helper wrapper to avoid any
# edge-case with the normalisation shim above.
# ---------------------------------------------------------------------------

_lib_impl = torch.library.Library("bitsandbytes", "IMPL")

def _impl_dequantize_4bit_default(A, absmax, blocksize: int, quant_type: str, shape: Sequence[int], dtype):
    return torchscript_dequantize_4bit_default(A, absmax, blocksize, quant_type, shape, dtype)


def _impl_quantize_blockwise_default(A, code, blocksize: int):
    return torchscript_quantize_blockwise_default(A, code, blocksize)


# Register the implementations (idempotent)
try:
    _lib_impl.impl("dequantize_4bit", _impl_dequantize_4bit_default, dispatch_key="")
except RuntimeError:
    pass

try:
    _lib_impl.impl("quantize_blockwise", _impl_quantize_blockwise_default, dispatch_key="")
except RuntimeError:
    pass 