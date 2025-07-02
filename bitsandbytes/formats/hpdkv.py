"""HPΔKV (High-Precision Delta Key/Value) residual serialization utilities.

This module defines a compact, self-describing binary format for storing
high-precision residuals corresponding to quantised weights.  The format is
intended for efficient disk/network storage while permitting loss-free
reconstruction — or controlled sparsification via *top-k* retention — when
needed.

Schema (little-endian, 16-byte aligned)
--------------------------------------
Offset  Size    Field       Description
0       4       magic       ASCII "HPDV"
4       1       version     uint8 (currently 1)
5       1       dtype       uint8 code: 0 = FP16, 1 = BF16
6       1       rank        uint8 (number of tensor dimensions)
7       1       flags       bit-mask: bit0 = indices present, bit1 = compressed
8       4×rank  shape       uint32 for each dimension, little-endian
8+4r    *       pad         zero-bytes so that payload starts at 16-byte boundary

If *indices present* flag is set, the header is followed by a variable-length
block:
    uint32  k           — number of retained indices
    uint32[k] indices   — flat indices into the original tensor

The residual payload comes last.  When *indices present* is true, the payload
contains exactly *k* fp16/bf16 values; otherwise, it stores the full dense
tensor.  If the *compressed* flag is set, the payload is zlib-compressed.
"""

from __future__ import annotations

import math
import struct
import zlib
from typing import List

import numpy as np
import torch

__all__ = [
    "pack_residuals",
    "unpack_residuals",
    "estimate_hpdkv_size",
]

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------
_MAGIC = b"HPDV"
_VERSION = 1

# dtype codes
_DTYPE_FP16 = 0
_DTYPE_BF16 = 1

# flag bits
_FLAG_INDICES = 1 << 0
_FLAG_COMPRESSED = 1 << 1

_TORCH_DTYPE_TO_CODE = {
    torch.float16: _DTYPE_FP16,
    torch.bfloat16: _DTYPE_BF16,
}
_CODE_TO_TORCH_DTYPE = {v: k for k, v in _TORCH_DTYPE_TO_CODE.items()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pack_residuals(
    tensor: torch.Tensor,
    *,
    store_topk: float | None = None,
    compress: bool = False,
) -> bytes:
    """Serialise *tensor* into HPΔKV binary blob.

    Parameters
    ----------
    tensor : torch.Tensor
        Residual values.  Will be cast to *float16* during storage.
    store_topk : float | None, optional
        If given, retain only the largest |residual| values corresponding to
        the top-`store_topk` fraction (0 < *store_topk* ≤ 1).  When the value is
        1 or *None*, the full dense tensor is stored and **indices** flag is
        cleared.
    compress : bool, default=False
        If *True*, zlib-compress the payload and set the *compressed* flag.

    Returns
    -------
    bytes
        Serialized blob.
    """
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("Only FP16/BF16/FP32 tensors may be packed as residuals")

    # Ensure contiguous CPU float16 array for deterministic byte layout.
    cpu_fp16: torch.Tensor = tensor.detach().to(torch.float16).contiguous().cpu()
    flat = cpu_fp16.flatten()
    total_elems = flat.numel()

    # ---------------------------------------------------------------------
    # Sparsification via top-k retention
    # ---------------------------------------------------------------------
    indices: np.ndarray | None = None
    values_to_store: torch.Tensor
    flags = 0

    if store_topk is not None and 0 < store_topk < 1:
        k = int(math.ceil(store_topk * total_elems))
        if k <= 0:
            raise ValueError("store_topk resulted in zero retained elements")
        # torch.topk is expensive on CPU for huge tensors; move to CUDA if
        # available and beneficial.
        if tensor.is_cuda:
            abs_tensor = tensor.detach().abs()
            _, topk_idx = torch.topk(abs_tensor.flatten(), k)
            values_to_store = flat[topk_idx.cpu()]
        else:
            _, topk_idx = torch.topk(flat.float().abs(), k)
            values_to_store = flat[topk_idx]
        indices = topk_idx.cpu().numpy().astype(np.uint32, copy=False)
        flags |= _FLAG_INDICES
    else:
        values_to_store = flat  # dense case
        k = total_elems

    # ---------------------------------------------------------------------
    # Serialise payload
    # ---------------------------------------------------------------------
    payload: bytes = values_to_store.numpy().tobytes()
    if compress:
        payload = zlib.compress(payload)
        flags |= _FLAG_COMPRESSED

    # ---------------------------------------------------------------------
    # Header (little-endian)
    # ---------------------------------------------------------------------
    rank = tensor.ndim
    if rank > 255:
        raise ValueError("HPΔKV supports tensors with at most 255 dimensions")

    dtype_code = _DTYPE_FP16  # we stored payload as fp16 regardless of input

    header_parts: List[bytes] = [
        _MAGIC,
        struct.pack("<BBBB", _VERSION, dtype_code, rank, flags),
    ]
    for dim in tensor.shape:
        if dim > 0xFFFFFFFF:
            raise ValueError("Dimensions larger than 2^32-1 not supported")
        header_parts.append(struct.pack("<I", int(dim)))

    header = b"".join(header_parts)

    # Pad header to next 16-byte boundary as per spec.
    pad_len = (16 - (len(header) % 16)) % 16
    if pad_len:
        header += b"\x00" * pad_len

    # ---------------------------------------------------------------------
    # Optional indices block
    # ---------------------------------------------------------------------
    index_block = b""
    if indices is not None:
        index_block = struct.pack("<I", indices.size) + indices.tobytes()

    blob = header + index_block + payload
    return blob


def unpack_residuals(blob: bytes | bytearray | memoryview, device=None) -> torch.Tensor:
    """Deserialize HPΔKV *blob* back into a `torch.Tensor`.

    Parameters
    ----------
    blob : bytes-like
        Serialized HPΔKV data.
    device : torch.device | str | None
        Destination device.  If *None*, keep tensor on CPU.

    Returns
    -------
    torch.Tensor
        Reconstructed residual tensor.
    """
    mv = memoryview(blob)

    # ---------------------------------------------------------------------
    # Parse fixed part (first 8 bytes)
    # ---------------------------------------------------------------------
    if mv[:4].tobytes() != _MAGIC:
        raise ValueError("Input does not start with HPΔKV magic header")

    version, dtype_code, rank, flags = struct.unpack("<BBBB", mv[4:8])
    if version != _VERSION:
        raise ValueError(f"Unsupported HPΔKV version {version}")
    if dtype_code not in _CODE_TO_TORCH_DTYPE:
        raise ValueError(f"Unsupported dtype code {dtype_code}")

    # ---------------------------------------------------------------------
    # Shape
    # ---------------------------------------------------------------------
    offset = 8
    shape: List[int] = []
    for _ in range(rank):
        (dim,) = struct.unpack("<I", mv[offset : offset + 4])
        shape.append(dim)
        offset += 4

    # Skip padding to reach 16-byte alignment
    pad_len = (16 - (offset % 16)) % 16
    offset += pad_len

    # ---------------------------------------------------------------------
    # Optional indices
    # ---------------------------------------------------------------------
    indices: np.ndarray | None = None
    if flags & _FLAG_INDICES:
        (k,) = struct.unpack("<I", mv[offset : offset + 4])
        offset += 4
        if k == 0:
            raise ValueError("Index count is zero while indices flag is set")
        indices = np.frombuffer(mv[offset : offset + 4 * k], dtype="<u4", count=k)
        offset += 4 * k

    # ---------------------------------------------------------------------
    # Payload
    # ---------------------------------------------------------------------
    payload_bytes = mv[offset:].tobytes()
    if flags & _FLAG_COMPRESSED:
        payload_bytes = zlib.decompress(payload_bytes)

    expected_dtype = np.float16 if dtype_code == _DTYPE_FP16 else np.float16  # stored as fp16
    if len(payload_bytes) % np.dtype(expected_dtype).itemsize != 0:
        raise ValueError("Payload size is not a multiple of element size")

    values = np.frombuffer(payload_bytes, dtype=expected_dtype)

    total_elems = int(np.prod(shape))

    if indices is not None:
        if values.size != indices.size:
            raise ValueError("Mismatch between indices and payload length")
        dense = np.zeros(total_elems, dtype=expected_dtype)
        dense[indices] = values
    else:
        if values.size != total_elems:
            raise ValueError("Dense payload size mismatch with tensor shape")
        dense = values

    tensor = torch.from_numpy(dense.reshape(shape))
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def estimate_hpdkv_size(
    tensor: torch.Tensor,
    *,
    store_topk: float | None = None,
    compress: bool = False,
) -> int:
    """Return the **estimated** number of bytes the packed tensor will occupy.

    Note that compression ratios are data-dependent; when *compress=True* this
    routine actually compresses a dummy zero payload to provide a pessimistic
    bound without allocating the real dense tensor on CPU.
    """
    rank = tensor.ndim
    header_basic = 8 + 4 * rank
    pad_len = (16 - (header_basic % 16)) % 16
    header_size = header_basic + pad_len

    total_elems = tensor.numel()

    # Determine k for top-k retention.
    if store_topk is not None and 0 < store_topk < 1:
        k = int(math.ceil(store_topk * total_elems))
        indices_bytes = 4 + 4 * k  # uint32 count + indices
        payload_elems = k
    else:
        indices_bytes = 0
        payload_elems = total_elems

    payload_bytes = 2 * payload_elems  # fp16 element size

    if compress:
        # Rough upper bound: zlib can expand data a little; worst-case overhead
        # is 5 bytes per 16 KiB block + 6-byte stream header/footer.
        compressed_estimate = payload_bytes + payload_bytes // (16 * 1024) * 5 + 6
        payload_bytes = compressed_estimate

    return header_size + indices_bytes + payload_bytes


# ---------------------------------------------------------------------------
# Module self-test (executed only when run as script, not on import)
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    x = torch.randn(512, 512, dtype=torch.float32)
    blob = pack_residuals(x, store_topk=0.05, compress=True)
    x_hat = unpack_residuals(blob)
    print("Recovered", torch.allclose(x.to(torch.float16), x_hat))
    print("Size vs fp16", len(blob), x.numel() * 2)
