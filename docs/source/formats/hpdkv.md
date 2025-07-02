# HPΔKV Residual Format

*(High-Precision Delta Key/Value)*

HPΔKV is a compact binary container for storing the **high-precision residuals**
of 4-bit‐quantised model weights.  The format is self-describing, portable, and
simple enough to be implemented in a few dozen lines of code.

---
## Why residuals?
Even aggressive 4-bit schemes such as NF4/FP4 preserve a very large fraction of
model quality, yet the tiny loss in perplexity or downstream accuracy sometimes
matters for evaluation benchmarks.  HPΔKV stores the remaining *delta* between
the original FP16 weights and their 4-bit reconstruction so the model can be
*exactly* recovered on demand or during finetuning.

---
## File Layout
The blob consists of a fixed-size **header** followed by an optional index block
and the **payload**.

| Offset | Size                     | Field      | Description                                      |
|-------:|:------------------------:|------------|--------------------------------------------------|
| 0      | 4 bytes                  | `magic`    | ASCII `"HPDV"`                                   |
| 4      | 1 byte                   | `version`  | `uint8` – currently `1`                          |
| 5      | 1 byte                   | `dtype`    | `0 = FP16`, `1 = BF16`                           |
| 6      | 1 byte                   | `rank`     | Number of tensor dimensions (≤255)               |
| 7      | 1 byte                   | `flags`    | Bit-mask (see below)                             |
| 8      | `4 × rank` bytes         | `shape`    | `uint32` (little-endian) for each dimension      |
| …      | 0–15 bytes               | *padding*  | Zero-bytes so payload starts at 16-byte boundary |

### Flags
* **bit 0** – *indices present* → a sparse *index block* precedes the payload.  
  The block layout is `uint32 k` followed by `k×uint32` flat indices.
* **bit 1** – *compressed payload* → payload is zlib-compressed.

### Payload
If **indices present**: exactly `k` FP16/BF16 values (one per stored index).  
Otherwise: a dense tensor with `∏shape` elements.

---
## Reference Python API
```python
from bitsandbytes.formats import (
    pack_residuals,       # Tensor → bytes
    unpack_residuals,     # bytes  → Tensor
    estimate_hpdkv_size,  # quick size check
)
```

Example usage:
```python
import torch, bitsandbytes as bnb

weights = torch.randn(1024, 1024, dtype=torch.float16, device="cpu")
# Keep only the top-5 % largest residuals and compress
blob = bnb.pack_residuals(weights, store_topk=0.05, compress=True)
print("Saved", len(blob)/ (weights.numel()*2), "× the FP16 size")

recovered = bnb.unpack_residuals(blob)
assert torch.allclose(weights, recovered, atol=1e-3)
```

---
## Compatibility Guarantees
*  Little-endian only – simplifies cross-platform support.
*  Future versions will increment the **version** byte; readers *must* error on
a version they do not understand.
*  Index data always uses 32-bit offsets; tensors larger than 4 Gi elements
should be sharded over multiple HPΔKV blobs.

---
## Implementation Notes
1. **Top-k selection** relies on `torch.topk` operating on `float32` to avoid the
   lack of `half` support on CPU.
2. When `compress=True`, HPΔKV calls `zlib.compress` – alternative codecs
   (LZ4, zstd) can be negotiated via extra flag bits in future versions.
3. The writer pads the header so the payload starts at a 16-byte boundary,
   making direct `mmap` access friendly for SIMD loads.

---
## License
HPΔKV is released under the same MIT license as Bits and Bytes.  You are
encouraged to adopt and extend the format in your own tooling. 