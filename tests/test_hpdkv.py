import math

import torch

from bitsandbytes.formats import estimate_hpdkv_size, pack_residuals, unpack_residuals


def test_dense_roundtrip():
    x = torch.randn(257, 129, dtype=torch.float32)
    blob = pack_residuals(x)
    x_hat = unpack_residuals(blob)
    # Stored as fp16; tolerance accordingly
    assert torch.allclose(x.to(torch.float16), x_hat, atol=1e-3)


def test_topk_roundtrip():
    x = torch.randn(1000, dtype=torch.float32)
    k_frac = 0.1
    blob = pack_residuals(x, store_topk=k_frac)
    x_hat = unpack_residuals(blob)

    nonzero_mask = x_hat != 0
    # Expected number of retained elements (within 1 due to ceil)
    expected_k = math.ceil(k_frac * x.numel())
    assert nonzero_mask.sum().item() == expected_k

    # Verify values match fp16 cast for retained indices
    assert torch.allclose(x.to(torch.float16)[nonzero_mask], x_hat[nonzero_mask], atol=1e-3)


def test_compressed_roundtrip():
    x = torch.randn(512, 512, dtype=torch.float32)
    blob = pack_residuals(x, compress=True)
    x_hat = unpack_residuals(blob)
    assert torch.allclose(x.to(torch.float16), x_hat, atol=1e-3)


def test_size_estimation():
    x = torch.randn(64, 64)
    est = estimate_hpdkv_size(x)
    blob = pack_residuals(x)
    # Estimation must not underestimate actual size
    assert est >= len(blob) 