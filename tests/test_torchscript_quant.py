import pytest
import torch
import bitsandbytes.functional as F

# Helper to pick the right quantization/dequantization functions
_quant_fns = {
    "nf4": F.quantize_nf4,
    "fp4": F.quantize_fp4,
}

@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
def test_torchscript_dequant_matches_default(quant_type):
    """Ensure TorchScript CPU kernel produces identical results to the reference Python implementation."""
    q_fn = _quant_fns[quant_type]

    torch.manual_seed(0)
    data = torch.randn(128, 256, dtype=torch.float16)

    quant, state = q_fn(data, blocksize=64)

    # Reference result (default backend)
    ref = torch.ops.bitsandbytes.dequantize_4bit.default(
        quant,
        state.absmax,
        state.blocksize,
        state.quant_type,
        state.shape,
        data.dtype,
    )

    # TorchScript CPU kernel result
    ts_cpu = torch.ops.bitsandbytes.dequantize_4bit.cpu(
        quant,
        state.absmax,
        state.blocksize,
        state.quant_type,
        state.shape,
        data.dtype,
    )

    assert torch.allclose(ref, ts_cpu, atol=1e-6), "TorchScript CPU result differs from reference"

    # If running on Apple Silicon with MPS available, test MPS kernel as well
    if torch.backends.mps.is_available():
        quant_mps = quant.to("mps")
        absmax_mps = state.absmax.to("mps")

        ts_mps = torch.ops.bitsandbytes.dequantize_4bit.mps(
            quant_mps,
            absmax_mps,
            state.blocksize,
            state.quant_type,
            state.shape,
            data.dtype,
        ).cpu()

        assert torch.allclose(ref, ts_mps, atol=1e-6), "TorchScript MPS result differs from reference" 