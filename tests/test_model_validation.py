import os
import pytest
import torch

import bitsandbytes as bnb
from tests.helpers import get_available_devices, id_formatter, describe_dtype

# -----------------------------------------------------------------------------
#  NOTE: This test is purposely lightweight so it can run inside the regular
#  CPU-only GitHub Actions runners.  We quantise a *single* Linear layer weight
#  and execute three inference passes.  The CPU/MPS result is compared against
#  the CUDA result (if available).  If CUDA is not available the test is
#  skipped – the full GPU check runs on dedicated CUDA runners.
# -----------------------------------------------------------------------------

# Environment knob to skip on very bandwidth-limited CI shards.
if os.getenv("BNB_SKIP_SLOW", "0") == "1":
    pytest.skip("Model-level validation tests are disabled via BNB_SKIP_SLOW", allow_module_level=True)


@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.bfloat16], ids=describe_dtype)
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"], ids=id_formatter("qt"))
def test_linear4bit_cpu_vs_gpu_equivalence(quant_type: str, compute_dtype: torch.dtype):
    """Compare CPU / MPS fallback logits with CUDA baseline for Linear4bit.

    The test is deliberately tiny – 128×64 weight – to keep runtime & memory
    footprint negligible while still exercising the full 4-bit dequant path.
    """

    # ------------------------------------------------------------------
    # Skip combinations that the platform cannot execute.
    # ------------------------------------------------------------------
    if compute_dtype == torch.bfloat16 and not torch.tensor([], dtype=torch.bfloat16).is_floating_point():
        pytest.skip("bfloat16 not supported on this platform")

    has_cuda = torch.cuda.is_available()
    if not has_cuda:
        pytest.skip("CUDA baseline unavailable on this runner")

    # ------------------------------------------------------------------
    # 1. Build a reference FP16 Linear layer on CUDA and quantise weights.
    # ------------------------------------------------------------------
    in_features, out_features = 128, 64

    ref_linear = torch.nn.Linear(in_features, out_features, bias=False, dtype=compute_dtype, device="cuda")
    with torch.no_grad():
        torch.nn.init.uniform_(ref_linear.weight, a=-1.0, b=1.0)

    # Quantise to 4-bit using bitsandbytes Params4bit helper.
    weight_qcpu = bnb.nn.Params4bit(data=ref_linear.weight.to("cpu"), requires_grad=False, quant_type=quant_type)

    # ------------------------------------------------------------------
    # 2. Instantiate two Linear4bit layers: one on CPU/MPS, one on CUDA.
    # ------------------------------------------------------------------
    lin_cpu = bnb.nn.Linear4bit(
        in_features,
        out_features,
        bias=False,
        compute_dtype=compute_dtype,
        quant_type=quant_type,
        device="cpu",
    )
    lin_cpu.weight = weight_qcpu.to("cpu")

    lin_gpu = bnb.nn.Linear4bit(
        in_features,
        out_features,
        bias=False,
        compute_dtype=compute_dtype,
        quant_type=quant_type,
        device="cuda",
    )
    lin_gpu.weight = weight_qcpu.to("cuda")

    # ------------------------------------------------------------------
    # 3. Run deterministic forward passes multiple times (default 10).
    #    The repetition count can be overridden via the BNB_VALID_ITERS
    #    environment variable.
    # ------------------------------------------------------------------

    num_iters = int(os.getenv("BNB_VALID_ITERS", "10"))

    torch.manual_seed(0)
    for _ in range(num_iters):
        x = torch.randn(4, in_features, dtype=compute_dtype, device="cuda")
        cpu_logits = lin_cpu(x.to("cpu")).to("cuda")
        gpu_logits = lin_gpu(x)
        torch.testing.assert_close(cpu_logits, gpu_logits, atol=1e-3, rtol=1e-3) 