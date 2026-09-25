"""Run inside the candidate container on a DGX Spark, without model weights."""

import importlib.metadata
import pathlib
import shutil
import subprocess
import sys

import torch


def main() -> int:
    import moshi

    print(f"moshi source: {pathlib.Path(moshi.__file__).resolve()}")
    for package in ("torch", "triton", "sphn", "moshi-personaplex"):
        print(f"{package}: {importlib.metadata.version(package)}")
    print(f"torch CUDA: {torch.version.cuda}")
    ptxas = shutil.which("ptxas")
    if not ptxas:
        raise RuntimeError("ptxas is missing inside the container")
    print(subprocess.check_output([ptxas, "--version"], text=True).strip())
    if not torch.cuda.is_available():
        print("SKIP: no CUDA GPU; compile and W8A16 runtime remain untested")
        return 0

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"compute capability: {torch.cuda.get_device_capability(0)}")
    x = torch.randn(128, device="cuda")
    compiled = torch.compile(lambda value: value.square() + 1)
    torch.testing.assert_close(compiled(x), x.square() + 1)
    print("torch.compile: PASS")

    from moshi.w8a16_quantize import w8a16_linear

    activation = torch.randn(1, 256, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, 256, device="cuda", dtype=torch.float32)
    scale = weight.abs().amax(dim=1).clamp_min(1e-6) / 448.0
    weight_fp8 = (weight / scale[:, None]).to(torch.float8_e4m3fn)
    actual = w8a16_linear(activation, weight_fp8, scale)
    reference = torch.nn.functional.linear(
        activation.float(), weight_fp8.float() * scale[:, None]
    )
    torch.testing.assert_close(actual.float(), reference, rtol=0.02, atol=0.03)
    print("W8A16 Triton kernel vs dequantized reference: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
