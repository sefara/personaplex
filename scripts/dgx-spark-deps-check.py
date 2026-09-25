"""Accept one known cusparseLt wheel tag warning only after ELF verification."""

import importlib.metadata
import pathlib
import platform
import subprocess
import sys


result = subprocess.run(
    [sys.executable, "-m", "pip", "check"], capture_output=True, text=True, check=False
)
if result.returncode == 0:
    print(result.stdout.strip())
    raise SystemExit(0)

warning = "nvidia-cusparselt-cu13 0.8.1 is not supported on this platform"
if result.stdout.strip() != warning or result.stderr.strip():
    sys.stderr.write(result.stdout + result.stderr)
    raise SystemExit("pip check found an unexpected dependency error")
if platform.machine().lower() not in {"aarch64", "arm64"}:
    raise SystemExit("cusparseLt wheel exception requires an ARM64 container")

distribution = importlib.metadata.distribution("nvidia-cusparselt-cu13")
libraries = [path for path in distribution.files or []
             if "cusparselt" in str(path).lower() and ".so" in path.name]
if not libraries:
    raise SystemExit("cusparseLt wheel contains no shared library")
for library in libraries:
    path = pathlib.Path(distribution.locate_file(library))
    with path.open("rb") as handle:
        header = handle.read(20)
    if header[:4] != b"\x7fELF" or int.from_bytes(header[18:20], "little") != 183:
        raise SystemExit(f"cusparseLt library is not AArch64 ELF: {path}")
print(f"pip check: only known cusparseLt tag warning; verified {len(libraries)} AArch64 ELF libraries")
