"""Prepare persistent TLS and pass signals directly to the server."""

import os
import pathlib
import subprocess
import sys


def main() -> None:
    command = sys.argv[1:]
    if not command:
        raise SystemExit("No server command supplied")
    if "--fast" in command and os.environ.get("NO_TORCH_COMPILE"):
        raise SystemExit("--fast requires NO_TORCH_COMPILE to be unset")

    ssl_dir = pathlib.Path("/app/ssl")
    ssl_dir.mkdir(parents=True, exist_ok=True)
    cert = ssl_dir / "cert.pem"
    key = ssl_dir / "key.pem"
    if not cert.exists() or not key.exists():
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                "-days", "365", "-keyout", str(key), "-out", str(cert),
                "-subj", "/CN=localhost",
                "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        key.chmod(0o600)
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
