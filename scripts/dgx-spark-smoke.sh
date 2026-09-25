#!/usr/bin/env bash
set -euo pipefail

url="${1:-https://127.0.0.1:8999/}"
curl --fail --insecure --show-error --silent --max-time 10 "$url" >/dev/null
printf 'HTTPS/Web UI responded: %s\n' "$url"
printf 'This does not verify GPU inference or live audio.\n'
