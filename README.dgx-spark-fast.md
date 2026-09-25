# PersonaPlex DGX Spark fast candidate

This branch adds the community GB10 real-time serving changes to the existing Spark container foundation. It is a candidate until live audio has been tested on the actual DGX Spark.

## Provenance

- Base: `origin/dgx-spark` at `0e4200a2edc8befcf1abe86d5eb4ad75ebd90cc5`.
- Community merge: `jethac/personaplex` `pr-gb10-realtime` at `b67851c0d284d583f8110375978e7910d10b5710`. This includes PR #102 build fixes, live server flags from `9b4470e5bb9c956b383e6b7ad050ffb49d7356f2`, and the optional `other_mimi` reset fix.
- Disconnect fix: `10022f1621f5fc9a88c0aa8087359a283ad4dd7c` from PR #104, applied separately with `cherry-pick -x`.
- The original `Dockerfile.blackwell`, original workflow, `dgx-spark` branch, and original GHCR package remain separate.

## Image environment

`Dockerfile.blackwell-fast` targets `linux/arm64` and uses `nvidia/cuda:13.0.1-devel-ubuntu24.04` at OCI index digest `sha256:7d2f6a8c2071d911524f95061a0db363e24d27aa51ec831fcccf9e76eb72bc92`. The `devel` variant supplies `ptxas` inside the container for Triton/`torch.compile`, without changing the host. Python is 3.12. `constraints.dgx-spark-fast.txt` pins the key packages, including `torch==2.13.0+cu130` and `triton==3.7.1`. The `sphn` ARM64/Python 3.12 wheel is downloaded from the community release and checked against SHA256 `8bc7127ee60485698f3cff1a58c3185c2fd4418916b1db60dc0e8ecbc5475144`. Moshi is installed from this checkout, not from the old community Moshi wheel. The build checks `pip check`, the import path, `--fast` in server help, and presence of `ptxas`.

Model weights and `HF_TOKEN` are never needed during build. The first server start downloads weights from Hugging Face into a runtime volume; accept the model license before starting. Keep secrets out of Git and image. This Dockerfile builds an ARM64 image; neither build nor static checks establish audio performance.

## Build and pull

Pushing `dgx-spark-fast` runs `.github/workflows/build-blackwell-fast.yml` and publishes only:

```text
ghcr.io/sefara/personaplex-nvidia-dgx-spark:sha-<full-commit-SHA>
ghcr.io/sefara/personaplex-nvidia-dgx-spark:fast-candidate
```

For a manual ARM64 build on a suitable machine:

```bash
docker buildx build --platform linux/arm64 -f Dockerfile.blackwell-fast -t personaplex-dgx-spark-fast:local .
```

Pull the immutable SHA tag from the successful workflow and verify `docker buildx imagetools inspect <image>` reports `linux/arm64`. Record the published digest from the workflow before deployment. The `fast-candidate` tag can move when the branch is pushed again.

## Portainer and Docker Compose

Paste `compose.dgx-spark-fast.yaml` into a **new** Portainer stack or use it with Docker Compose. Set `HF_TOKEN` in the Portainer stack environment variables. Optionally set `PERSONAPLEX_FAST_IMAGE` to the SHA tag or digest from the successful build and `PERSONAPLEX_FAST_PORT` (default `8999`). No local `.env` file is required by Portainer. `.env.dgx-spark-fast.example` contains placeholders only; do not commit a filled-in copy.

```bash
export HF_TOKEN='<your Hugging Face token>'
export PERSONAPLEX_FAST_IMAGE='ghcr.io/sefara/personaplex-nvidia-dgx-spark:sha-<full-commit-SHA>'
docker compose -f compose.dgx-spark-fast.yaml up -d
```

The stack requests one NVIDIA GPU and keeps Hugging Face models, TLS certificates, Triton cache and TorchInductor cache in separate named volumes. It does not reuse the old compiled kernel cache. To explicitly reuse an existing HF cache, change only the `personaplex-fast-models:/data/huggingface` mount after confirming the actual old cache path. The first start may take time to download weights and compile kernels; the healthcheck allows 20 minutes for warmup.

The entrypoint creates a persistent self-signed localhost certificate when none is mounted. For remote browser access, install a certificate trusted by that browser with SAN matching the DGX hostname/IP as `cert.pem` and `key.pem` in the SSL volume. Browser microphone access requires a secure context. The healthcheck and `dgx-spark-smoke.sh` establish only an HTTPS response, not a working audio conversation.

Do not start two model servers simultaneously until GPU/RAM capacity is known. The new default host port is 8999 so it does not replace the existing stack, but both still need substantial memory.

## Diagnose and compare

Run inside the new container on the Spark:

```bash
docker exec <new-container-name> python /app/scripts/dgx-spark-gpu-check.py
docker logs <new-container-name> --tail 200
```

The GPU check prints versions, `ptxas`, GPU name/capability, executes `torch.compile`, and compares a W8A16 Triton calculation with a dequantized reference. Without CUDA it reports **SKIP**, not PASS. `NO_TORCH_COMPILE` must be absent; even the string `0` disables the upstream helper. The fast entrypoint and server reject that setting.

For a baseline with the **same new image** but no `--fast`, add this `command` to the new Compose service and redeploy that test stack:

```yaml
command: ["python", "-m", "moshi.server", "--host", "0.0.0.0", "--port", "8998", "--ssl", "/app/ssl"]
```

Remove the override to restore `--fast`. Never combine `--fast` and `--fp8`; `--fast` activates W8A16, depformer early exit at 8, skipping unused `other_mimi`, and FP16/compiled Mimi.

For a live acceptance test, warm the model, then have a 30-minute conversation through the actual web client. Include interrupted speech, mid-stream browser disconnect, a fresh connection and a restart of the **new** container. Compare stock and fast sequentially with the same prompts and similar load. Note audible stutter and any growing latency. The server logs timing for **one in every 50 frames**. To summarize those samples:

```bash
docker logs <new-container-name> 2>&1 | python3 scripts/dgx-spark-frame-stats.py
```

The report gives p50/p95/p99/max and sample counts over 80 ms. It cannot assert that unlogged frames met the budget. Community benchmark numbers are not measurements from this Spark.

## Return to the original image

Keep the existing Portainer stack and original image ID/digest recorded on the DGX. If the candidate misbehaves, stop **only the new test stack** and run the original stack again using its preserved image/configuration. Do not rebuild the old Dockerfile as a substitute for the exact original image because its dependencies were unpinned. No existing service, volume, driver or host CUDA installation is changed by this branch.
