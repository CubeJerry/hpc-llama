#!/usr/bin/env bash
# Download and register the exact Q4_K_P release plus both companion GGUFs.
set -euo pipefail
umask 077
APP_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
for arg in "$@"; do
  case "$arg" in
    --dry-run|--verify-full) ;;
    *) echo "Usage: bash scripts/install-qwen38-q4kp.sh [--dry-run] [--verify-full]" >&2; exit 2 ;;
  esac
done

cat <<'INFO'
Installing Qwen3.8 27B Q4_K_P, BF16 vision projector, and FastMTP sidecar.
Total model download: about 19.76 GB (18.40 GiB), before existing cached files.
The main model already includes embedded MTP. The separate FastMTP sidecar
requires the publisher's patched llama.cpp runtime; installing it alone does
not patch your runtime. This launcher's Auto mode uses the embedded MTP head
on the bundled runtime; the FastMTP sidecar stays registered for a compatible
patched runtime. Embedded mode explicitly selects the built-in head.
INFO

"$APP_DIR/hpc-llm" models install \
  HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF \
  --revision 993a5971fda8f30dd1b7eb2654792ba4415c7460 \
  --quant Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf \
  --projector mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf \
  --mtp Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf \
  "$@"

cat <<'INFO'
Refresh Models in the browser to select the registered model after installation.
FastMTP runtime instructions and the verified source patch are published at:
https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF#run-hauhaucs-fastmtp
INFO
