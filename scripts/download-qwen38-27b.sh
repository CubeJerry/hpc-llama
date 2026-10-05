#!/usr/bin/env bash
# Download only the requested quant and matching F16 vision projector.
set -euo pipefail
app_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${1:-$app_dir/models/Qwen3.8-27B}"
mkdir -p -- "$model_dir"
model_dir="$(CDPATH= cd -- "$model_dir" && pwd)"
base='https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/4ca720788d1e01f1bff70c033e0d0028fd02e502'
fetch() {
    local name="$1" digest="$2" target="$model_dir/$1"
    if [[ -L "$target" || -L "$target.part" ]]; then
        echo "Refusing symbolic link: $target" >&2; return 1
    fi
    if [[ -f "$target" ]]; then
        printf '%s  %s\n' "$digest" "$target" | sha256sum --check -
        return
    fi
    curl --fail --location --retry 3 --connect-timeout 30 --continue-at - \
        --output "$target.part" "$base/$name"
    printf '%s  %s\n' "$digest" "$target.part" | sha256sum --check -
    mv -- "$target.part" "$target"
}
fetch Qwen3.8-27B-UD-Q4_K_XL.gguf 3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e
fetch mmproj-F16.gguf cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e
"$app_dir/hpc-llm" models register "$model_dir/Qwen3.8-27B-UD-Q4_K_XL.gguf" \
    --projector "$model_dir/mmproj-F16.gguf"
echo 'Model and vision projector registered. Run ./hpc-llm and start a new session.'
