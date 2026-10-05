#!/usr/bin/env bash
# Managed installation: no site Python, llama.cpp, compiler, or CUDA module required.
set -euo pipefail
umask 077
APP_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
profile=
backend=cuda12
cache_dir=
wheelhouse=
offline=false
while (($#)); do
  case "$1" in
    --profile) profile="${2:?Supply a profile name or JSON path}"; shift 2 ;;
    --backend) backend="${2:?Supply cuda12 or cpu}"; shift 2 ;;
    --cache-dir) cache_dir="${2:?Supply the model cache directory}"; shift 2 ;;
    --wheelhouse) wheelhouse="${2:?Supply a wheel directory}"; shift 2 ;;
    --offline) offline=true; shift ;;
    --help)
      echo 'Usage: bash install.sh [--profile NAME|JSON_PATH] [--backend cuda12|cpu] [--cache-dir MODEL_PATH] [--wheelhouse PATH] [--offline]'
      echo 'Installs managed Python, application packages, llama.cpp and its user-space libraries inside this folder.'
      echo 'Default backend: cuda12. Requires Linux x86_64, curl, tar and sha256sum; glibc >=2.28 for CUDA or >=2.17 for CPU.'
      echo 'NVIDIA kernel driver and Slurm clients belong to the host. No system Python or CUDA module is used.'
      echo '--offline requires previously cached bootstrap/Python/runtime packages plus wheels/cache.'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$backend" in cuda12|cpu) ;; *) echo 'Backend must be cuda12 or cpu.' >&2; exit 2;; esac
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo 'This pinned managed installer currently supports Linux x86_64 only.' >&2; exit 1
fi
for utility in curl tar sha256sum; do
  command -v "$utility" >/dev/null || { echo "Required bootstrap utility missing: $utility" >&2; exit 1; }
done
export UV_CACHE_DIR="$APP_DIR/cache/uv"
export UV_PYTHON_INSTALL_DIR="$APP_DIR/cache/python"
unset UV_PYTHON_PREFERENCE
export UV_MANAGED_PYTHON=true
export UV_NO_CONFIG=1
export UV_HTTP_TIMEOUT=60
export UV_HTTP_RETRIES=2
export PIP_CACHE_DIR="$APP_DIR/cache/pip"
export PIP_DISABLE_PIP_VERSION_CHECK=1
if $offline; then export UV_OFFLINE=true; fi
mkdir -p "$APP_DIR/cache/downloads" "$APP_DIR/cache/bootstrap"
uv_archive="$APP_DIR/cache/downloads/uv-x86_64-unknown-linux-gnu-0.8.22.tar.gz"
uv_sha=741ff1f5742c5a4a25d2f829e8395355e43f7a5ae2ebc6368e9ae2df0efb69cf
if [[ -L "$uv_archive" ]]; then echo 'Refusing bootstrap archive symlink.' >&2; exit 1; fi
if [[ ! -f "$uv_archive" ]] || ! printf '%s  %s\n' "$uv_sha" "$uv_archive" | sha256sum --check --status; then
  if $offline; then echo 'Offline pinned uv bootstrap archive missing or checksum mismatch.' >&2; exit 1; fi
  temporary="$(mktemp "$APP_DIR/cache/downloads/.uv-XXXXXX")"
  trap 'rm -f -- "${temporary:-}"' EXIT
  echo 'Downloading checksum-pinned uv 0.8.22 bootstrap…'
  curl --fail --location --retry 2 --connect-timeout 30 --max-time 300 \
    --output "$temporary" 'https://github.com/astral-sh/uv/releases/download/0.8.22/uv-x86_64-unknown-linux-gnu.tar.gz'
  printf '%s  %s\n' "$uv_sha" "$temporary" | sha256sum --check --status || { echo 'uv archive checksum mismatch.' >&2; exit 1; }
  mv -- "$temporary" "$uv_archive"
  trap - EXIT
fi
# Extract the one known, verified executable, never arbitrary archive paths.
uv_cmd="$APP_DIR/cache/bootstrap/uv-0.8.22"
if [[ -L "$uv_cmd" ]]; then echo 'Refusing bootstrap executable symlink.' >&2; exit 1; fi
uv_tmp="$(mktemp "$APP_DIR/cache/bootstrap/.uv-XXXXXX")"
trap 'rm -f -- "${uv_tmp:-}"' EXIT
tar -xOf "$uv_archive" uv-x86_64-unknown-linux-gnu/uv > "$uv_tmp"
chmod 700 "$uv_tmp"
mv -- "$uv_tmp" "$uv_cmd"
trap - EXIT
echo 'Installing managed Python 3.11.13 inside this folder…'
"$uv_cmd" python install --no-bin 3.11.13
python_cmd="$("$uv_cmd" python find --managed-python 3.11.13)"
args=(--app "$APP_DIR" --uv "$uv_cmd" --python "$python_cmd" --backend "$backend")
if [[ -n "$profile" ]]; then args+=(--profile "$profile"); fi
if [[ -n "$cache_dir" ]]; then args+=(--cache-dir "$cache_dir"); fi
if [[ -n "$wheelhouse" ]]; then args+=(--wheelhouse "$wheelhouse"); fi
if $offline; then args+=(--offline); fi
exec "$python_cmd" "$APP_DIR/src/hpc_llm/installation.py" "${args[@]}"
