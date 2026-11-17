#!/usr/bin/env bash
# Compatibility entry point: TUI selection, then batch-owned native browser chat.
set -euo pipefail
APP_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$APP_DIR/hpc-llm" launch "$@"
