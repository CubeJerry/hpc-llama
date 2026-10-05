#!/usr/bin/env python3
"""Create a source release with per-file checksums, without changing the source."""
import argparse
import hashlib
from pathlib import Path
import zipfile


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=root.parent / "hpc-llm-release.zip")
    args = parser.parse_args()
    files = [root / name for name in ("README.md", "LICENSE", ".gitignore", "pyproject.toml", "install.sh", "hpc-llm")]
    for directory in ("src/hpc_llm", "scripts", "profiles", "assets"):
        for path in (root / directory).rglob("*"):
            if path.is_symlink():
                raise SystemExit(f"Refusing source symlink: {path}")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files.append(path)
    files = sorted(set(files))
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"Missing or unsafe release file: {path}")
        if path.suffix in {".key", ".gguf", ".safetensors"}:
            raise SystemExit(f"Private data or model weights in source set: {path}")
    checksums = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}\n"
        for path in files
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            info = zipfile.ZipInfo(f"hpc-llm/{path.relative_to(root)}", date_time=(2026, 10, 4, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o100755 if path.name in {"hpc-llm", "install.sh"} or path.suffix == ".sh" else 0o100644
            info.external_attr = mode << 16
            archive.writestr(info, path.read_bytes())
        info = zipfile.ZipInfo("hpc-llm/SOURCE_CHECKSUMS.sha256", date_time=(2026, 10, 4, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, checksums)
    checksum = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".zip.sha256").write_text(f"{checksum}  {args.output.name}\n")
    print(f"{args.output.name}: {len(files) + 1} files, {args.output.stat().st_size} bytes, SHA256 {checksum}")


if __name__ == "__main__":
    main()
