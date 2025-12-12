#!/usr/bin/env python3
import subprocess
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd, cwd=None):
    print(f"\n>>> Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Error running command: {cmd}")
        sys.exit(result.returncode)


def ensure_torch():
    try:
        import torch  # noqa: F401
    except Exception:
        print(
            "\n[!] PyTorch is not installed. This project requires a PyTorch nightly.\n"
            "    Please install the correct nightly (CUDA or ROCm),\n"
            "    then re-run this installer."
        )
        sys.exit(1)


def safe_symlink(src: Path, dst: Path):
    if dst.exists():
        os.unlink(dst)
    os.symlink(src, dst)


def main():
    ensure_torch()

    # Step 1: Clone TritonBench
    if not (ROOT / "tritonbench").exists():
        run("git clone https://github.com/meta-pytorch/tritonbench.git")
    else:
        print("tritonbench folder already exists, skipping clone.")

    # Step 2: Initialize submodules
    run("git submodule update --init --recursive", cwd=str(ROOT / "tritonbench"))

    # Step 3: Install TritonBench
    run(f"{sys.executable} -m pip install -r requirements.txt", cwd=str(ROOT / "tritonbench"))
    run(f"{sys.executable} -m pip install -e .", cwd=str(ROOT / "tritonbench"))
    # Step 4: Copy the operators from ./operators into tritonbench/tritonbench/operators/<name>
    source_dir = ROOT / "operators"
    dest_dir = ROOT / "tritonbench" / "tritonbench" / "operators"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if source_dir.exists():
        for item in sorted(source_dir.iterdir()):
            if item.is_dir():
                target = dest_dir / item.name
                safe_symlink(item, target)
                print(f"Copied {item.name} -> {target}")
    else:
        print("No local ./operators directory found; not creating symlink.")

    # Step 5: Clone Helion
    helion_dir = ROOT / "helion"
    if not helion_dir.exists():
        run("git clone https://github.com/pytorch/helion helion")

    # Editable install
    run(f"{sys.executable} -m pip install -e helion")

    # Install pytest
    run(f"{sys.executable} -m pip install pytest")
    print("\nSetup complete!")


if __name__ == "__main__":
    main()
