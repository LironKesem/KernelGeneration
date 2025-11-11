#!/usr/bin/env python3
import subprocess
import sys
import shutil
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


def safe_copytree(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


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
    run(f"{sys.executable} install.py", cwd=str(ROOT / "tritonbench"))

    # Step 4: Copy the operators from ./operators into tritonbench/tritonbench/operators/<name>
    source_dir = ROOT / "operators"
    dest_dir = ROOT / "tritonbench" / "tritonbench" / "operators"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if source_dir.exists():
        for item in sorted(source_dir.iterdir()):
            if item.is_dir():
                target = dest_dir / item.name
                safe_copytree(item, target)
                print(f"Copied {item.name} -> {target}")
    else:
        print("No local ./operators directory found; skipping operator copy.")

    # Step 5: Clone custom Helion (kernel-gen-rh branch) and install editable
    helion_dir = ROOT / "helion"
    if not helion_dir.exists():
        # Clone only the needed branch
        run(
            "git clone -b kernel-gen-rh --single-branch https://github.com/fulvius31/helion helion"
        )
    else:
        print("helion folder already exists, updating branch kernel-gen-rh...")
        run("git fetch origin", cwd=str(helion_dir))
        run("git checkout kernel-gen-rh", cwd=str(helion_dir))
        run("git pull --ff-only", cwd=str(helion_dir))

    # Editable install
    run(f"{sys.executable} -m pip install -e helion")

    # Install pytest
    run(f"{sys.executable} -m pip install pytest")
    print("\nSetup complete!")


if __name__ == "__main__":
    main()
