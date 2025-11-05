#!/usr/bin/env python3
import subprocess
import os
import sys
import shutil

def run(cmd, cwd=None):
    print(f"\n>>> Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Error running command: {cmd}")
        sys.exit(result.returncode)

# Step 1: Install PyTorch with ROCm
run("pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4")

# Step 2: Clone TritonBench
if not os.path.exists("tritonbench"):
    run("git clone https://github.com/meta-pytorch/tritonbench.git")
else:
    print("tritonbench folder already exists, skipping clone.")

# Step 3: Initialize submodules
run("git submodule update --init --recursive", cwd="tritonbench")

# Step 4: Install TritonBench
run("python install.py", cwd="tritonbench")

# Step 5: Copy the operators
source_dir = "./operators" 
dest_dir = os.path.join("tritonbench", "tritonbench", "operators")
os.makedirs(dest_dir, exist_ok=True) #make sure destination directory exists

for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    if os.path.isdir(folder_path):
        dest_path = os.path.join(dest_dir, folder_name)
        shutil.copytree(folder_path, dest_path)
        print(f"Copied {folder_name} to {dest_dir}")

for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    if os.path.isdir(folder_path):
        dest_path = os.path.join(dest_dir, folder_name)
        # Remove existing folder if it exists
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
        # Copy the folder
        shutil.copytree(folder_path, dest_path)

print(f"Copied {folder_name} to {dest_dir}")


# Step 6: Clone KernelLLM
if not os.path.exists("KernelLLM"):
    run("git clone https://huggingface.co/facebook/KernelLLM")
else:
    print("KernelLLM folder already exists, skipping clone.")

# Step 7: Install transformers and accelerate
run("pip install transformers accelerate")

print("\nSetup complete!")
tritonbench_dir = "./tritonbench" 
run("python run.py --op bf16_matmul", cwd=tritonbench_dir)

### TODO: add a script to collect the data(folders), maybe we will use the plot option.
### TODO: add a script to analyze the data -> dump it into a json file
