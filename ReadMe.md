## KernelGeneration
KernelGeneration is a repository created to evaluate GPU kernel metrics, including:
- correctness
- performance
- portability

### Requirements
- **PyTorch nightly** (CUDA or ROCm). Install the nightly that matches your system *before* running the installer.
  - CUDA example:
    ```bash
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124
    ```
  - ROCm example:
    ```bash
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm6.4
    ```
  Use the official PyTorch install selector to pick the correct nightly wheel for your OS/driver stack.

- (Optional) **KernelLLM prompts**  
  If you want to try KernelLLM-style prompting, see the templates in `./prompt/`.  
  *Note:* the installer does **not** clone KernelLLM or install `transformers/accelerate`.

### Environment
After setup, export:
```bash
export TRITONBENCH_RUN_CONFIG="$(pwd)/benchmark_helion_runner.yaml"
export TRITONBENCH_HELION_PATH="$(pwd)/helion"
```

### Project Structure (before running `install.py`)
```
kernelGen/
├── README.md
├── install.py
├── benchmark_helion_runner.yaml
├── prompt/
│   └── ...                # optional prompt templates for KernelLLM-style tests
└── operators/
    ├── bf16_layernorm/
    └── bf16_matmul/
```

- `install.py` — installs TritonBench, **clones Helion (kernel-gen-rh branch) and installs it in editable mode**, and copies local operators into TritonBench.  
  It **does not** install PyTorch or KernelLLM.
- `operators/` — GPU kernel operators (copied into TritonBench during install).

### Evaluated Kernels
- `MatMul`
- `LayerNorm` — **WIP** (optional KernelLLM prompts available in `./prompt/`)
- **WIP**: `GELU`

### Backends Tested
- TorchInductor
- Triton
- Helion
- [KernelLLM](https://huggingface.co/facebook/KernelLLM) *(optional, prompts only via `./prompt/`)*
- **WIP**: Mako (based on KernelLLM)

### Tested on
- [TritonBench](https://github.com/meta-pytorch/tritonbench/tree/main)

## Usage
1. **Install PyTorch nightly** (see *Requirements* above).
2. **Run the installer** (clones TritonBench, clones & installs Helion, and copies operators):
   ```bash
   python install.py
   ```
3. **Export environment variables** (adjust paths as needed):
   ```bash
   export TRITONBENCH_RUN_CONFIG="$(pwd)/benchmark_helion_runner.yaml"
   export TRITONBENCH_HELION_PATH="$(pwd)/helion"
   ```
4. Run the benchmark:
   ```bash
   python runner.py 
   ```
5. Check the results:
You can check the benchmark results under ./results 
