## KernelGeneration
KernelGeneration is a repository created to evaluate GPU kernel metrics, including:
- correctness
- performance 
- portability

### Project Structure (Before Cloning)
```
kernelGen/
├── README.md
├── install.py
└── operators/
    ├── bf16_layernorm/
    └── bf16_matmul/
```
- install.py — script to install dependencies, copy operators, and clone TritonBench and KernelLLM
- operators/ — contains GPU kernel operators
- **TODO/WIP**: run_benchmarks.py — script to run all operators and collect results into a JSON file for analysis.

### Evaluated Kernels
- `MatMul`
- `LayerNorm` - **WIP** add KernelLLM 
- **WIP**: `GELU`

### Backend Tested:
- TorchInductor
- Triton
- [KernelLLM](https://huggingface.co/facebook/KernelLLM)
- **WIP**: Mako (based on kernelLLM)

### Tested on: [TritonBench](https://github.com/meta-pytorch/tritonbench/tree/main)

## Usage
Run the setup and operator installation script: 
```bash
python install.py
```
