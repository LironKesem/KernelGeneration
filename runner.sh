#!/usr/bin/env bash

# prequisites: install.py  
ops=(
  "bf16_gemm_gelu"
  "bf16_layernorm"
  "bf16_matmul"
)


# Do not forget to change to rename the folder to the tested GPU model.
mkdir -p results_a30

echo "=== Running TritonBench (no Triton) ==="
echo "=== Running TritonBench (no Triton) ==="

for op in "${ops[@]}"; do
    echo "Running op: $op"
    python tritonbench/run.py --op "$op" \
        --output-json "results_a30/${op}_no_triton_a30.json" \
        > "results_a30/${op}_no_triton_a30.txt"
done


echo "=== Running TritonBench with Helion ==="

# Export Helion env variables
export TRITONBENCH_RUN_CONFIG="$(pwd)/benchmark_helion_runner.yaml"
export TRITONBENCH_HELION_PATH="$(pwd)/helion"

for op in "${ops[@]}"; do
    echo "Running op with Helion: $op"
    python tritonbench/run.py --op "$op" \
        --output-json "results_a30/${op}_helion_a30.json" \
        > "results_a30/${op}_helion_a30.txt"
done

echo "=== All benchmarks finished ==="
echo "Results saved in ./results_a30/"