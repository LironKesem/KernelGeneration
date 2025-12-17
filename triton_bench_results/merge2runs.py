import json

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def merge_runs(run1, run2):
    """
    Merge benchmark_values from run2 into run1
    by matching on metric.name.
    """
    metric_map = {
        entry["metric"]["name"]: entry
        for entry in run1
    }

    for entry in run2:
        metric_name = entry["metric"]["name"]

        if metric_name not in metric_map:
            # If metric does not exist in run1, just add it
            metric_map[metric_name] = entry
        else:
            # Append benchmark_values
            metric_map[metric_name]["metric"]["benchmark_values"].extend(
                entry["metric"]["benchmark_values"]
            )

    # Return merged list
    return list(metric_map.values())

if __name__ == "__main__":
    run1 = load_json("triton_bench_results/AMD_Radeon_Graphics_gfx1200/AMD_Radeon_Graphics_gfx1200_MatMul_Raw_Results_Across_Different_Sizes/AMD_Radeon_Graphics_gfx1200_20251211_085835/benchmark_bf16_matmul_AMD_Radeon_Graphics_gfx1200.json")
    run2 = load_json("triton_bench_results/AMD_Radeon_Graphics_gfx1200/AMD_Radeon_Graphics_gfx1200_MatMul_Raw_Results_Across_Different_Sizes/AMD_Radeon_Graphics_gfx1200_20251216_081312/benchmark_bf16_matmul_AMD_Radeon_Graphics_gfx1200.json")
    merged = merge_runs(run1, run2)
    file_name ="merged_benchmark_bf16_matmul_AMD_Radeon_Graphics_gfx1200.json"
    with open(file_name, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged benchmark written to {file_name}")