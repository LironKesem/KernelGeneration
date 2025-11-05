import argparse
import itertools
import math
import os
from typing import Callable, Generator, List, Optional, Tuple

import torch
import triton
import triton.language as tl
#import triton.profiler as proton



from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)
from .kernels import matmul_kernel
from .llm_kernel import call_64_128_32, call_256_256_32

class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["latency", "accuracy", "best_config"]

    @register_benchmark()
    def triton_matmul(self,  a: torch.Tensor, b: torch.Tensor):
        print(f"A SIZE: {a.size()}")
        print(f"B SIZE: {b.size()}")

        # Check constraints.
        assert a.shape[1] == b.shape[0], "Incompatible dimensions"
        assert a.is_contiguous(), "Matrix A must be contiguous"
        assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16, "Matrics must be of type bfloat16"
        M, K = a.shape
        K, N = b.shape
        # Allocates output.
        c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
        
        def _inner():

            #session_id = proton.start(name="matmul_profile", context="python")
            #with proton.scope("matmul_kernel"):

            matmul_kernel[grid](
                a, b, c,  #
                M, N, K,  #
                a.stride(0), a.stride(1),  #
                b.stride(0), b.stride(1),  #
                c.stride(0), c.stride(1),  #
            )
            #proton.finalize()
            return c
        return _inner

    @register_benchmark()
    def kernelllm_matmul(self,  a: torch.Tensor, b: torch.Tensor):
        M, K, N = a.shape[0], a.shape[1], b.shape[1]
        if (M, K, N) == (256, 256, 32):
            return lambda: call_256_256_32([a, b])
        elif (M, K, N) == (64, 128, 32):
            return lambda: call_64_128_32([a, b])
        else:
            raise RuntimeError(f"No kernel implemented for shape ({M}, {K}, {N})")

    @register_benchmark(baseline=True)
    def torch_matmul(self, a: torch.Tensor, b: torch.Tensor):
        return lambda: torch.matmul(a, b)

    @register_benchmark()
    def torch_compile_matmul(self, a: torch.Tensor, b: torch.Tensor):
        @torch.compile(mode="max-autotune-no-cudagraphs")
        def _inner(a, b):
            return torch.matmul(a, b)
        return lambda: _inner(a, b)

    def accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output = fn()
        baseline_output = baseline_fn()
        result =torch.allclose(output, baseline_output, atol=1e-2, rtol=1e-2)
        assert result, "Outputs are not close!"
        return result

    @register_metric()
    def gbps(self, fn, example_inputs, metrics: BenchmarkOperatorMetrics):
        a, b = example_inputs
        # bytes moved = size of a + size of b + size of c
        c = fn()
        total_bytes = (a.numel() + b.numel() + c.numel()) * a.element_size()
        return total_bytes / metrics.latency * 1e-9  # GB/s

    @register_metric()
    def input_shape(self, fn_name: str, example_inputs, metrics: BenchmarkOperatorMetrics):
        return example_inputs[0].shape


    def generate_sizes(self) -> List[Tuple[int, int, int]]:
        # use those shapes without kernelllm_matmul 
        #     [(32, 64, 16),(128, 256, 64),(512, 1024, 128), (1024, 2048, 256)]
        return [(256, 256, 32),(64, 128, 32)]


    def get_input_iter(self) -> Generator:
        for size in self.generate_sizes():
            M, K, N = size
            x = torch.rand((M, K), device=self.device, dtype=torch.bfloat16)
            y = torch.rand((K, N), device=self.device, dtype=torch.bfloat16)
            yield x, y