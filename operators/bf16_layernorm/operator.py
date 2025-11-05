import argparse
import itertools
import math
import os
from typing import Callable, Generator, List, Optional, Tuple

import torch
import triton
import triton.language as tl
#import triton.profiler as proton

from .kernels import layer_norm


from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    Mode,
    register_benchmark,
    register_metric,
)
from .kernels import layer_norm

class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["latency", "accuracy", "best_config"]

    class LayerNormWrapper:
        """Wrap Triton layer_norm to match BenchmarkOperator expected interface."""
        def __init__(self, x, weight, bias, eps=1e-5):
            self.x = x
            self.weight = weight
            self.bias = bias
            self.eps = eps
        # we will focus only on forward pass for now
        def forward(self):
            self.y = layer_norm(self.x, self.x.shape[-1:], self.weight, self.bias, self.eps)
            return self.y
    
    @register_benchmark()
    def triton_layernorm(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
        ln = self.LayerNormWrapper(x, weight, bias, eps)
        def run():
            y = ln.forward()
            return y

        return run

    @register_benchmark(baseline=True)
    def torch_layernorm(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
        return lambda: torch.nn.functional.layer_norm(x, x.shape[-1:], weight, bias, eps).to(torch.bfloat16)

    @register_benchmark()
    def torch_compile_layernorm(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
        @torch.compile(mode="max-autotune-no-cudagraphs")
        def _inner():
            return torch.nn.functional.layer_norm(x, x.shape[-1:], weight, bias, eps).to(torch.bfloat16)
        return _inner

    def accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output = fn()
        baseline_output = baseline_fn()
        result = torch.allclose(output, baseline_output, atol=1e-2, rtol=1e-2)
        assert result, "Outputs are not close!"
        return result

    @register_metric()
    def gbps(self, fn, example_inputs, metrics: BenchmarkOperatorMetrics):
        x, weight, bias = example_inputs
        y = fn()
        total_bytes = (x.numel() + weight.numel() + bias.numel()) * x.element_size()
        return total_bytes / metrics.latency * 1e-9  # GB/s

    @register_metric(x_only=True)
    def input_shape(self, fn_name: str, example_inputs, metrics: BenchmarkOperatorMetrics):
        return example_inputs[0].shape

    def generate_sizes(self) -> List[Tuple[int, int, int]]:
        sizes = []
        for M in [512, 1151, 1024, 2048]:
            for N in [512, 1024, 2048, 8192]:
                sizes.append((M, N))
        return sizes

    def get_input_iter(self) -> Generator:
        dtype = torch.bfloat16
        eps = 1e-5
        for size in self.generate_sizes():
            w_shape = (size[-1], )
            x = -2.3 + 0.5 * torch.rand(size, device=self.device, dtype=dtype)
            x.requires_grad_(True)
            weight = torch.rand(w_shape, dtype=dtype, device=self.device, requires_grad=True)
            bias = torch.rand(w_shape, dtype=dtype, device=self.device, requires_grad=True)
            yield x, weight, bias, eps