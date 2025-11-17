import argparse
import itertools
import math
import os
from typing import Callable, Generator, List, Optional, Tuple

import torch
import triton
import triton.language as tl

from .kernels import layer_norm


from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    Mode,
    register_benchmark,
    register_metric,
    register_x_val
)
from .kernels import layer_norm
from .kernelllm import call_512_512, call_1024_8192
from .mako_kernel import mako_layernorm

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

    @register_benchmark()
    def kernelllm_layernorm(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
        M, D = x.shape

        if (M, D) == (512, 512):
            return lambda: call_512_512([weight, bias, x])[0]
        elif (M, D) == (1024, 8192):
            return lambda: call_1024_8192([weight, bias, x])[0]
        else:
            raise RuntimeError(f"No kernel implemented for shape ({M}, {D})")


    @register_benchmark()
    def mako_layernorm(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
        return lambda: mako_layernorm(x, weight, bias, eps)


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

    def generate_sizes(self) -> List[Tuple[int, int]]:
        return [(512, 512), (1024, 8192)]


    def get_input_iter(self) -> Generator:
        dtype = torch.bfloat16
        eps = 1e-5
        for size in self.generate_sizes():
            normalized_shape = (size[-1])
            x = torch.randn(size, device='cuda', dtype=torch.bfloat16)
            x.requires_grad_(True)

            weight = torch.ones(normalized_shape, device=self.device, dtype=dtype, requires_grad=True)
            bias = torch.zeros(normalized_shape, device=self.device, dtype=dtype, requires_grad=True)
            yield x, weight, bias, eps

    @register_x_val(label="(M, D)")
    def get_x_val(self, example_inputs) -> Tuple[int, int]:
        x, weight, bias, eps = example_inputs
        M, D = x.size()
        return (M, D)