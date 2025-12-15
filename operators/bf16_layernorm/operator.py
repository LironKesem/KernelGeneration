from typing import Callable, Generator, List, Optional, Tuple

import torch
import triton.language as tl

from .triton_kernel import layer_norm


from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
    register_x_val,
)
from .triton_kernel import layer_norm
from .kernelllm import KLayerNorm
from .mako_kernel import mako_layernorm


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["latency", "accuracy", "best_config", "gbps"]

    class LayerNormWrapper:
        """Wrap Triton layer_norm to match BenchmarkOperator expected interface."""

        def __init__(self, x, weight, bias, eps=1e-5):
            self.x = x
            self.weight = weight
            self.bias = bias
            self.eps = eps

        # we will focus only on forward pass for now
        def forward(self):
            self.y = layer_norm(
                self.x, self.x.shape[-1:], self.weight, self.bias, self.eps
            )
            return self.y

    @register_benchmark()
    def triton_layernorm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-5,
    ):
        ln = self.LayerNormWrapper(x, weight, bias, eps)

        def run():
            y = ln.forward()
            return y

        return run

    @register_benchmark()
    def kernelllm_layernorm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-5,
    ):
        kllm_layernorm = KLayerNorm()
        return lambda: kllm_layernorm(x, weight, bias)

    @register_benchmark()
    def mako_layernorm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-5,
    ):
        return lambda: mako_layernorm(x, weight, bias, eps)

    @register_benchmark(baseline=True)
    def torch_layernorm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-5,
    ):
        return lambda: torch.nn.functional.layer_norm(
            x, x.shape[-1:], weight, bias, eps
        ).to(torch.bfloat16)

    @register_benchmark()
    def torch_compile_max_layernorm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-5,
    ):
        @torch.compile(mode="max-autotune")
        def _inner():
            return torch.nn.functional.layer_norm(
                x, x.shape[-1:], weight, bias, eps
            ).to(torch.bfloat16)

        return _inner

    @register_benchmark()
    def torch_compile_default_layernorm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-5,
    ):
        @torch.compile(mode="default")
        def _inner():
            return torch.nn.functional.layer_norm(
                x, x.shape[-1:], weight, bias, eps
            ).to(torch.bfloat16)

        return _inner

    def accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output = fn()
        baseline_output = baseline_fn()
        result = torch.allclose(output, baseline_output, atol=1e-5, rtol=1e-2)
        if not result:
            print("Accuracy check failed!")
        return result

    @register_metric()
    def gbps(self, fn, example_inputs, metrics: BenchmarkOperatorMetrics):
        # we only going to test the forward,
        x = example_inputs[0]
        base = x.numel() * x.element_size() / metrics.latency * 1e-6
        return 2 * base

    @register_metric(x_only=True)
    def input_shape(
        self, fn_name: str, example_inputs, metrics: BenchmarkOperatorMetrics
    ):
        return example_inputs[0].shape

    def generate_sizes(self) -> List[Tuple[int, int]]:
        return [(512, 512), (32, 1024), (2048, 2048)]

    def get_input_iter(self) -> Generator:
        assert self.dtype == torch.bfloat16, "Only bf16 is supported"
        eps = 1e-5
        for size in self.generate_sizes():
            normalized_shape = size[-1]
            x = -2.3 + 0.5 * torch.randn(
                size,
                dtype=self.dtype,
                device=self.device,
            )
            x.requires_grad_(True)

            weight = torch.ones(
                normalized_shape, device=self.device, dtype=self.dtype, requires_grad=True
            )
            bias = torch.zeros(
                normalized_shape, device=self.device, dtype=self.dtype, requires_grad=True
            )
            yield x, weight, bias, eps

    @register_x_val(label="(M, D)")
    def get_x_val(self, example_inputs) -> Tuple[int, int]:
        x, weight, bias, eps = example_inputs
        M, D = x.size()
        return (M, D)
