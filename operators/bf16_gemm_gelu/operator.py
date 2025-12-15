from typing import Callable, Generator, List, Tuple

import torch
import triton

import torch._inductor.config as inductor_config
from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
    register_x_val,
)
from .triton_kernel import triton_gemm_gelu_kernel
from .kernelllm import KGemmGelu
from .mako_kernel import MakoFusGelu


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["latency", "accuracy", "speedup", "tflops"]

    @register_benchmark(baseline=True)
    def torch_gemm_gelu(self, a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor):
        def _fn():
            out = torch.matmul(a, b)
            out = out + bias  # broadcast along dim 0
            out = torch.nn.functional.gelu(out)  # uses erf internally
            return out

        return _fn

    @register_benchmark()
    def mako_gemm_gelu(self, a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor):
        mako_gelu = MakoFusGelu()
        return lambda: mako_gelu(a, b, bias)

    @register_benchmark()
    def kernelllm_gemm_gelu(self, a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor):
        kllm_gelu = KGemmGelu()
        return lambda: kllm_gelu.forward(a, b, bias)

    @register_benchmark()
    def torch_compile_max_gemm_gelu(
        self, a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor
    ):
        torch._dynamo.reset()
        with inductor_config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="TRITON",
        ):

            def f(a, b, bias):
                out = a.matmul(b)
                out = out + bias  # broadcast along dim 0
                out = torch.nn.functional.gelu(out)
                return out

            compiled = torch.compile(f, dynamic=False)
            # Force compilation & autotune before timing
            compiled(a, b, bias)

        return lambda: compiled(a, b, bias)

    @register_benchmark()
    def torch_compile_default_gemm_gelu(
        self, a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor
    ):
        torch._dynamo.reset()
        with inductor_config.patch(
            max_autotune_gemm_backends="TRITON",
        ):

            def f(a, b, bias):
                out = a.matmul(b)
                out = out + bias  # broadcast along dim 0
                out = torch.nn.functional.gelu(out)
                return out

            compiled = torch.compile(f, dynamic=False)
            # Force compilation & autotune before timing
            compiled(a, b, bias)

        return lambda: compiled(a, b, bias)

    @register_benchmark()
    def triton_gemm_gelu_kernel(
        self, a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor
    ):
        assert a.shape[1] == b.shape[0], "Incompatible dimensions"
        assert (
            a.is_contiguous() and b.is_contiguous() and bias.is_contiguous()
        ), "All inputs must be contiguous"
        assert (
            a.dtype == torch.bfloat16
            and b.dtype == torch.bfloat16
            and bias.dtype == torch.bfloat16
        ), "Matrices must be bfloat16"
        M, K = a.shape
        K2, N = b.shape
        assert K == K2, "Inner dimensions must match"
        assert bias.shape[0] == N, "Bias shape must be [N]"

        c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)

        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

        def _inner():
            triton_gemm_gelu_kernel[grid](
                a_ptr=a,
                b_ptr=b,
                c_ptr=c,
                bias_ptr=bias,
                M=M,
                N=N,
                K=K,
                stride_am=a.stride(0),
                stride_ak=a.stride(1),
                stride_bk=b.stride(0),
                stride_bn=b.stride(1),
                stride_cm=c.stride(0),
                stride_cn=c.stride(1),
            )
            return c

        return _inner

    def accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output = fn()
        baseline_output = baseline_fn()
        result = torch.allclose(output, baseline_output, atol=1e-5, rtol=1e-2)
        if not result:
            print("Accuracy check failed!")
        return result

    @register_metric()
    def input_shape(
        self, fn_name: str, example_inputs, metrics: BenchmarkOperatorMetrics
    ):
        return example_inputs[0].shape

    def _latency_seconds(self, lat) -> float:
        # check if it's numeric already
        if isinstance(lat, (int, float)):
            return float(lat) / (1e3)  # convert ms -> s

        # TritonBench Latency object: prefer p50, fall back to min/max, then median of times
        for attr in ("p50", "min", "max"):
            v = getattr(lat, attr, None)
            if isinstance(v, (int, float)):
                return float(v) / (1e3)  # convert ms -> s

        times = getattr(lat, "times", None)
        if isinstance(times, (list, tuple)) and times:
            # median
            s = sorted(float(t) for t in times)
            n = len(s)
            if n % 2 == 1:
                median = s[n // 2]
            else:
                median = (s[n // 2 - 1] + s[n // 2]) / 2
            return median / (1e3)  # convert ms -> s

        raise TypeError(
            f"Unsupported latency type: {type(lat)}; available fields: {dir(lat)}"
        )

    @register_metric()
    def tflops(self, fn_name, example_inputs, metrics):
        # All benchmarks for this operator use (a, b, bias)
        a, b, bias = example_inputs
        M, K = a.shape
        K2, N = b.shape
        assert K == K2, f"Incompatible dimensions: {K} != {K2}"

        # GEMM FLOPs: 2 * M * N * K (mul + add)
        gemm_flops = 2.0 * M * N * K

        # Optional: rough count for bias add + GELU
        # bias add: M * N
        # GELU: ~4 ops per element (very rough)
        extra_flops = M * N * (1 + 4)

        total_flops = gemm_flops + extra_flops

        sec = self._latency_seconds(metrics.latency)
        return total_flops / max(sec, 1e-12) / 1e12  # TFLOPs

    def generate_sizes(self) -> List[Tuple[int, int, int]]:
        return [
            (32, 64, 16),
            (128, 256, 64),
            (512, 1024, 128),
            (1024, 2048, 256),
            (512, 2048, 4096),
        ]

    def get_input_iter(self) -> Generator:
        for size in self.generate_sizes():
            M, K, N = size
            A = torch.rand((M, K), device=self.device, dtype=torch.bfloat16)
            B = torch.rand((K, N), device=self.device, dtype=torch.bfloat16)
            bias = torch.rand((N,), device=self.device, dtype=torch.bfloat16)
            yield A, B, bias

    @register_x_val(label="(M, K, N)")
    def get_x_val(self, example_inputs) -> Tuple[int, int, int]:
        a, b, bias = example_inputs
        M, K = a.size()
        K_, N = b.size()
        return (M, K, N)
