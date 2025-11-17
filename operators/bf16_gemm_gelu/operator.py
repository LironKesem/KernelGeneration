from typing import Callable, Generator, List, Tuple

import torch
import triton

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
    register_x_val
)
from .triton_kernel import triton_gemm_gelu_kernel
from .kernelllm import call_512_2048_4096
from .mako_kernel import fused_gemm_bias_gelu

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

    # NOTE: it uses the tahn approximate, 
    #.      tested compare the torch with default(erf) and tahn the test fails(accuracy check)
    # @register_benchmark(baseline=True)
    # def mako_gemm_gelu(self, a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor):
    #         return lambda:fused_gemm_bias_gelu(a, b, bias)


    @register_benchmark()
    def kernelllm_gemm_gelu(self, a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor):
        M, K = a.shape
        K2, N = b.shape
        assert K == K2, "Incompatible dimensions"
        if (M, K, N) == (512, 2048, 4096):
            return lambda: call_512_2048_4096([a, b, bias])[0]
        #elif (M, K, N) == (1024, 8192, 1024):
        #    return lambda: call([a, b, bias])[0]
        else:
            raise RuntimeError(f"No kernel implemented for shape ({M}, {K}, {N})")

    @register_benchmark()
    def torch_compile_gemm_gelu(self, a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor):
        @torch.compile(mode="max-autotune-no-cudagraphs")
        def _inner(a, b, bias):
            out = torch.matmul(a, b)
            out = out + bias  # broadcast along dim 0
            out =  torch.nn.functional.gelu(out)
            return out

        return lambda: _inner(a, b, bias)

    # TODO: inside the triton kernel we need to check how we can use triton erf function
    @register_benchmark()
    def triton_gemm_gelu_kernel(self, a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor):
        assert a.shape[1] == b.shape[0], "Incompatible dimensions"
        assert a.is_contiguous() and b.is_contiguous() and bias.is_contiguous(), "All inputs must be contiguous"
        assert (
            a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16 and bias.dtype == torch.bfloat16
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
        result = torch.allclose(output, baseline_output, atol=1e-2, rtol=1e-2)
        assert result, "Outputs are not close!"
        return result

    @register_metric()
    def input_shape(
        self, fn_name: str, example_inputs, metrics: BenchmarkOperatorMetrics
    ):
        return example_inputs[0].shape

    def _latency_seconds(self, lat) -> float:
        # check if it's numeric already
        if isinstance(lat, (int, float)):
            return float(lat)

        # TritonBench Latency object: prefer p50, fall back to min/max, then median of times
        for attr in ("p50", "min", "max"):
            v = getattr(lat, attr, None)
            if isinstance(v, (int, float)):
                return float(v)

        times = getattr(lat, "times", None)
        if isinstance(times, (list, tuple)) and times:
            # median
            s = sorted(float(t) for t in times)
            return s[len(s) // 2]

        raise TypeError(
            f"Unsupported latency type: {type(lat)}; available fields: {dir(lat)}"
        )

    def generate_sizes(self) -> List[Tuple[int, int, int]]:
        # use those shapes without kernelllm_matmul
        #     [(32, 64, 16),(128, 256, 64),(512, 1024, 128), (1024, 2048, 256)]
        return [(512, 2048, 4096)]

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