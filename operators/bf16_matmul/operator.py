from typing import Callable, Generator, List, Tuple

import torch
import triton


from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
    register_x_val,
    register_x_val,
)
from .triton_kernel import matmul_kernel
from .kernelllm import KMatmul
from .mako_kernel import mako_kernel

class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["latency", "accuracy", "speedup", "tflops"]

    @register_benchmark()
    def triton_matmul(self, a: torch.Tensor, b: torch.Tensor):
        assert a.shape[1] == b.shape[0], "Incompatible dimensions"
        assert a.is_contiguous(), "Matrix A must be contiguous"
        assert (
            a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
        ), "Matrics must be of type bfloat16"
        M, K = a.shape
        K, N = b.shape
        # Allocates output.
        c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

        def _inner():
            matmul_kernel[grid](
                a,
                b,
                c,  #
                M,
                N,
                K,  #
                a.stride(0),
                a.stride(1),  #
                b.stride(0),
                b.stride(1),  #
                c.stride(0),
                c.stride(1),  #
            )
            return c

        return _inner

    @register_benchmark()
    def kernelllm_matmul(self, a: torch.Tensor, b: torch.Tensor):
        kllm_matmul= KMatmul()
        return lambda:kllm_matmul.forward(a,b)

    @register_benchmark()
    def mako_matmul(self, a: torch.Tensor, b: torch.Tensor):
        return lambda: mako_kernel(a, b)

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
            return float(lat) / (1e3)  # convert ms -> s

        # TritonBench Latency object: prefer p50, fall back to min/max, then median of times
        for attr in ("p50", "min", "max"):
            v = getattr(lat, attr, None)
            if isinstance(v, (int, float)):
                return float(v) / (1e3) # convert ms -> s

        times = getattr(lat, "times", None)
        if isinstance(times, (list, tuple)) and times:
            # median
            s = sorted(float(t) for t in times)
            n = len(s)
            if n%2 == 1 :
                median = s[n // 2]
            else:
                median =(s[n //2 -1] + s[n//2]) /2 
            return median / (1e3)  # convert ms -> s

        raise TypeError(
            f"Unsupported latency type: {type(lat)}; available fields: {dir(lat)}"
        )

    @register_metric()
    def tflops(self, fn_name, example_inputs, metrics):
        x, y = example_inputs
        m, k = x.shape
        n = y.shape[1]
        flops = 2.0 * m * n * k
        sec = self._latency_seconds(metrics.latency)
        return flops / max(sec, 1e-12) / 1e12

    def generate_sizes(self) -> List[Tuple[int, int, int]]:
        return [
            (256, 384, 512),
            (384, 512, 640),
            (512, 640, 768),
            (640, 768, 896),
            (768, 896, 1024),
            (896, 1024, 1152),
            (1024, 1152, 1280),
            (1152, 1280, 1408),
            (1280, 1408, 1536),
            (1408, 1536, 1664),
            (1536, 1664, 1792),
            (1664, 1792, 1920),
            (1792, 1920, 2048),
            (2048, 2176, 2304),
            (2176, 2304, 2432),
            (2304, 2432, 2560),
            (2432, 2560, 2688),
            (2560, 2688, 2816),
            (2688, 2816, 2944),
            (2816, 2944, 3072),
            (2944, 3072, 3200),
            (3072, 3200, 3328),
            (3200, 3328, 3456),
            (3328, 3456, 3584),
            (3456, 3584, 3712),
            (3584, 3712, 3840),
            (3712, 3840, 3968),
            (3840, 3968, 4096),
            (3968, 4096, 4224),
            (4096, 4224, 4352)
        ]

    def get_input_iter(self) -> Generator:
        for size in self.generate_sizes():
            M, K, N = size
            x = torch.rand((M, K), device=self.device, dtype=torch.bfloat16)
            y = torch.rand((K, N), device=self.device, dtype=torch.bfloat16)
            yield x, y
    
    @register_x_val(label="(M, K, N)")
    def get_x_val(self, example_inputs) -> Tuple[int, int, int]:
        a, w = example_inputs
        m, k = a.size()
        k, n = w.size()
        return (m, k, n)
