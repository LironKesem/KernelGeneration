import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def layer_norm_fwd_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    M,  # number of rows
    N,  # number of features per row
    eps,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    row_start = row_id * N

    # compute mean and variance
    sum_ = tl.zeros([1], dtype=tl.float32)
    sumsq_ = tl.zeros([1], dtype=tl.float32)

    offs = tl.arange(0, BLOCK_SIZE)

    for col_start in range(0, N, BLOCK_SIZE):
        idx = col_start + offs
        mask = idx < N
        x = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0).to(tl.float32)
        sum_ += tl.sum(x, axis=0)
        sumsq_ += tl.sum(x * x, axis=0)

    n_float = tl.full([1], N, dtype=tl.float32)
    mean = sum_ / n_float
    var = sumsq_ / n_float - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # normalize and apply affine
    for col_start in range(0, N, BLOCK_SIZE):
        idx = col_start + offs
        mask = idx < N

        x = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + idx, mask=mask, other=0.0).to(tl.float32)

        y = (x - mean) * rstd
        y = y * w + b

        # cast to output dtype
        y_out = tl.where(mask, y, 0.0)
        if IS_FP16:
            y_out = y_out.to(tl.float16)
        elif IS_BF16:
            y_out = y_out.to(tl.bfloat16)
        else:
            y_out = y_out.to(tl.float32)

        tl.store(y_ptr + row_start + idx, y_out, mask=mask)


def _next_power_of_2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def mako_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA."
    assert x.shape[-1] == weight.numel() == bias.numel(), "Mismatched normalized_shape."

    orig_dtype = x.dtype
    assert orig_dtype in (torch.float16, torch.bfloat16, torch.float32), "Unsupported dtype."

    # Flatten leading dims into rows, keep last dim as features
    N = x.shape[-1]
    M = x.numel() // N

    x_flat = x.view(M, N).contiguous()
    w = weight.contiguous()
    b = bias.contiguous()

    y_flat = torch.empty_like(x_flat)

    BLOCK_SIZE = min(1024, _next_power_of_2(N))
    num_warps = 8 if BLOCK_SIZE >= 1024 else (4 if BLOCK_SIZE >= 256 else 2)

    grid = lambda meta: (M,)

    IS_FP16 = int(orig_dtype == torch.float16)
    IS_BF16 = int(orig_dtype == torch.bfloat16)

    layer_norm_fwd_kernel[grid](
        x_flat,
        w,
        b,
        y_flat,
        M,
        N,
        float(eps),
        IS_FP16=IS_FP16,
        IS_BF16=IS_BF16,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=2,
    )

    return y_flat.view_as(x)
"""

class ModelNew(nn.Module):
    #Optimized LayerNorm with fused affine using Triton.
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        return triton_layer_norm(x, weight, bias, eps)


# Provide both x, weight, and bias as inputs
def get_inputs():
    N = 1024
    normalized_shape = 8192
    x = torch.rand((N, normalized_shape), dtype=torch.bfloat16, device='cuda')
    weight = torch.ones(normalized_shape, dtype=torch.bfloat16, device='cuda')
    bias = torch.zeros(normalized_shape, dtype=torch.bfloat16, device='cuda')
    eps = 1e-5
    return [x, weight, bias, eps]


def get_init_inputs():
    return [(8192)]
"""