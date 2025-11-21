"""
this kernel taken from triton tutorials, see https://triton-lang.org/tutorials/02-layer-norm.html
we modified it to support only bfloat16 dtype and in the operator we only going to test forward pass.

TODO: add the auto tune! but try it in triton first and then port it here
Note: the normalization happens over the last dimension.
TODO: in the kernels.py and in bf16_matmul, I need to increase the shape size, and test edge conditions, so it will not fit nicely into blocks.
===========================================
"""


import torch

import triton
import triton.language as tl

try:
    # This is https://github.com/NVIDIA/apex, NOT the apex on PyPi, so it
    # should not be added to extras_require in setup.py.
    import apex
    HAS_APEX = True
except ModuleNotFoundError:
    HAS_APEX = False

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

## TODO: modify these configs to be more suitable for CUDA
def get_cuda_autotune_config():
    return[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
    ]

def get_hip_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE': 32}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
    ]

def get_autotune_config():
    if is_cuda():
        print("Using CUDA autotune config")
        return get_cuda_autotune_config()
    else:
        print("Using HIP autotune config")
        return get_hip_autotune_config()

# -------------------------------------------------------------------------
# Forward kernel
# -------------------------------------------------------------------------
@triton.autotune(
    configs=get_autotune_config(),
    key=['N'],
)
@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)


class LayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
        # heuristics for number of warps
        # enqueue kernel
        _layer_norm_fwd_fused[(M, )](  #
            x_arg, y, weight, bias, mean, rstd,  #
            x_arg.stride(0), N, eps)
        #ctx is for saving it for the backward pass
        #ctx.BLOCK_SIZE = BLOCK_SIZE
        #ctx.num_warps = num_warps
        #ctx.eps = eps
        return y


layer_norm = LayerNorm.apply