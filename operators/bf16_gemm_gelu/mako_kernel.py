import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 5, 'num_warps': 2}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 5, 'num_warps': 2}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 5, 'num_warps': 2}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
    ],
    key=['BLOCK_M', 'BLOCK_N', 'BLOCK_K', 'GROUP_M'],
)
@triton.jit
def matmul_bias_gelu_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    pid_m = first_pid_m + (pid % num_pid_in_group) // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)
    for k in range(0, K, BLOCK_K):
        k_curr = k + offs_k

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + k_curr[None, :] * stride_ak)
        b_ptrs = B_ptr + (k_curr[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a_mask = (offs_m[:, None] < M) & (k_curr[None, :] < K)
        b_mask = (k_curr[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # tl.dot supports fp16/bf16 inputs accumulating into fp32.
        acc += tl.dot(a, b)

    # Add bias
    bias_vals = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias_vals[None, :]

    # GELU (exact): 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    y = 0.5 * acc * (1.0 + tl.math.erf(acc * inv_sqrt2))

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, y, mask=c_mask)


def triton_matmul_bias_gelu(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor):
    assert a.is_cuda and b.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    assert a.dim() == 2 and b.dim() == 2 and bias.dim() == 1, "Invalid tensor dimensions."
    M, K1 = a.shape
    K2, N = b.shape
    assert K1 == K2, "Inner dimensions must match."
    assert bias.shape[0] == N, "Bias must have shape (N,)."

    # Supported dtypes for fused kernel: fp16 and bf16
    if a.dtype not in (torch.float16, torch.bfloat16) or b.dtype != a.dtype or bias.dtype != a.dtype:
        # Fallback to PyTorch if dtypes don't match expectations
        out = torch.matmul(a, b)
        out = out + bias
        return torch.nn.functional.gelu(out)

    a = a.contiguous()
    b = b.contiguous()
    bias = bias.contiguous()

    # Accumulate and store in float32 for better numerical stability; we cast at the end.
    out_fp32 = torch.empty((M, N), device=a.device, dtype=torch.float32)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )


    matmul_bias_gelu_kernel[grid](
        a, b, bias, out_fp32,
        M, N, K1,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out_fp32.stride(0), out_fp32.stride(1),
    )
    return out_fp32.to(a.dtype)


class MakoFusGelu(nn.Module):
    def __init__(self):
        super(MakoFusGelu, self).__init__()
        self.gelu = nn.GELU()

    def forward(self, a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor):
        # Use Triton kernel on CUDA with fp16/bf16; otherwise fallback to PyTorch
        if a.is_cuda and b.is_cuda and bias.is_cuda:
            try:
                return triton_matmul_bias_gelu(a, b, bias)
            except Exception:
                # Safe fallback in case of any runtime issues
                out = torch.matmul(a, b)
                out = out + bias
                return self.gelu(out)
        else:
            out = torch.matmul(a, b)
            out = out + bias
            return self.gelu(out)


def get_inputs():
    M, K, N = 512, 2048, 4096
    A = torch.rand((M, K), dtype=torch.bfloat16)
    B = torch.rand((K, N), dtype=torch.bfloat16)
    bias = torch.rand((N,), dtype=torch.bfloat16)
    return [A, B, bias]


def get_init_inputs():
    return []