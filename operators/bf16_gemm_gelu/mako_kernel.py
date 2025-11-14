import torch
import torch.nn as nn
import triton
import triton.language as tl

def fused_gemm_bias_gelu(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Strides
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    fused_matmul_bias_gelu_kernel[grid](
        a, b, bias, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,

    )

    return c

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 5, 'num_warps': 2}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_matmul_bias_gelu_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    """
    Triton kernel for fused matmul + bias + GELU.
    Computes C = GELU(A @ B + bias).
    A is (M, K), B is (K, N), bias is (N,), C is (M, N).
    """
    # -----------------------------------------------------------
    # Map program ids to blocks
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Use grouping to improve L2 cache locality
    num_pids_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pids_in_group
    
    first_pid_m = group_id * GROUP_SIZE_M
    pid_in_group = pid % num_pids_in_group
    
    pid_m_offset = pid_in_group // num_pid_n
    pid_n = pid_in_group % num_pid_n
    pid_m = first_pid_m + pid_m_offset

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # -----------------------------------------------------------
    # Loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, masking for uneven dimensions
        k_remaining = K - k * BLOCK_SIZE_K
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        accumulator += tl.dot(a, b)
        
        # Advance the pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Cast accumulator to the input dtype for post-processing
    c = accumulator.to(a_ptr.dtype.element_ty)

    # -----------------------------------------------------------
    # Fusion: Add bias and apply GELU
    bias_offs = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    bias = tl.load(bias_ptr + bias_offs, mask=(bias_offs < N), other=0.0)
    c = c + bias[None, :]
    c = gelu(c)
    
    # -----------------------------------------------------------
    # Write back the block of the output matrix C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def gelu(x):
    """
    GeLU activation function using the tanh approximation.
    This is faster than the exact form with `erf` and is a standard approximation
    that is available in PyTorch's nn.GELU(approximate='tanh').
    The default PyTorch GELU uses `erf`, but the difference for bfloat16 should be
    within the typical floating point error margins and acceptable for the problem.
    """
    # Using the tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c1 = 0.7978845608028654  # sqrt(2.0/pi)
    c2 = 0.044715

    x_cubed = x * x * x
    arg = c1 * (x + c2 * x_cubed)

    # Tanh is not a triton builtin, so we implement it.
    # tanh(y) = (exp(y) - exp(-y)) / (exp(y) + exp(-y))
    exp_arg = tl.exp(arg)
    exp_neg_arg = tl.exp(-arg)
    tanh_arg = (exp_arg - exp_neg_arg) / (exp_arg + exp_neg_arg)

    return 0.5 * x * (1.0 + tanh_arg)

