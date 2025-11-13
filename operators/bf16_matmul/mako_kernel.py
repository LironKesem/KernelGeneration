import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_bf16_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        b_ptrs = B_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn

        a_mask = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)
        b_mask = (k + offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # tl.dot will accumulate in float32 for bf16 inputs
        acc += tl.dot(a, b, out_dtype=tl.float32)

        k += BLOCK_K

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def mako_kernel(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:

    if not (A.is_cuda and B.is_cuda):
        return torch.matmul(A, B)

    assert A.dim() == 2 and B.dim() == 2, "Inputs must be 2D matrices."
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, "Incompatible shapes for matmul."

    # Ensure contiguous for correct stride math
    A = A.contiguous()
    B = B.contiguous()

    # Output dtype follows PyTorch's matmul promotion rules
    out_dtype = torch.result_type(A, B)
    C = torch.empty((M, N), device=A.device, dtype=out_dtype)

    # Strides
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()

    # Tune block sizes for (256x32) * (32x256)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32  # K is small; use single K-tile to avoid loop overhead

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_bf16_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return C
