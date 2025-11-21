import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_gelu_0(in_ptr0, in_ptr1, out_ptr0, xnumel,ydim, XBLOCK:
    tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % ydim
    tmp0 = tl.load(in_ptr0 + x2, None)
    tmp1 = tl.load(in_ptr1 + x0, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(out_ptr0 + x2, tmp10, None)

# was designed for (512, 2048, 4096)
def call(args):
    A, B, bias = args
    args.clear()
    M, K = A.size() 
    K_, N = B.size()
    assert K == K_, "Incompatible dimensions"
    assert_size_stride(A, (M, K), (K, 1))
    assert_size_stride(B, (K, N), (N, 1))
    assert_size_stride(bias, (N,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        a_matmul_b = empty_strided_cuda((M, N), (N, 1), torch.bfloat16)
        extern_kernels.mm(A, B, out=a_matmul_b)
        del A
        del B
        output = empty_strided_cuda((M, N), (N, 1), torch.bfloat16)
        numel= output.numel()
        get_raw_stream(0)
        XBLOCK =M
        grid = lambda meta: (triton.cdiv(numel, meta['XBLOCK']),)
        triton_poi_fused_add_gelu_0[grid](a_matmul_b, bias, output, 
            numel,N, XBLOCK=M, num_warps=8, num_stages=1)
        del bias
        del a_matmul_b
    return output,

# KerenlLLM was originally designed for (512, 2048, 4096)
class KGemmGelu(nn.Module):
    def __init__(self):
        super(KGemmGelu, self).__init__()
        self.gelu = nn.GELU()

    def forward(self, a, b, bias):
        A = a
        B = b
        Bias = bias
        output = call([A, B, Bias])
        return output[0]
