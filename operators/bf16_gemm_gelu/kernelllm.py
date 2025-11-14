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
def triton_poi_fused_add_gelu_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 4096
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


def call_512_2048_4096(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (512, 2048), (2048, 1))
    assert_size_stride(arg1_1, (2048, 4096), (4096, 1))
    assert_size_stride(arg2_1, (4096,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((512, 4096), (4096, 1), torch.bfloat16)
        extern_kernels.mm(arg0_1, arg1_1, out=buf0)
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((512, 4096), (4096, 1), torch.bfloat16)
        get_raw_stream(0)
        XBLOCK =512
        grid = lambda meta: (triton.cdiv(2097152, meta['XBLOCK']),)
        triton_poi_fused_add_gelu_0[grid](buf0, arg2_1, buf1, 
            2097152, XBLOCK=512, num_warps=8, num_stages=1)
        del arg2_1
        del buf0
    return buf1,


# class ModelNew(nn.Module):
#     def __init__(self):
#         super(ModelNew, self).__init__()
#         self.gelu = nn.GELU()  # GELU module

#     def forward(self, input_0, input_1, input_2):
#         arg0_1 = input_0
#         arg1_1 = input_1
#         arg2_1 = input_2
#         output = call([arg0_1, arg1_1, arg2_1])
#         return output[0]


# M, K, N = 512, 2048, 4096
# device= 'cuda'
# a = torch.randn(M, K, device=device, dtype=torch.bfloat16)
# b = torch.randn(K, N, device=device, dtype=torch.bfloat16)
# bias = torch.randn(N, device=device, dtype=torch.bfloat16)  # broadcast along dim 0

# ref_out = torch.matmul(a, b) + bias
# gelu = nn.GELU()
# ref_out = gelu(ref_out)

# model_new = ModelNew().cuda()
# with torch.no_grad():
#     triton_out = model_new(a, b, bias)

# assert torch.allclose(ref_out, triton_out, atol=1e-2, rtol=1e-2)
# print("Verification passed! Triton output matches PyTorch reference.")
