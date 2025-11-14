import torch
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_per_fused_native_layer_norm_0(in_out_ptr0, in_ptr0, out_ptr0,
    xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    #tl.full([RBLOCK], True, tl.int1) no needed
    rindex = tl.arange(0, RBLOCK)[:]
    #tl.full([RBLOCK], True, tl.int1) no needed
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 512 * x0), None).to(tl.float32)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3.to(tl.float32), 0))
    tmp6 = tl.full([1], 512, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 512.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + x0, tmp18, None)
    tl.store(out_ptr0 + x0, tmp8, None)


@triton.jit
def triton_poi_fused_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 512
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + x2, None)
    tmp1 = tl.load(in_ptr1 + x1, None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x0, None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + x0, None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + x2, tmp8, None)


def call_512_512(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (512,), (1,))
    assert_size_stride(primals_2, (512,), (1,))
    assert_size_stride(primals_3, (512, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((512, 1), (1, 1), torch.bfloat16)
        buf1 = empty_strided_cuda((512, 1), (1, 512), torch.bfloat16)
        buf3 = reinterpret_tensor(buf1, (512, 1), (1, 1), 0)
        del buf1
        get_raw_stream(0)
        grid0 = lambda META: (triton.cdiv(512, 1),)
        triton_per_fused_native_layer_norm_0[grid0](buf3, primals_3,
            buf0, 512, 512, num_warps=4, num_stages=1)
        buf4 = empty_strided_cuda((512, 512), (512, 1), torch.bfloat16)
        XBLOCK=512
        grid1 = lambda META: (triton.cdiv(262144, META['XBLOCK']),)

        triton_poi_fused_native_layer_norm_1[grid1](primals_3, buf0,
            buf3, primals_1, primals_2, buf4, 262144, XBLOCK=XBLOCK, num_warps
            =8, num_stages=1)
        del primals_1
        del primals_2
    return buf4, primals_3, buf0, buf3

"""
class ModelNew(nn.Module):
    #Simple model that performs Layer Normalization.
    def __init__(self, normalized_shape: tuple):
        #Initializes the LayerNorm layer.

        #Args:
        #    normalized_shape (tuple): Shape of the input tensor to be normalized.
        super(ModelNew, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, input_0):
        primals_1 = self.ln.weight
        primals_2 = self.ln.bias
        primals_3 = input_0
        output = call_512_512([primals_1, primals_2, primals_3])
        return output[0]


#validation:
A = torch.randn((512, 512), device='cuda', dtype=torch.bfloat16).contiguous()
layer_norm_eager = nn.LayerNorm(512).to(device='cuda', dtype=torch.bfloat16)
output_triton, _,_,_ = call_512_512([layer_norm_eager.weight, layer_norm_eager.bias, A])
output_eager = layer_norm_eager(A)

torch.testing.assert_close(output_triton, output_eager, rtol=1.6e-2, atol=1e-3)
assert torch.allclose(output_triton, output_eager, rtol=1.6e-2, atol=1e-3)
print("Test passed!")
"""


import torch
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_per_fused_native_layer_norm(in_ptr0, in_ptr1, in_ptr2,
    out_ptr2, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    RBLOCK: tl.constexpr = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 8192 * x0), xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + r1, None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + r1, None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 8192, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 8192.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.store(out_ptr2 + (r1 + 8192 * x0), tmp27, xmask)


def call_1024_8192(args):
    arg1_1, arg2_1, arg0_1 = args
    args.clear()
    assert_size_stride(arg1_1, (8192,), (1,))
    assert_size_stride(arg2_1, (8192,), (1,))
    assert_size_stride(arg0_1, (1024, 8192), (8192, 1))

    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((1024, 8192), (8192, 1), torch.bfloat16)
        get_raw_stream(0)
        grid = lambda META: (triton.cdiv(1024 + 8 - 1, 8),)
        triton_per_fused_native_layer_norm[grid](arg0_1, arg1_1,
            arg2_1, buf3, 1024, 8192, XBLOCK=8, num_warps=8, num_stages=1)
        del arg0_1
        del arg1_1
        del arg2_1
    return buf3


class ModelNew(nn.Module):
    """
    Simple model that performs Layer Normalization with learnable weight and bias.
    """
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape, dtype=torch.bfloat16
            )
    
    def forward(self, input_0, input_1, input_2):
        arg1_1 = self.ln.weight
        arg2_1 = self.ln.bias
        arg0_1 = input_0
        output = call([arg0_1, arg1_1, arg0_1])
        return output[0]

# #I did switch the signature to match call_512_512
# A = torch.randn((1024, 8192), device='cuda', dtype=torch.bfloat16).contiguous()
# layer_norm_eager = nn.LayerNorm(8192).to(device='cuda', dtype=torch.bfloat16)
# output_triton= call_1024_8192([layer_norm_eager.weight, layer_norm_eager.bias,A])
# output_eager = layer_norm_eager(A)

# torch.testing.assert_close(output_triton, output_eager, rtol=1.6e-2, atol=1e-3)
# assert torch.allclose(output_triton, output_eager, rtol=1.6e-2, atol=1e-3)
# print("Test passed!")