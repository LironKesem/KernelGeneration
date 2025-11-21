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
    xnumel, rnumel, XBLOCK: tl.constexpr = 1, RBLOCK: tl.constexpr = 512):

    xoffset = tl.program_id(0) * XBLOCK
    x0 = tl.full([1], xoffset, tl.int32)
    r1 = tl.arange(0, RBLOCK)
    x_tile = tl.load(in_ptr0 + (r1 + RBLOCK * x0), None).to(tl.float32)
    tmp1 = tl.broadcast_to(x_tile, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp1.to(tl.float32), 0))
    tmp6 = tl.full([1], RBLOCK, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = float(RBLOCK)
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + x0, tmp18, None)
    tl.store(out_ptr0 + x0, tmp8, None)


@triton.jit
def triton_poi_fused_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr, D:tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // D
    x0 = xindex % D
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


def call(args):
    weight, bias, x = args
    args.clear()
    N, D = x.shape
    assert_size_stride(weight, (D,), (1,))
    assert_size_stride(bias, (D,), (1,))
    assert_size_stride(x, (N, D), (D, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((N, 1), (1, 1), torch.bfloat16)
        buf1 = empty_strided_cuda((N, 1), (1, D), torch.bfloat16)
        buf3 = reinterpret_tensor(buf1, (N, 1), (1, D), 0)
        del buf1
        get_raw_stream(0)
        grid0 = lambda META: (triton.cdiv(N, 1),)
        triton_per_fused_native_layer_norm_0[grid0](buf3, x,
            buf0, N, D, XBLOCK=1,RBLOCK=D)
        buf4 = empty_strided_cuda((N, D), (D, 1), torch.bfloat16)
        grid1 = lambda META: (triton.cdiv(x.numel(), N),)

        triton_poi_fused_native_layer_norm_1[grid1](x, buf0,
            buf3, weight, bias, buf4, x.numel(),XBLOCK=N, D=D)
        del weight
        del bias
    return buf4, x, buf0, buf3

# originally designed for (512, 512)
class KLayerNorm(nn.Module):
    def __init__(self):

        super(KLayerNorm, self).__init__()

    def forward(self, input_0,weight,bias):
        weight = weight
        bias = bias
        x = input_0
        output = call([weight, bias, x])
        return output[0]
