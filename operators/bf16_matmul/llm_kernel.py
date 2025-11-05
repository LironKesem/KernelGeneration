import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
#from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + x0, tmp1, None)


def call_256_256_32(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (256, 256), (256, 1))
    assert_size_stride(arg1_1, (256, 32), (32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
        get_raw_stream(0)
        num_elements0 = 65536
        XBLOCK0 = 512
        grid0 = ((num_elements0 + XBLOCK0 - 1) // XBLOCK0,)

        triton_poi_fused__to_copy_0[grid0](
            arg0_1, buf0, 65536, XBLOCK=512, num_warps=4, num_stages=1
        )
        del arg0_1
        num_elements1 = 8192
        XBLOCK1 = 256
        grid1 = ((num_elements1 + XBLOCK1 - 1) // XBLOCK1,)

        buf1 = empty_strided_cuda((256, 32), (32, 1), torch.bfloat16)
        triton_poi_fused__to_copy_0[grid1](
            arg1_1, buf1, 8192, XBLOCK=256, num_warps=4, num_stages=1
        )
        del arg1_1
        buf2 = empty_strided_cuda(
            (256, 32), (32, 1), torch.bfloat16
        )  # bug: it was torch.float32
        extern_kernels.mm(buf0, buf1, out=buf2)
        del buf0
        del buf1
    # return reinterpret_tensor(buf2, (32, 256, 256), (65536, 256, 1), 0) # Bug: it ended with , but that return a shape of [torch.Size([32, 256, 256])]
    return buf2


def call_64_128_32(args):
    arg0_1, arg1_1 = args
    args.clear()
    # I switched the assert for arg0_1 and arg1_1  to match the correct inputs
    assert_size_stride(arg0_1, (64, 128), (128, 1))
    assert_size_stride(arg1_1, (128, 32), (32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda(
            (128, 32), (32, 1), torch.bfloat16
        )  # bug: it was torch.float16
        get_raw_stream(0)
        XBLOCK = 256
        grid0 = ((8192 + XBLOCK - 1) // XBLOCK,)
        triton_poi_fused__to_copy_0[grid0](
            arg1_1, buf0, 8192, XBLOCK=256, num_warps=4, num_stages=1
        )
        del arg1_1
        buf1 = empty_strided_cuda((64, 32), (32, 1), torch.bfloat16)
        extern_kernels.mm(arg0_1, buf0, out=buf1)
        del arg0_1
        del buf0
    return buf1
