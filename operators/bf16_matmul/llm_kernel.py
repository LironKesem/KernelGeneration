import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"



def get_autotune_config_XBLOCK():
    if is_cuda():
        print("Using CUDA autotune config")
        return [
        triton.Config({'XBLOCK': 128}, num_stages=3, num_warps=8),
        triton.Config({'XBLOCK': 64},num_stages=4, num_warps=4),
        triton.Config({'XBLOCK': 128}, num_stages=4,num_warps=4),
        triton.Config({'XBLOCK': 64}, num_stages=5, num_warps=2),
        triton.Config({'XBLOCK': 32}, num_stages=5, num_warps=2),
         ]
    else:
        print("Using HIP autotune config")
        sizes = [
            {'XBLOCK': 32},
            {'XBLOCK': 64},
            {'XBLOCK': 128},
            {'XBLOCK': 256},
        ]
        return [triton.Config(s | {'matrix_instr_nonkdim': 16}, num_warps=8, num_stages=2) for s in sizes]

@triton.autotune(
    configs=get_autotune_config_XBLOCK(),
    key=['XBLOCK'],
)
@triton.jit
def triton_poi_fused__to_copy(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + x0, tmp1, None)

    
def call_64_128_32(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 128), (128, 1))
    assert_size_stride(arg1_1, (128, 32), (32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda(
            (128, 32), (32, 1), torch.bfloat16
        )
        get_raw_stream(0)
        grid0 = lambda META: (triton.cdiv(8192 + META['XBLOCK'] - 1, META['XBLOCK']))
        triton_poi_fused__to_copy[grid0](arg1_1, buf0, 8192)
        del arg1_1
        buf1 = empty_strided_cuda((64, 32), (32, 1), torch.bfloat16)
        extern_kernels.mm(arg0_1, buf0, out=buf1)
        del arg0_1
        del buf0
    return buf1




def get_autotune_config_XBLOCK0_XBLOCK1():
    if is_cuda():
        print("Using CUDA autotune config")
        return [
        triton.Config({'XBLOCK0': 128,'XBLOCK1': 64}, num_stages=3, num_warps=8),
        triton.Config({'XBLOCK0': 64, 'XBLOCK1': 32},num_stages=4, num_warps=4),
        triton.Config({'XBLOCK0': 256, 'XBLOCK1': 128}, num_stages=4,num_warps=4),
        triton.Config({'XBLOCK0': 64, 'XBLOCK1': 64}, num_stages=5, num_warps=2),
        triton.Config({'XBLOCK0': 32, 'XBLOCK1': 32}, num_stages=5, num_warps=2),
        ]
        
    else:
        print("Using HIP autotune config")
        sizes = [
        {'XBLOCK0': 32, 'XBLOCK1': 32},
        {'XBLOCK0': 64, 'XBLOCK1': 64},
        {'XBLOCK0': 128, 'XBLOCK1': 128},
        {'XBLOCK0': 256, 'XBLOCK1': 256},
        ]
        return [triton.Config(s | {'matrix_instr_nonkdim': 16}, num_warps=8, num_stages=2) for s in sizes]

@triton.autotune(
    configs=[triton.Config({'XBLOCK0': c.kwargs['XBLOCK0']}, num_stages=c.num_stages, num_warps=c.num_warps) 
             for c in get_autotune_config_XBLOCK0_XBLOCK1()],
     key=['XBLOCK0'],
)
@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK0: tl.constexpr
    ):
    xoffset = tl.program_id(0) * XBLOCK0
    xindex = xoffset + tl.arange(0, XBLOCK0)[:]
    tl.full([XBLOCK0], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + x0, tmp1, None)



@triton.autotune(
    configs=[triton.Config({'XBLOCK1': c.kwargs['XBLOCK1']}, num_stages=c.num_stages, num_warps=c.num_warps) 
             for c in get_autotune_config_XBLOCK0_XBLOCK1()],
     key=['XBLOCK1'],
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK1: tl.constexpr
    ):
    xoffset = tl.program_id(0) * XBLOCK1
    xindex = xoffset + tl.arange(0, XBLOCK1)[:]
    tl.full([XBLOCK1], True, tl.int1)
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
        grid0 = lambda META: (triton.cdiv(num_elements0 + META['XBLOCK0'] - 1, META['XBLOCK0']), )

        triton_poi_fused__to_copy_0[grid0](arg0_1, buf0, num_elements0)
        del arg0_1
        num_elements1 = 8192
        grid1 = lambda META: (triton.cdiv(num_elements1 + META['XBLOCK1'] - 1, META['XBLOCK1']), )

        buf1 = empty_strided_cuda((256, 32), (32, 1), torch.bfloat16)
        triton_poi_fused__to_copy_1[grid1](
            arg1_1, buf1, num_elements1
        )
        del arg1_1
        buf2 = empty_strided_cuda(
            (256, 32), (32, 1), torch.bfloat16
        )  
        extern_kernels.mm(buf0, buf1, out=buf2)
        del buf0
        del buf1
    return buf2