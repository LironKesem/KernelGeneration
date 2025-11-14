from typing import Callable
import torch
import torch.nn.functional as F
import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def gemm_gelu(x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # Shapes
    m, k = x.shape
    k2, n = y.shape
    assert k == k2, f"size mismatch {k} != {k2}"
    assert bias.shape[0] == n, "wrong bias shape"

    # Result dtype follows PyTorch promotion rules (bf16×bf16→bf16; bf16×fp32→fp32)
    out_dtype = torch.promote_types(x.dtype, y.dtype)
    out = torch.empty((m, n), dtype=out_dtype, device=x.device)

    # GPU part
    for tile_m, tile_n in hl.tile([m, n]):
        # Accumulate in fp32 for numerical stability
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            # Use input dtypes; PyTorch addmm will promote as needed into fp32 acc
            a_tile = x[tile_m, tile_k]
            b_tile = y[tile_k, tile_n]
            acc = torch.addmm(acc, a_tile, b_tile)
        # add bias and apply gelu in fp32
        acc = acc + bias[tile_n]
        acc = F.gelu(acc)

        out[tile_m, tile_n] = acc.to(out_dtype)
    return out


def gemm_gelu_tritonbench(
        tb_op, a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor
) -> Callable[[], torch.Tensor]:
    """TritonBench wrapper: MUST return a zero-arg callable for timing."""
    return lambda: gemm_gelu(a, b, bias)