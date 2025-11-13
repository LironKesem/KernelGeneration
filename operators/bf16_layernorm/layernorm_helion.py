from __future__ import annotations
from typing import Any, Callable
import torch
import helion
import helion.language as hl


@helion.kernel
def layer_norm_fwd(
    x: torch.Tensor,
    normalized_shape: list[int],
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = x.size()
    assert len(normalized_shape) == 1, ("1D layer norm only")
    assert normalized_shape[0] == n, ("normalized shape mismatch")

    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    mean = torch.empty([m], dtype=torch.float32, device=x.device)
    rstd = torch.empty([m], dtype=torch.float32, device=x.device)

    for tile_m in hl.tile(m):
        acc = x[tile_m, :].to(torch.float32)

        mean_val = torch.sum(acc, dim=-1) / n

        centered = acc - mean_val[:, None]
        var_val = torch.sum(centered * centered, dim=-1) / n

        rstd_val = torch.rsqrt(var_val + eps)

        normalized = centered * rstd_val[:, None]

        acc = normalized
        out[tile_m, :] = acc.to(x.dtype)
        mean[tile_m] = mean_val
        rstd[tile_m] = rstd_val

    return out, mean, rstd


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        normalized_shape: list[int],
        eps: float,
    ) -> torch.Tensor:
        y = layer_norm_fwd(x, normalized_shape, eps)[0]
        ctx.normalized_shape = normalized_shape
        return y


def layer_norm(
    x: torch.Tensor, normalized_shape: list[int], eps: float = 1e-5
) -> torch.Tensor:
    return LayerNormFunction.apply(x, normalized_shape, eps)


def layer_norm_tritonbench(
    tb_op: object,
    x: torch.Tensor,
    normalized_shape: list[int],
    eps: float = 1e-5,
) -> Callable[[], torch.Tensor]:
    return lambda: layer_norm(x, normalized_shape, eps)
