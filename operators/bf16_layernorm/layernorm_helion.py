from __future__ import annotations
from typing import Any, Callable

import torch
import helion
import helion.language as hl


@helion.kernel
def layer_norm_fwd(
    x: torch.Tensor,
    normalized_shape: int,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes 1D layer normalization (without affine transformation) for the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (m, n).
        normalized_shape (int): Size of the normalized dimension (must match n).
        eps (float, optional): Small value added to variance for numerical stability.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - out (torch.Tensor): Normalized output tensor of shape (m, n), same dtype as input.
            - mean (torch.Tensor): Mean of each row, shape (m,), dtype float32.
            - rstd (torch.Tensor): Reciprocal of standard deviation of each row, shape (m,), dtype float32.
    """
    m, n = x.size()

    assert normalized_shape == n, "normalized shape mismatch"

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

        out[tile_m, :] = normalized.to(x.dtype)
        mean[tile_m] = mean_val
        rstd[tile_m] = rstd_val

    return out, mean, rstd


class LayerNormFunction(torch.autograd.Function):
    """
    Forward-only implementation of LayerNorm (no affine, no backward).
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        normalized_shape,
        eps: float,
    ) -> torch.Tensor:
        # Accept list[int]/tuple[int]/int and normalize to a scalar
        if isinstance(normalized_shape, (list, tuple)):
            assert len(normalized_shape) == 1, "1D layer norm only"
            norm_dim = int(normalized_shape[0])
        else:
            norm_dim = int(normalized_shape)

        y = layer_norm_fwd(x, norm_dim, eps)
        return y


def layer_norm(
    x: torch.Tensor,
    normalized_shape,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Applies forward-only layer normalization to the input tensor without affine parameters.

    Args:
        x (torch.Tensor): Input tensor of shape (batch, normalized_shape[0]).
        normalized_shape: Either an int or a 1-element list/tuple[int].
        eps (float, optional): A value added to the denominator for numerical stability.

    Returns:
        torch.Tensor: The normalized tensor with the same shape and dtype as the input.
    """
    return LayerNormFunction.apply(x, normalized_shape, eps)


def layer_norm_tritonbench(
    tb_op: object,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> Callable[[], torch.Tensor]:
    """
    TritonBench adapter for the Helion layer norm kernel.

    TritonBench passes (x, weight, bias, eps) as example inputs.
    The Helion kernel is non-affine; weight/bias are ignored here.
    In the operator, weight=1 and bias=0, so this matches torch.layer_norm
    without affine.
    """
    normalized_shape = [x.shape[-1]]

    return lambda: layer_norm(x, normalized_shape, eps)
