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
    """
    Computes 1D layer normalization (without affine transformation) for the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (m, n).
        normalized_shape (list[int]): List containing a single integer, the size of the normalized dimension (must match n).
        eps (float, optional): Small value added to variance for numerical stability. Default is 1e-5.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - out (torch.Tensor): Normalized output tensor of shape (m, n), same dtype as input.
            - mean (torch.Tensor): Mean of each row, shape (m,), dtype float32.
            - rstd (torch.Tensor): Reciprocal of standard deviation of each row, shape (m,), dtype float32.

    Constraints:
        - Only supports 1D layer normalization (len(normalized_shape) == 1).
        - Does not include affine transformation (no learnable weight or bias).
    """
    m, n = x.size()
    assert len(normalized_shape) == 1, "1D layer norm only"
    assert normalized_shape[0] == n, "normalized shape mismatch"

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
    """
    Forward-only implementation of LayerNorm.

    This class does not support backward pass (no gradient computation).
    Affine transformation (weight and bias) is not supported.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        normalized_shape: list[int],
        eps: float,
    ) -> torch.Tensor:
        y = layer_norm_fwd(x, normalized_shape, eps)[0]
        return y


def layer_norm(
    x: torch.Tensor, normalized_shape: list[int], eps: float = 1e-5
) -> torch.Tensor:
    """
    Applies forward-only layer normalization to the input tensor without affine parameters.

    This function normalizes the last dimension of the input tensor using the mean and variance,
    but does not apply any learnable scale or bias (affine transformation).

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, normalized_shape[0]).
        normalized_shape (list[int]): List containing the size of the normalized dimension (must be 1D).
        eps (float, optional): A value added to the denominator for numerical stability. Default: 1e-5.

    Returns:
        torch.Tensor: The normalized tensor with the same shape and dtype as the input.
    """
    return LayerNormFunction.apply(x, normalized_shape, eps)


def layer_norm_tritonbench(
    tb_op: object,
    x: torch.Tensor,
    normalized_shape: list[int],
    eps: float = 1e-5,
) -> Callable[[], torch.Tensor]:
    return lambda: layer_norm(x, normalized_shape, eps)
