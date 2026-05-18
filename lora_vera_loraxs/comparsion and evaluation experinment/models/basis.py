from __future__ import annotations

from typing import Literal

import torch


def svd_basis(
    weight: torch.Tensor,
    rank: int,
    basis: Literal["svd_top", "svd_bottom"],
    include_sigma: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return left and right factors for a frozen SVD basis.

    The wrapped nn.Linear weight has shape (out_features, in_features). The
    returned factors have shapes (out_features, rank) and (rank, in_features).
    If include_sigma=True, singular values are folded into the left factor.
    """

    matrix = weight.detach().float()
    u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    if basis == "svd_top":
        index = torch.arange(rank, device=matrix.device)
    elif basis == "svd_bottom":
        index = torch.arange(singular_values.numel() - rank, singular_values.numel(), device=matrix.device)
    else:
        raise ValueError(f"unsupported SVD basis {basis}")

    left = u[:, index]
    if include_sigma:
        left = left * singular_values[index].unsqueeze(0)
    right = vh[index, :]
    return left.to(weight.dtype), right.to(weight.dtype)


def random_orthogonal_basis(
    out_features: int,
    in_features: int,
    rank: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    left_raw = torch.randn(out_features, rank, generator=generator)
    right_raw = torch.randn(in_features, rank, generator=generator)
    left, _ = torch.linalg.qr(left_raw, mode="reduced")
    right, _ = torch.linalg.qr(right_raw, mode="reduced")
    return left.to(device=device, dtype=dtype), right.T.to(device=device, dtype=dtype)
