from __future__ import annotations

import torch


def project_matrix(
    gradient: torch.Tensor,
    left_basis: torch.Tensor,
    right_basis: torch.Tensor,
    *,
    assume_orthonormal: bool = False,
) -> torch.Tensor:
    """Project gradient onto {A X B^T}.

    left_basis is A with shape (out, r), right_basis is B with shape (in, r).
    If A and B have orthonormal columns, the projection is
    A(A^T G B)B^T. Otherwise the Gram-corrected orthogonal projection is
    A(A^T A)^+ A^T G B(B^T B)^+ B^T.
    """

    a = left_basis
    b = right_basis
    if assume_orthonormal:
        return a @ (a.T @ gradient @ b) @ b.T
    gram_a_inv = torch.linalg.pinv(a.T @ a)
    gram_b_inv = torch.linalg.pinv(b.T @ b)
    return a @ gram_a_inv @ a.T @ gradient @ b @ gram_b_inv @ b.T


def projection_energy(
    gradient: torch.Tensor,
    left_basis: torch.Tensor,
    right_basis: torch.Tensor,
    *,
    assume_orthonormal: bool = False,
    eps: float = 1e-12,
) -> float:
    projected = project_matrix(
        gradient,
        left_basis,
        right_basis,
        assume_orthonormal=assume_orthonormal,
    )
    return float(projected.pow(2).sum().div(gradient.pow(2).sum().clamp_min(eps)).item())


def svd_orthonormal_bases(weight: torch.Tensor, rank: int, *, bottom: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Return U_r and V_r with orthonormal columns for energy-capture checks."""

    matrix = weight.detach().float()
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    if bottom:
        index = torch.arange(u.shape[1] - rank, u.shape[1], device=matrix.device)
    else:
        index = torch.arange(rank, device=matrix.device)
    return u[:, index], vh[index, :].T


def reconstruction_error_for_subspace_member(
    left_basis: torch.Tensor,
    right_basis: torch.Tensor,
    coefficient: torch.Tensor,
    *,
    assume_orthonormal: bool,
) -> float:
    """Return ||P(A X B^T) - A X B^T||_F / ||A X B^T||_F."""

    member = left_basis @ coefficient @ right_basis.T
    projected = project_matrix(member, left_basis, right_basis, assume_orthonormal=assume_orthonormal)
    denominator = member.norm().clamp_min(1e-12)
    return float((projected - member).norm().div(denominator).item())
