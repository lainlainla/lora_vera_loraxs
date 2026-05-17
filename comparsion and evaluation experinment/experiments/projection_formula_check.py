from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.subspace import reconstruction_error_for_subspace_member  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Check when A(A^TGB)B^T is a valid projection.")
    parser.add_argument("--out-features", type=int, default=32)
    parser.add_argument("--in-features", type=int, default=24)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    q_left, _ = torch.linalg.qr(torch.randn(args.out_features, args.rank, generator=generator), mode="reduced")
    q_right, _ = torch.linalg.qr(torch.randn(args.in_features, args.rank, generator=generator), mode="reduced")
    coefficient = torch.randn(args.rank, args.rank, generator=generator)

    orthonormal_error = reconstruction_error_for_subspace_member(
        q_left,
        q_right,
        coefficient,
        assume_orthonormal=True,
    )

    scales_left = torch.linspace(1.0, 3.0, args.rank)
    scales_right = torch.linspace(0.5, 2.0, args.rank)
    nonorth_left = q_left * scales_left.unsqueeze(0)
    nonorth_right = q_right * scales_right.unsqueeze(0)

    naive_nonorth_error = reconstruction_error_for_subspace_member(
        nonorth_left,
        nonorth_right,
        coefficient,
        assume_orthonormal=True,
    )
    corrected_nonorth_error = reconstruction_error_for_subspace_member(
        nonorth_left,
        nonorth_right,
        coefficient,
        assume_orthonormal=False,
    )

    print("Projection identity check for G = A X B^T")
    print(f"orthonormal A/B with naive formula:        relative error = {orthonormal_error:.3e}")
    print(f"non-orthonormal A/B with naive formula:    relative error = {naive_nonorth_error:.3e}")
    print(f"non-orthonormal A/B with Gram correction:  relative error = {corrected_nonorth_error:.3e}")
    print()
    print("Conclusion: A(A^T G B)B^T is an orthogonal projection only when A and B have orthonormal columns.")
    print("For LoRA-XS energy capture, use U,V from SVD or the Gram-corrected projection, not U Sigma as A without correction.")


if __name__ == "__main__":
    main()
