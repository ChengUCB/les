import torch
import torch.nn as nn
from typing import Optional

from ..util import grad

__all__ = ['AlphaDerivative']

class AlphaDerivative(nn.Module):
    """
    Computes d(total_alpha_ij)/d(r_{n,k}) — the position derivative of the
    summed polarizability tensor, analogous to Born Effective Charges.

    Output convention (mirrors BEC): output indices first, position index last.
      - anisotropic alpha [n_atoms, 3, 3] → [n_atoms, 3, 3, 3]  [n, i, j, k]
      - isotropic   alpha [n_atoms]       → [n_atoms, 3]         [n, k]

    Velocity contraction (anisotropic):
        dalpha_rate = einsum('nijk,nk->nij', alpha_deriv, velocity)  # [N, 3, 3]
        total_dalpha = dalpha_rate.sum(0)                             # [3, 3]
    """

    def forward(self,
                alpha: torch.Tensor,   # [n_atoms, 3, 3] or [n_atoms]
                r: torch.Tensor,       # [n_atoms, 3], requires autograd path from upstream
                cell: torch.Tensor,
                batch: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:

        n_atoms = r.shape[0]
        anisotropic = (alpha.dim() == 3)

        if batch is None:
            batch = torch.zeros(n_atoms, dtype=torch.int64, device=r.device)
        unique_batches = torch.unique(batch)

        all_alpha = []
        for i in unique_batches.long():
            mask = batch == i
            alpha_now = alpha[mask]
            if anisotropic:
                # [n_atoms_in_batch, 3, 3] → sum → [3, 3] → flatten → [9]
                all_alpha.append(alpha_now.sum(dim=0).reshape(9))
            else:
                # [n_atoms_in_batch] → sum → scalar → [1]
                all_alpha.append(alpha_now.sum().unsqueeze(0))

        # [batch_size, 9] or [batch_size, 1]
        alpha_stacked = torch.stack(all_alpha, dim=0)

        # grad returns [n_atoms, r_k, dim_y]
        # transpose(1,2) → [n_atoms, dim_y, r_k]  (output index first, position last)
        alpha_grad = grad(y=alpha_stacked, x=r).transpose(1, 2).contiguous()

        if anisotropic:
            # [n_atoms, 9, 3] → [n_atoms, 3, 3, 3] with [n, alpha_i, alpha_j, r_k]
            return alpha_grad.reshape(n_atoms, 3, 3, 3)
        else:
            # [n_atoms, 1, 3] → [n_atoms, 3]
            return alpha_grad.squeeze(1)
