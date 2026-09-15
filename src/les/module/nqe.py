from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Nqe']


class Nqe(nn.Module):
    """
    Neural charge equilibration: spread the charge a structure is missing over its atoms
    with learned Fukui-like weights,

        q_i = q~_i + f_i / (sum_j f_j) * (Q_tot - sum_j q~_j)

    which is Eq. 6 of Anstine, Zubatyuk & Isayev (AIMNet2) and Eqs. 13-14 of Batatia et
    al. (MACE-POLAR), where f_i is identified with the condensed Fukui function
    dq_i/dQ, i.e. how readily atom i accepts charge.

    O(n_atoms), no linear solve, and it enforces the total charge exactly. It is the
    constraint half of a charge equilibration without the global coupling through A^-1 --
    for that, use Qeq.

    Uniform weights with Q_tot = 0 give q_i - mean(q), so this is a strict generalization
    of the `remove_mean` option.

    A mask restricts which atoms absorb the difference; the rest keep their incoming
    charge exactly. masked_charge_scope then selects what Q_tot refers to: False (the
    default) the whole structure, True the masked atoms alone.

    f is passed in raw and mapped through softplus: the weights must be positive for the
    normalization to be meaningful, and the shares must be f/sum(f). A softmax would give
    exp(f)/sum(exp(f)) instead, which is normalized but no longer proportional to f, and
    so is not a Fukui redistribution.
    """

    def __init__(self, eps: float = 1e-6, masked_charge_scope: bool = False):
        super().__init__()
        self.eps = eps
        self.masked_charge_scope = masked_charge_scope

    def forward(self,
                q: torch.Tensor,                    # [n_atoms]
                f_raw: torch.Tensor,                # [n_atoms]
                batch: torch.Tensor,                # [n_atoms]
                total_charge: torch.Tensor,         # [batch_size]
                mask: Optional[torch.Tensor] = None,  # [n_atoms] bool, True = absorbs
                ) -> torch.Tensor:
        device = q.device
        dtype = q.dtype
        B = total_charge.shape[0]

        q = q.reshape(-1)
        f = F.softplus(f_raw.reshape(-1).to(dtype=dtype)) + self.eps           # [n_atoms]
        total_charge = total_charge.reshape(-1).to(dtype=dtype)
        if mask is None:
            m = torch.ones_like(q)
        else:
            m = mask.to(dtype=dtype)
        # zero weight outside the mask, so those atoms keep their charge untouched
        f = f * m                                                              # [n_atoms]

        onehot = (batch.unsqueeze(0)
                  == torch.arange(B, device=device).unsqueeze(1)).to(dtype)    # [B, n_atoms]
        onehot_t = onehot.transpose(0, 1)                                      # [n_atoms, B]

        # a structure with nothing to absorb has f summing to zero; clamping the
        # denominator leaves every share at zero, so its charges pass through
        f_sum = torch.matmul(onehot, f).clamp(min=self.eps if self.eps > 0.0 else 1e-30)
        share = f / torch.matmul(onehot_t, f_sum)                              # [n_atoms]

        # what the deficit is measured against: the whole structure, or the masked part
        counted = q * m if self.masked_charge_scope else q
        deficit = torch.matmul(onehot_t, total_charge - torch.matmul(onehot, counted))
        return q + share * deficit
