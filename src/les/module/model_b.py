import torch
import torch.nn as nn
from typing import Optional, Tuple

from les.module.frontier import (segment_logsumexp, segment_onehot, segment_softmax)

__all__ = ['ModelB']


class ModelB(nn.Module):
    """
    Frontier allocation where the cost of each candidate is split into a learned local
    part and the electrostatics the existing Ewald already knows how to compute:

        epsilon_i = a_i + U(z + d^(i)) - U(z)
                  = a_i + k * (K z)_i + 1/2 k^2 K_ii

    d^(i) puts the whole carrier on atom i, so with U(z) = 1/2 z^T K z the difference is
    exact rather than approximated, and one matrix-vector product covers every candidate.
    a_i is whatever local energy change is *not* electrostatic -- bonding, short-range
    relaxation. It is not the full addition energy and not an orbital eigenvalue.

    The energy is the smooth minimum over candidates, so the model pays the cost of the
    site it actually picks:

        G = -logsumexp_structure(-beta * epsilon) / beta
        E = U(z_ref) + m * G

    U(z_report) is deliberately absent. Each candidate's electrostatic cost is already
    inside epsilon, and the energy of the averaged charge is not the average of the
    candidate energies; adding both double counts.

    K must come from compute_A_matrix(self_interaction = not remove_self_interaction),
    which is the matrix whose quadratic form is this module's own monopole energy.

    With constrain_charge the reference charges are corrected before the carrier is
    placed. The correction weight and the occupation are then two different things: the
    weight normalises charges the model got wrong, the occupation places a real carrier.
    Defining both by the corrected charges would be an implicit equation, so they are
    computed in a fixed order -- weights from the raw charges, occupation from the
    corrected ones -- which keeps the total charge exact without a solver. This is a
    choice of this implementation, not a self-consistent stationary point.
    """

    def __init__(self,
                 beta: float = 1.0,
                 epsilon_e: float = 1.0,
                 constrain_charge: bool = False,
                 carrier_sign: float = -1.0,
                 ):
        super().__init__()
        if epsilon_e <= 0.0:
            raise ValueError(f"epsilon_e must be positive, got {epsilon_e}")
        self.beta = float(beta)
        self.scale = float(epsilon_e) ** 0.5
        self.constrain_charge = bool(constrain_charge)
        self.carrier_sign = float(carrier_sign)

    def candidate_costs(self, a: torch.Tensor, K: torch.Tensor,
                        z: torch.Tensor) -> torch.Tensor:
        """epsilon_i for every atom as the carrier's host, [n_atoms]."""
        k = self.carrier_sign / self.scale
        return (a + k * torch.matmul(K, z)
                + (0.5 * k * k) * torch.diagonal(K))

    def forward(self,
                z_raw: torch.Tensor,                              # [n_atoms]
                a: torch.Tensor,                                  # [n_atoms]
                K: torch.Tensor,                                  # [n_atoms, n_atoms]
                batch: torch.Tensor,                              # [n_atoms]
                carrier_count: torch.Tensor,                      # [n_batch]
                n_batch: int,
                reference_total_charge: Optional[torch.Tensor] = None,  # [n_batch]
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (z_report, z_ref, p, G); charges in LES units, G per structure."""
        dtype = z_raw.dtype
        onehot = segment_onehot(batch, n_batch, dtype)
        onehot_t = onehot.transpose(0, 1)
        z_raw = z_raw.reshape(-1)
        a = a.reshape(-1).to(dtype=dtype)

        if self.constrain_charge:
            if reference_total_charge is None:
                raise ValueError("reference_total_charge is required when "
                                 "frontier_constrain_charge is set; neutrality is not "
                                 "assumed on your behalf")
            weights = segment_softmax(
                -self.beta * self.candidate_costs(a, K, z_raw), onehot)
            q_ref = reference_total_charge.reshape(-1).to(dtype=dtype)
            deficit = q_ref / self.scale - torch.matmul(onehot, z_raw)   # [n_batch]
            z_ref = z_raw + weights * torch.matmul(onehot_t, deficit)
        else:
            z_ref = z_raw

        scores = -self.beta * self.candidate_costs(a, K, z_ref)
        p = segment_softmax(scores, onehot)
        smooth_min = -segment_logsumexp(scores, onehot) / self.beta       # [n_batch]

        carrier = carrier_count.reshape(-1).to(dtype=dtype)
        k = self.carrier_sign / self.scale
        z_report = z_ref + (k * torch.matmul(onehot_t, carrier)) * p
        return z_report, z_ref, p, smooth_min
