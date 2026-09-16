import torch
import torch.nn as nn
from typing import Tuple

from les.module.frontier import segment_onehot, segment_softmax

__all__ = ['Model0A']


class Model0A(nn.Module):
    """
    Model 0-a: the two-head extension of Model 0.

    Model 0 spreads every charge change with one distribution. Here the host predicts
    two allocation scores per atom, and adding an electron draws on a different
    distribution from removing one:

        p_add = softmax_structure(-beta * epsilon_add)
        p_rem = softmax_structure(-beta * epsilon_rem)

    Both heads take the same minus sign, so a low score means a large weight in either.
    They are latent allocation scores, not electron affinities or ionization energies:
    only differences within a structure matter, so energy and force training leaves
    each head's absolute scale undetermined.

    The reference total charge is always enforced. Its deficit is measured in physical
    charge units,

        D = Q_ref - s * sum_i z_raw_i,        s = sqrt(epsilon_e)

    and its sign says which way the charge has to move: D < 0 needs electrons added,
    D > 0 needs them removed. Rather than branching on that sign -- which would leave
    the force discontinuous wherever geometry carries D through zero, since the two
    distributions differ -- the two are blended by a smooth gate,

        g(D)  = (1 + tanh(D / delta)) / 2          the removal share
        p_cor = (1 - g) * p_add + g * p_rem
        z_ref = z_raw + (D / s) * p_cor

    Both distributions are normalized, so p_cor is too and s * sum(z_ref) = Q_ref
    exactly, whatever the gate does. At D = 0 the gate sits at 50:50 but the
    correction is identically zero, atom by atom.

    The carrier is then placed with the head matching its direction, and the gate is
    not reapplied:

        electron (carrier_sign -1):  z_final = z_ref - (m / s) * p_add
        hole     (carrier_sign +1):  z_final = z_ref + (m / s) * p_rem

    so the final total is Q_ref - m or Q_ref + m. Q_ref is the reference total before
    the carrier, not the final one.
    """

    def __init__(self,
                 beta: float = 1.0,
                 epsilon_e: float = 1.0,
                 carrier_sign: float = -1.0,
                 charge_switch_width: float = 0.01,
                 ):
        super().__init__()
        if beta <= 0.0:
            raise ValueError(f"beta must be positive, got {beta}")
        if epsilon_e <= 0.0:
            raise ValueError(f"epsilon_e must be positive, got {epsilon_e}")
        if charge_switch_width <= 0.0:
            raise ValueError("charge_switch_width must be positive, got "
                             f"{charge_switch_width}")
        if float(carrier_sign) not in (-1.0, 1.0):
            raise ValueError(f"carrier_sign must be -1 or +1, got {carrier_sign}")
        self.beta = float(beta)
        self.scale = float(epsilon_e) ** 0.5
        self.carrier_sign = float(carrier_sign)
        self.switch_width = float(charge_switch_width)

    def forward(self,
                z_raw: torch.Tensor,                  # [n_atoms]
                frontier: torch.Tensor,               # [n_atoms, 2]: add, remove
                batch: torch.Tensor,                  # [n_atoms]
                carrier_count: torch.Tensor,          # [n_batch]
                n_batch: int,
                reference_total_charge: torch.Tensor,  # [n_batch], physical
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                           torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (z_final, z_ref, p_selected, p_add, p_rem, deficit, gate).

        Charges are in LES units; the deficit is physical; occupations and the gate
        are dimensionless.
        """
        dtype = z_raw.dtype
        if frontier.dim() != 2 or frontier.shape[1] != 2:
            raise ValueError("latent_frontier must be [n_atoms, 2] for use_model0_a: "
                             "column 0 is the addition score, column 1 the removal one")
        onehot = segment_onehot(batch, n_batch, dtype)
        onehot_t = onehot.transpose(0, 1)
        z_raw = z_raw.reshape(-1)
        frontier = frontier.to(dtype=dtype)

        p_add = segment_softmax(-self.beta * frontier[:, 0], onehot)
        p_rem = segment_softmax(-self.beta * frontier[:, 1], onehot)

        # the deficit is a ratio of physical charges, so the gate width keeps its
        # meaning in e when epsilon_e changes
        q_ref = reference_total_charge.reshape(-1).to(dtype=dtype)
        deficit = q_ref - self.scale * torch.matmul(onehot, z_raw)        # [n_batch]
        gate = 0.5 * (1.0 + torch.tanh(deficit / self.switch_width))      # [n_batch]

        gate_at = torch.matmul(onehot_t, gate)
        p_cor = (1.0 - gate_at) * p_add + gate_at * p_rem
        z_ref = z_raw + torch.matmul(onehot_t, deficit / self.scale) * p_cor

        carrier = carrier_count.reshape(-1).to(dtype=dtype)
        p_sel = p_add if self.carrier_sign < 0.0 else p_rem
        k = self.carrier_sign / self.scale
        z_final = z_ref + (k * torch.matmul(onehot_t, carrier)) * p_sel
        return z_final, z_ref, p_sel, p_add, p_rem, deficit, gate
