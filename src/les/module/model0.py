import torch
import torch.nn as nn
from typing import Optional, Tuple

from les.module.frontier import segment_onehot, segment_softmax

__all__ = ['Model0']


class Model0(nn.Module):
    """
    Frontier-level charge allocation: put the carrier where adding it costs least.

        p_i = softmax_structure(-beta * epsilon_i)
        z_i = z_ref_i + m * k * p_i

    epsilon comes from the host model; it is a selection score, not an orbital energy
    and not an electron affinity. Only differences within a structure matter -- adding
    a constant to every epsilon of a structure leaves p, the charges and the energy
    untouched -- so its absolute scale is not determined by energy and force training.

    Charges are carried in LES units, z = q_physical / sqrt(epsilon_e), which is the
    convention of docs/theory.md. The carrier is one electron (carrier_sign = -1) or one
    hole (+1), so it enters as k = carrier_sign / sqrt(epsilon_e) per unit of p.

    With constrain_charge the raw charges are first corrected onto the reference total
    charge, spreading the deficit with the same p:

        D_ref  = Q_ref / s - sum_i z_raw_i
        z_ref  = z_raw + p * D_ref

    Without it z_ref is z_raw: the absolute total is then whatever the model produced,
    but the *change* from the carrier is still exactly carrier_sign * m in physical
    charge units.
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

    def forward(self,
                z_raw: torch.Tensor,                              # [n_atoms]
                epsilon: torch.Tensor,                            # [n_atoms]
                batch: torch.Tensor,                              # [n_atoms]
                carrier_count: torch.Tensor,                      # [n_batch]
                n_batch: int,
                reference_total_charge: Optional[torch.Tensor] = None,  # [n_batch]
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (z_final, z_ref, p), all in LES charge units."""
        dtype = z_raw.dtype
        onehot = segment_onehot(batch, n_batch, dtype)
        onehot_t = onehot.transpose(0, 1)
        z_raw = z_raw.reshape(-1)

        p = segment_softmax(-self.beta * epsilon.reshape(-1).to(dtype=dtype), onehot)

        if self.constrain_charge:
            if reference_total_charge is None:
                raise ValueError("reference_total_charge is required when "
                                 "frontier_constrain_charge is set; neutrality is not "
                                 "assumed on your behalf")
            q_ref = reference_total_charge.reshape(-1).to(dtype=dtype)
            deficit = q_ref / self.scale - torch.matmul(onehot, z_raw)   # [n_batch]
            z_ref = z_raw + p * torch.matmul(onehot_t, deficit)
        else:
            z_ref = z_raw

        carrier = carrier_count.reshape(-1).to(dtype=dtype)
        k = self.carrier_sign / self.scale
        z_final = z_ref + (k * torch.matmul(onehot_t, carrier)) * p
        return z_final, z_ref, p
