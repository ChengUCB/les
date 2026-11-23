import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union

__all__ = ['Qeq']


class Qeq(nn.Module):
    """
    Charge equilibration of n atoms: minimise

        U(q) = 1/2 q^T A q + sum_i (b_i q_i + 1/2 J_i q_i^2),   sum_i q_i = Q in every structure

    A: charge-charge coupling of the electrostatics, with the Gaussian self energy on its diagonal
    b: electronegativity chi plus the potential of everything that is not equilibrated
    J: hardness, >= 0

    The stationary point solves [[A + J, C], [C^T, 0]] [q; lambda] = [-b; Q], with C_is = 1 if atom i
    is in structure s. lambda is the chemical potential of the equilibrated atoms of each structure.

    J = 0 with constant chi is the Siepmann-Sprik electrode (MetalWall).

    Hardness: one value for all atoms (hardness_trainable=False), or one trainable value per
    element starting from it; latent_hardness adds an atom-wise part, softplus(latent_hardness).

    Electronegativity: chi is one number for all elements or {atomic number: value} (0 for the others),
    fixed, or with chi_trainable one trainable value per element starting from it; latent_chi adds an
    atom-wise part.
    """
    __constants__ = ['hardness_trainable']

    def __init__(self, hardness: float = 1.0, hardness_trainable: bool = True,
                 chi: Union[None, float, Dict[int, float]] = None, chi_trainable: bool = False):
        super().__init__()
        chi_elem = torch.zeros(119)  # indexed by atomic number
        if isinstance(chi, dict):
            for z, value in chi.items():
                chi_elem[int(z)] = float(value)
        elif chi is not None:
            chi_elem += float(chi)
        if chi_trainable:
            self.chi_elem = nn.Parameter(chi_elem)
        else:
            self.register_buffer('chi_elem', chi_elem)
        self.hardness_trainable = hardness_trainable
        if hardness_trainable:
            if hardness <= 0.0:
                raise ValueError("a trainable hardness must start from a positive value")
            # softplus^{-1}(hardness), indexed by atomic number
            self.J_raw = nn.Parameter(torch.full((119,), hardness + math.log(-math.expm1(-hardness))))
        else:
            if hardness < 0.0:
                raise ValueError("the hardness must not be negative")
            self.register_buffer('J_const', torch.tensor(float(hardness)))

    def hardness(self,
                 atomic_numbers: torch.Tensor,  # [n]
                 latent_hardness: Optional[torch.Tensor] = None,  # [n]
                 ) -> torch.Tensor:
        if self.hardness_trainable:
            J = F.softplus(self.J_raw)[atomic_numbers]
        else:
            J = self.J_const.expand(atomic_numbers.shape[0])
        if latent_hardness is not None:
            J = J + F.softplus(latent_hardness.reshape(-1))
        return J

    def electronegativity(self,
                          atomic_numbers: torch.Tensor,  # [n]
                          latent_chi: Optional[torch.Tensor] = None,  # [n]
                          ) -> torch.Tensor:
        chi = self.chi_elem[atomic_numbers]
        if latent_chi is not None:
            chi = chi + latent_chi.reshape(-1)
        return chi

    def forward(self,
                A: torch.Tensor,  # [n, n]
                b: torch.Tensor,  # [n]
                J: torch.Tensor,  # [n]
                batch: torch.Tensor,  # [n], structure of each atom
                total_charge: torch.Tensor,  # [batch_size]
                variational: bool = False,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the charges, lambda and the inverse of the bordered matrix (detached).

        variational=True, when the energy that is differentiated is U itself: the gradient goes through
        q = q0 + K^-1 (rhs - K q0) with K^-1 and q0 detached, i.e. through matrix products only. Forces
        and their derivatives with respect to the weights are those of the direct solve, because what is
        left out is multiplied by dU/dq, which is the same on all equilibrated atoms.
        """
        n = A.shape[0]
        dtype = A.dtype
        C, empty = self.constraint(batch, total_charge.shape[0], dtype)
        K = torch.cat([torch.cat([A + torch.diag(J.to(dtype)), C], dim=1),
                       torch.cat([C.transpose(0, 1), torch.diag(empty)], dim=1)], dim=0)
        rhs = torch.cat([-b.to(dtype), total_charge.to(dtype) * (1.0 - empty)])
        if variational:
            K_inv = torch.linalg.inv(K.detach())
            x0 = K_inv @ rhs.detach()
            sol = x0 + K_inv @ (rhs - K @ x0)
        else:
            K_inv = torch.zeros(0, 0, dtype=dtype, device=A.device)
            sol = torch.linalg.solve(K, rhs)
        return sol[:n], sol[n:], K_inv

    def reuse(self,
              K_inv: torch.Tensor,  # [n + batch_size, n + batch_size], from forward with the same A, J, batch
              b: torch.Tensor,  # [n]
              batch: torch.Tensor,  # [n]
              total_charge: torch.Tensor,  # [batch_size]
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = b.shape[0]
        dtype = K_inv.dtype
        C, empty = self.constraint(batch, total_charge.shape[0], dtype)
        sol = K_inv @ torch.cat([-b.to(dtype), total_charge.to(dtype) * (1.0 - empty)])
        return sol[:n], sol[n:]

    def constraint(self, batch: torch.Tensor, n_batch: int, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        C = (batch.unsqueeze(1) == torch.arange(n_batch, device=batch.device).unsqueeze(0)).to(dtype)  # [n, batch_size]
        # a structure without equilibrated atoms has an empty constraint: lambda = 0 instead
        empty = (C.sum(dim=0) < 0.5).to(dtype)
        return C, empty
