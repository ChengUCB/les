import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

__all__ = ['Qeq', 'batch_slots']


def batch_slots(batch: torch.Tensor, n_batch: int, n_max: int):
    """
    Where each atom sits inside its own structure, for laying a batch out as
    [n_batch, n_max, ...] instead of one flat [n_atoms, ...] block.

    Returns (slot, valid): slot[i] is atom i's index within its structure, and
    valid[b, s] marks the slots a real atom occupies. n_max is supplied rather than
    measured so every shape stays static and the graph can be compiled or exported;
    an n_max below the largest structure raises an index error rather than quietly
    dropping atoms.

    Atoms need not be ordered by structure.
    """
    device = batch.device
    N = batch.shape[0]
    counts = torch.zeros(n_batch, dtype=torch.long, device=device).scatter_add_(
        0, batch, torch.ones_like(batch))                              # [n_batch]
    order = torch.argsort(batch, stable=True)                          # [n_atoms]
    offsets = torch.cumsum(counts, 0) - counts                         # exclusive
    slot_sorted = torch.arange(N, device=device) - offsets[batch[order]]
    slot = torch.empty_like(slot_sorted)
    slot[order] = slot_sorted
    valid = torch.zeros(n_batch, n_max, dtype=torch.bool, device=device)
    valid[batch, slot] = True
    return slot, valid

# indexed directly by atomic number, Z = 0..118, so no element list and no
# host-synchronising dict lookup is needed (cf. FixedCharges, which is not
# compile-safe for exactly that reason)
_N_ELEMENTS = 119


class Qeq(nn.Module):
    """
    Charge equilibration: minimize

        E_Qeq = 1/2 q^T A q + sum_i (chi_i q_i + 1/2 J_i q_i^2)

    subject to sum_i q_i = Q_tot per structure, where A is the charge-charge coupling
    matrix of the electrostatics (Ewald_vectorized.compute_A_matrix). Stationarity gives
    the bordered linear system

        [[A + J, 1], [1^T, 0]] [q; lambda] = [-chi; Q_tot]

    which is Eq. 7 of Ko, Finkler, Goedecker & Behler, Nat. Commun. 12, 398 (2021).
    lambda is the structure's chemical potential.

    A is passed in rather than computed here, so this module is pure linear algebra: it
    can be tested against a hand-built A, and a matrix-free solver could later replace
    the explicit matrix without touching it.

    Two subtleties:

    * chi is only ever felt through chi + lambda*1, so a uniform shift of chi is absorbed
      by lambda and leaves q exactly unchanged. Forcing chi > 0 is therefore a
      reparameterization of the readout, not a physical constraint; softplus is used
      because it is monotone and keeps a healthy gradient at the origin, unlike squaring.
    * J > 0 *is* a real requirement: it is what makes A + J positive definite, so that the
      solve returns the minimum rather than a saddle. Note this needs A to carry the
      Gaussian self-energy on its diagonal (compute_A_matrix(self_interaction=True)).

    Atoms may be equilibrated selectively. A frozen atom keeps its incoming charge and
    still fields the equilibrated ones: its row of the system is replaced by the identity
    row, which turns the active rows into

        A_EE q_E + J q_E + lambda = -(chi_E + A_EF q_F),   1^T q_E = Q_tot - sum q_F

    the masked Qeq equations, with no index selection and hence static shapes.

    masked_charge_scope selects what Q_tot refers to. False (the default) reads it as the
    charge of the whole structure, so the equilibrated atoms take up whatever the frozen
    ones leave over. True reads it as the charge of the equilibrated atoms alone, which is
    the same linear system with Q_tot shifted by sum q_F -- useful when the frozen region
    is a separate reservoir rather than part of one neutral system.
    """

    def __init__(self,
                 j_init: float = 1.0,
                 chi_positive: bool = True,
                 eps_j: float = 1e-6,
                 masked_charge_scope: bool = False,
                 ):
        super().__init__()
        self.chi_positive = chi_positive
        self.eps_j = eps_j
        self.masked_charge_scope = masked_charge_scope
        # store softplus^{-1}(j_init) so the effective hardness starts at j_init.
        # log(exp(y) - 1) written as y + log(1 - exp(-y)): the direct form overflows
        # for y greater than about 88 and would seed J_raw with an inf.
        j_init = float(j_init)
        j_raw_init = j_init + math.log(-math.expm1(-j_init))
        self.J_raw = nn.Parameter(torch.full((_N_ELEMENTS,), j_raw_init))

    def forward(self,
                A: torch.Tensor,               # [n_atoms, n_atoms] or [B, n_max, n_max]
                chi: torch.Tensor,             # [n_atoms]
                atomic_numbers: torch.Tensor,  # [n_atoms]
                batch: torch.Tensor,           # [n_atoms]
                total_charge: torch.Tensor,    # [batch_size]
                q_prior: torch.Tensor,         # [n_atoms], charge kept by frozen atoms
                mask: torch.Tensor,            # [n_atoms] bool, True = equilibrate
                slot: Optional[torch.Tensor] = None,   # [n_atoms], from batch_slots
                valid: Optional[torch.Tensor] = None,  # [B, n_max], from batch_slots
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        if slot is not None and valid is not None:
            return self._solve_padded(A, chi, atomic_numbers, batch, total_charge,
                                      q_prior, mask, slot, valid)
        device = A.device
        dtype = A.dtype
        N = A.shape[0]
        B = total_charge.shape[0]

        chi = chi.reshape(-1).to(dtype=dtype)
        if self.chi_positive:
            chi = F.softplus(chi)
        J = F.softplus(self.J_raw.to(dtype=dtype))[atomic_numbers] + self.eps_j  # [n_atoms]
        m = mask.to(dtype=dtype)                                                 # [n_atoms]
        q_prior = q_prior.reshape(-1).to(dtype=dtype)
        total_charge = total_charge.reshape(-1).to(dtype=dtype)

        onehot = (batch.unsqueeze(0)
                  == torch.arange(B, device=device).unsqueeze(1)).to(dtype)      # [B, n_atoms]
        eye_n = torch.eye(N, device=device, dtype=dtype)

        if self.masked_charge_scope:
            # Q_tot counts the equilibrated atoms only. The constraint row below still
            # sums every atom, so add back what the frozen ones carry: the two readings
            # are the same equation with the constant moved across.
            total_charge = total_charge + torch.matmul(onehot, (1.0 - m) * q_prior)

        # atom rows: equilibrated -> (A + J) q + lambda = -chi, frozen -> q_i = q_prior_i
        top_left = (m.unsqueeze(1) * (A + torch.diag(J))
                    + (1.0 - m).unsqueeze(1) * eye_n)                            # [N, N]
        top_right = m.unsqueeze(1) * onehot.transpose(0, 1)                      # [N, B]

        # constraint rows: sum over all atoms (frozen included) = Q_tot. A structure with
        # nothing to equilibrate would make that row inconsistent and the matrix singular,
        # so it is replaced by lambda_b = 0 -- arithmetically, without a branch.
        active = (onehot @ m > 0.5).to(dtype)                                    # [B]
        bot_left = active.unsqueeze(1) * onehot                                  # [B, N]
        bot_right = torch.diag(1.0 - active)                                     # [B, B]

        M = torch.cat([torch.cat([top_left, top_right], dim=1),
                       torch.cat([bot_left, bot_right], dim=1)], dim=0)          # [N+B, N+B]
        rhs = torch.cat([m * (-chi) + (1.0 - m) * q_prior,
                         active * total_charge], dim=0)                           # [N+B]

        sol = torch.linalg.solve(M, rhs.unsqueeze(-1)).squeeze(-1)               # [N+B]
        return sol[:N], sol[N:]

    def _solve_padded(self,
                      A: torch.Tensor,               # [B, n_max, n_max]
                      chi: torch.Tensor,
                      atomic_numbers: torch.Tensor,
                      batch: torch.Tensor,
                      total_charge: torch.Tensor,
                      q_prior: torch.Tensor,
                      mask: torch.Tensor,
                      slot: torch.Tensor,
                      valid: torch.Tensor,
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The same bordered system, but one [n_max+1, n_max+1] block per structure instead
        of a single [n_atoms+B, n_atoms+B] matrix whose cross-structure blocks are zero.
        For B structures of n atoms that is B times less memory and a batched solve
        rather than one B times larger one.

        Padded slots are handled exactly like frozen atoms -- identity row, zero
        right-hand side -- so they solve to zero and drop out of the constraint.
        """
        device = A.device
        dtype = A.dtype
        B = A.shape[0]
        n_max = A.shape[1]

        chi_f = chi.reshape(-1).to(dtype=dtype)
        if self.chi_positive:
            chi_f = F.softplus(chi_f)
        J_f = F.softplus(self.J_raw.to(dtype=dtype))[atomic_numbers] + self.eps_j
        m_f = mask.to(dtype=dtype)
        v = valid.to(dtype=dtype)                                        # [B, n_max]

        # scatter the per-atom quantities into [B, n_max]
        chi_p = torch.zeros(B, n_max, device=device, dtype=dtype)
        J_p = torch.zeros(B, n_max, device=device, dtype=dtype)
        m_p = torch.zeros(B, n_max, device=device, dtype=dtype)
        q_p = torch.zeros(B, n_max, device=device, dtype=dtype)
        chi_p[batch, slot] = chi_f
        J_p[batch, slot] = J_f
        m_p[batch, slot] = m_f
        q_p[batch, slot] = q_prior.reshape(-1).to(dtype=dtype)
        total_charge = total_charge.reshape(-1).to(dtype=dtype)

        active = m_p * v                                                 # real AND selected
        if self.masked_charge_scope:
            total_charge = total_charge + ((1.0 - active) * q_p * v).sum(dim=-1)

        eye_n = torch.eye(n_max, device=device, dtype=dtype).unsqueeze(0)   # [1, n, n]
        top_left = (active.unsqueeze(-1) * (A + torch.diag_embed(J_p))
                    + (1.0 - active).unsqueeze(-1) * eye_n)              # [B, n, n]
        top_right = active.unsqueeze(-1)                                 # [B, n, 1]

        n_active = active.sum(dim=-1)                                    # [B]
        act = (n_active > 0.5).to(dtype)                                 # [B]
        bot_left = (act.unsqueeze(-1) * v).unsqueeze(1)                  # [B, 1, n]
        bot_right = (1.0 - act).reshape(B, 1, 1)                         # [B, 1, 1]

        M = torch.cat([torch.cat([top_left, top_right], dim=2),
                       torch.cat([bot_left, bot_right], dim=2)], dim=1)  # [B, n+1, n+1]
        rhs = torch.cat([active * (-chi_p) + (1.0 - active) * q_p,
                         (act * total_charge).unsqueeze(-1)], dim=1)     # [B, n+1]

        sol = torch.linalg.solve(M, rhs.unsqueeze(-1)).squeeze(-1)       # [B, n+1]
        return sol[:, :n_max][batch, slot], sol[:, n_max]
