import torch
import torch.nn as nn
from typing import Optional, Tuple

from les.module.make_kernels_vectorized import multipole_pair_energy

__all__ = ['Ewald_vectorized']


class Ewald_vectorized(nn.Module):
    def __init__(self,
                 dl=2.0,  # grid resolution
                 sigma=1.0,  # width of the Gaussian on each atom
                 remove_self_interaction=True,
                 norm_factor=90.4756,
                 is_periodic: bool = True,
                 N_max: int = 10, # cell vector norm 20 divided by dl=2.0, increase if needed
                 use_epsilon_r_scaling=False,
                 ):
        super().__init__()
        self.dl = dl
        self.sigma = sigma
        self.sigma_sq_half = sigma ** 2 / 2.0
        self.twopi = 2.0 * torch.pi
        self.twopi_sq = self.twopi ** 2
        self.remove_self_interaction = remove_self_interaction
        # 1/2\epsilon_0, where \epsilon_0 is the vacuum permittivity
        # \epsilon_0 = 5.52635*10^{-3} e^2 eV^{-1} A^{-1}
        self.norm_factor = norm_factor
        self.k_sq_max = (self.twopi / self.dl) ** 2

        self.is_periodic = is_periodic
        self.N_max = N_max
        self.use_epsilon_r_scaling = use_epsilon_r_scaling

        ### fixed k-grid for periodic case, precompute ###
        nvec_all = torch.stack(
            torch.meshgrid(
                torch.arange(-N_max, N_max + 1),
                torch.arange(-N_max, N_max + 1),
                torch.arange(-N_max, N_max + 1),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 3) # [K,3], K = (2*N_max+1)^3
        self.register_buffer('nvec_all', nvec_all, persistent=False)

        non_zero = (nvec_all != 0)
        has_non_zero = non_zero.any(dim=1)
        first_non_zero_idx = torch.argmax(non_zero.to(torch.int), dim=1)
        sign = torch.gather(nvec_all, 1, first_non_zero_idx.unsqueeze(1)).squeeze(1) # [K]
        hemisphere_mask = (sign > 0) | (~has_non_zero) # [K]
        is_origin = ~has_non_zero # [K]
        factors = torch.where(is_origin, 1.0, 2.0)  # [K]

        self.register_buffer('hemisphere_mask', hemisphere_mask, persistent=False) # [K] bool
        self.register_buffer('factors', factors, persistent=False) # [K] float


    def forward(self,
                q: torch.Tensor,  # [n_atoms, n_q] or [n_atoms]
                r: torch.Tensor, # [n_atoms, 3]
                cell: torch.Tensor, # [batch_size, 3, 3]
                batch: Optional[torch.Tensor] = None,
                u: Optional[torch.Tensor] = None, # [n_atoms, n_q, 3] or [n_atoms, 3]
                quad: Optional[torch.Tensor] = None, # [n_atoms, 3, 3]
                kappa: Optional[torch.Tensor] = None, # [n_atoms, n_q] or [n_atoms]
                alpha: Optional[torch.Tensor] = None, # [n_atoms, n_q] or [n_atoms]
                e_ext: Optional[torch.Tensor] = None,
                compute_field: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # terms not vectorized yet: use the legacy module (omit is_periodic) for these
        if kappa is not None or alpha is not None or e_ext is not None or compute_field:
            raise NotImplementedError(
                "the vectorized Ewald currently supports latent charges, dipoles and "
                "quadrupoles; omit 'is_periodic' to fall back to the legacy module for "
                "induced charges/dipoles, external fields or field output"
            )

        if q.dim() == 1:
            q = q.unsqueeze(1)

        # Check the input dimension
        n, d = r.shape
        assert d == 3, 'r dimension error, r.shape[1] must be 3'
        assert n == q.size(0), 'q dimension error, q.shape[0] must be n_atoms'
        if batch is None:
            batch = torch.zeros(n, dtype=torch.int64, device=r.device)
        else:
            batch = batch.to(device=r.device, dtype=torch.long)


        if u is not None:
            if u.dim() == 2 and u.shape[1] == 3:
                u = u.unsqueeze(1)  # [n_atoms, 3] -> [n_atoms, 1, 3]
            assert u.shape == (n, q.shape[1], 3), 'u dimension error, expected [n_atoms, n_q, 3]'

        if quad is not None:
            if quad.dim() == 3 and quad.shape[1] == 3:
                quad = quad.unsqueeze(1)  # [n_atoms, 3, 3] -> [n_atoms, 1, 3, 3]
            assert quad.shape == (n, q.shape[1], 3, 3), 'quad dimension error, expected [n_atoms, n_q, 3, 3]'

        if not self.is_periodic: # non-periodic
            assert cell is not None, 'fake cell needed for non-periodic case (ex. torch.zeros(n_batch, 3,3))'
            results = self.compute_potential_realspace(r, q, cell, batch, u=u, quad=quad)
        else: # periodic
            results = self.compute_potential_triclinic(r, q, cell, batch, u=u, quad=quad)

        # same (pot, q_induced, u_induced) interface as the legacy module
        q_induced = torch.zeros_like(q)
        u_induced = torch.zeros((n, q.shape[1], 3), device=r.device, dtype=r.dtype)
        return results, q_induced, u_induced
    


    def compute_potential_realspace(self, r, q, cell, batch,
                                    u: Optional[torch.Tensor] = None,
                                    quad: Optional[torch.Tensor] = None):
        """
        Realspace (non-periodic) Ewald over an [N, N] pair grid (N = total
        atoms).

        Masks cross-batch pairs, cost is O(N^2). Handles latent monopoles and,
        when given, latent dipoles (u) and quadrupoles (quad), reproducing the
        corresponding terms of the loop-based reference module.

        The quadrupole kernels are contracted analytically to per-pair scalars
        instead of building the [N, N, 3, 3, 3(, 3)] tensors of the reference.
        """
        device = r.device
        dtype = r.dtype
        N = r.shape[0]
        n_q = q.shape[1]
        B = cell.shape[0]
        q = q.to(dtype=dtype)

        idx = torch.arange(N, device=device, dtype=torch.long)
        pair_i, pair_j = torch.meshgrid(idx, idx, indexing="ij")
        same_batch = batch[pair_i] == batch[pair_j]
        pair_j_safe = torch.where(same_batch, pair_j, torch.zeros_like(pair_j))
        keep = (same_batch & (pair_i != pair_j)).to(dtype)               # [N, N]

        # r_ij = r_j - r_i, the convention the reference kernels are written in
        # (the sign matters for the odd-in-r_ij charge-dipole term).
        r_ij = r[pair_j_safe] - r[pair_i]                                # [N, N, 3]
        dist_sq = r_ij.pow(2).sum(dim=-1).clamp(min=1e-12)               # [N, N]
        dist = dist_sq.pow(0.5)                                          # [N, N]
        rinv = dist.pow(-1)                                              # [N, N]
        a = 1.0 / (self.sigma * (2.0 ** 0.5))
        erf_val = torch.special.erf(dist * a)                            # [N, N]

        # monopole-monopole: 1/2 Σ_{i≠j} q_i q_j erf(a r)/r
        qq_pair = (q[pair_i] * q[pair_j_safe]).sum(dim=-1)               # [N, N]
        pot_per_pair = 0.5 * qq_pair * erf_val * rinv                    # [N, N]

        # dipole/quadrupole terms, contracted per pair (see make_kernels_vectorized)
        if u is not None or quad is not None:
            pot_per_pair = pot_per_pair + multipole_pair_energy(
                q=q,
                u=u.to(dtype=dtype) if u is not None else None,
                quad=quad.to(dtype=dtype) if quad is not None else None,
                r_ij=r_ij, dist=dist, dist_sq=dist_sq, rinv=rinv, erf_val=erf_val,
                sigma=self.sigma, pair_i=pair_i, pair_j=pair_j_safe,
            )

        pot_per_pair = pot_per_pair * keep                               # [N, N]

        pair_batch = batch[pair_i]                                       # [N, N]
        pot_per_batch = torch.zeros(B, device=device, dtype=dtype)
        pot_per_batch.scatter_add_(0, pair_batch.reshape(-1), pot_per_pair.reshape(-1))
        pot_per_batch = pot_per_batch / self.twopi                       # 1/(4πε0) prefactor

        if not self.remove_self_interaction:
            q_sq_per_atom = (q ** 2).sum(dim=1)        # [N]
            self_per_batch = torch.zeros(B, device=device, dtype=dtype)
            self_per_batch.scatter_add_(0, batch, q_sq_per_atom)
            pot_per_batch = pot_per_batch + self_per_batch / (self.sigma * self.twopi ** (3.0 / 2.0))
            if u is not None:
                u_sq_per_atom = (u ** 2).sum(dim=-1).sum(dim=-1)   # [N]
                u_self_per_batch = torch.zeros(B, device=device, dtype=dtype)
                u_self_per_batch.scatter_add_(0, batch, u_sq_per_atom)
                pot_per_batch = pot_per_batch + u_self_per_batch / (
                    3.0 * self.sigma ** 3.0 * self.twopi ** (3.0 / 2.0))
            if quad is not None:
                quad_sq_per_atom = (quad ** 2).sum(dim=-1).sum(dim=-1).sum(dim=-1)   # [N]
                quad_self_per_batch = torch.zeros(B, device=device, dtype=dtype)
                quad_self_per_batch.scatter_add_(0, batch, quad_sq_per_atom)
                pot_per_batch = pot_per_batch + quad_self_per_batch / (
                    10.0 * self.sigma ** 5.0 * self.twopi ** (3.0 / 2.0))

        return pot_per_batch * self.norm_factor
    

    def compute_potential_triclinic(self, r, q, cell, batch,
                                    u: Optional[torch.Tensor] = None,
                                    quad: Optional[torch.Tensor] = None):

        device = r.device
        # single source of truth for the compute dtype: callers may hand over a
        # float64 cell (LAMMPS/ASE do) with float32 positions, and mixing the two
        # makes the structure-factor accumulation below fail on dtype.
        dtype = r.dtype

        N, n_q = q.shape
        B = cell.shape[0]
        cell = cell.to(dtype=dtype)
        q = q.to(dtype=dtype)
        nvec = self.nvec_all.to(device=device, dtype=dtype)  # [K, 3]
        K = nvec.shape[0] # K = (2*N_max+1)^3

        # --- 1. Reciprocal lattice G_b = 2π (M_b^{-1})^T ---
        cell_inv = torch.linalg.inv(cell) # [B, 3, 3]
        G = 2 * torch.pi * cell_inv.transpose(-2, -1)  # [B, 3, 3], G = 2π(M^{-1}).T

        # --- 2. kvec[b, k, :] = nvec[k, :] @ G[b, :, :] ---
        nvec_expanded = nvec.unsqueeze(0).expand(B, -1, -1)  # [B, K, 3]
        kvec = torch.bmm(nvec_expanded, G)  # [B, K, 3]
        k_sq = (kvec ** 2).sum(dim=-1)  # [B, K]
        # --- 3. k cutoff + hemisphere mask ---
        # Apply k-space cutoff and filter
        # Determine symmetry factors (handle hemisphere to avoid double-counting)
        # Include nvec if first non-zero component is positive
        valid_kcut = (k_sq > 0) & (k_sq <= self.k_sq_max)  # [B, K]
        hemi = self.hemisphere_mask.unsqueeze(0) # [1, K]
        valid_mask = valid_kcut & hemi  # [B, K]

        # --- 4. k-factor, exp(-σ^2/2 k^2) / k^2 for exponent = 1 ---
        eps = 1e-12
        kfac_full = torch.exp(-self.sigma_sq_half * k_sq) / (k_sq + eps) # [B, K]
        factors = self.factors.to(device=kfac_full.device, dtype=kfac_full.dtype) # [K]

        weight = kfac_full * factors.unsqueeze(0) # [B, K]
        weight = weight * valid_mask.to(dtype=weight.dtype) # [B, K]

        # --- 5. Structure factor S(k) = Σ_i q_i e^{i k·r_i} ---
        kvec_for_atoms = kvec[batch] # [N, K, 3]
        k_dot_r = (kvec_for_atoms * r.unsqueeze(1)).sum(dim=-1)  # [N, K]
        #for torchscript compatibility, to avoid dtype mismatch, only use real part
        cos_k_dot_r = torch.cos(k_dot_r) # [N, K]
        sin_k_dot_r = torch.sin(k_dot_r) # [N, K]
        # expand dimensions for broadcasting
        cos_exp = cos_k_dot_r.unsqueeze(1)  # [N, 1, K]
        sin_exp = sin_k_dot_r.unsqueeze(1)  # [N, 1, K]
        q_exp = q.unsqueeze(2)               # [N, n_q, 1]

        S_k_real_per_atom = q_exp * cos_exp  # [N, n_q, K]
        S_k_imag_per_atom = q_exp * sin_exp  # [N, n_q, K]

        if u is not None:
            # dipoles enter the structure factor as S(k) += i (k·u) e^{i k·r}
            u = u.to(dtype=dtype)
            uk = torch.bmm(u, kvec_for_atoms.transpose(1, 2))  # [N, n_q, K]
            S_k_real_per_atom = S_k_real_per_atom - uk * sin_exp
            S_k_imag_per_atom = S_k_imag_per_atom + uk * cos_exp

        if quad is not None:
            # quadrupoles enter the structure factor as S(k) += -1/2 (k·Q·k) e^{i k·r}
            quad = quad.to(dtype=dtype)
            kQ = torch.matmul(kvec_for_atoms.unsqueeze(1), quad)  # [N, n_q, K, 3]
            qk2 = (kQ * kvec_for_atoms.unsqueeze(1)).sum(dim=-1)  # [N, n_q, K]
            S_k_real_per_atom = S_k_real_per_atom - 0.5 * qk2 * cos_exp
            S_k_imag_per_atom = S_k_imag_per_atom - 0.5 * qk2 * sin_exp

        # sum over atoms to get S_k
        S_real = torch.zeros(B, n_q, K, device=device, dtype=dtype)  # [B, n_q, K]
        S_imag = torch.zeros_like(S_real)  # [B, n_q, K]

        index = batch.view(N, 1, 1).expand(-1, n_q, K)  # [N, n_q, K]
        S_real = S_real.scatter_add_(0, index, S_k_real_per_atom)
        S_imag = S_imag.scatter_add_(0, index, S_k_imag_per_atom)
        S_k_sq = S_real.pow(2) + S_imag.pow(2)  # [B, n_q, K]


        # --- 6. Compute potential, (2π/volume)* sum(factors * kfac * |S(k)|^2)---
        w = weight.unsqueeze(1)  # [B, 1, K]
        contrib = w * S_k_sq  # [B, n_q, K]

        volume = torch.det(cell)  # [B]
        pot_per_batch_per_q = contrib.sum(dim=-1) / volume.view(B, 1)  # [B, n_q]

        # --- Remove self-interaction if applicable ---
        if self.remove_self_interaction:
            index_q = batch.view(N, 1).expand(-1, n_q)  # [N, n_q]
            self_per_batch = torch.zeros(B, n_q, device=device, dtype=dtype)  # [B, n_q]
            self_per_batch.scatter_add_(0, index_q, q ** 2)  # [B, n_q]
            self_term = self_per_batch / (self.sigma * self.twopi ** 1.5)  # [B, n_q]
            pot_per_batch_per_q = pot_per_batch_per_q - self_term  # [B, n_q]
            if u is not None:
                u_self_per_batch = torch.zeros(B, n_q, device=device, dtype=dtype)
                u_self_per_batch.scatter_add_(0, index_q, (u ** 2).sum(dim=2))  # [B, n_q]
                pot_per_batch_per_q = pot_per_batch_per_q - u_self_per_batch / (
                    3.0 * self.sigma ** 3.0 * self.twopi ** 1.5)
            if quad is not None:
                quad_self_per_batch = torch.zeros(B, n_q, device=device, dtype=dtype)
                quad_self_per_batch.scatter_add_(0, index_q,
                                                 (quad ** 2).sum(dim=-1).sum(dim=-1))
                pot_per_batch_per_q = pot_per_batch_per_q - quad_self_per_batch / (
                    10.0 * self.sigma ** 5.0 * self.twopi ** 1.5)

        pot_per_batch = pot_per_batch_per_q.sum(dim=1)  # [B]
        return pot_per_batch * self.norm_factor  # [B]
