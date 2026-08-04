import torch
import torch.nn as nn
from typing import Optional, Tuple

from les.module.make_kernels_vectorized import (
    multipole_pair_energy,
    multipole_potential_field,
)

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
            out = self.compute_potential_realspace(r, q, cell, batch, u=u, quad=quad,
                                                   kappa=kappa, alpha=alpha, e_ext=e_ext,
                                                   compute_field=compute_field)
        else: # periodic
            out = self.compute_potential_triclinic(r, q, cell, batch, u=u, quad=quad,
                                                   kappa=kappa, alpha=alpha, e_ext=e_ext,
                                                   compute_field=compute_field)
        # same (pot, q_induced, u_induced) interface as the legacy module
        return out['pot'], out['q_induced'], out['u_induced']
    


    def _get_induced_q(self, e_phi, kappa):
        if kappa.dim() == 1:
            kappa = kappa.unsqueeze(1)
        assert kappa.dim() == 2, 'kappa dimension error'
        return - kappa.to(dtype=e_phi.dtype) * e_phi  # [N, n_q]

    def _get_induced_u(self, e_field, alpha, e_ext: Optional[torch.Tensor] = None):
        if e_ext is not None:
            e_field = e_field + e_ext.to(dtype=e_field.dtype)[None, None, :]
        alpha = alpha.to(dtype=e_field.dtype)
        if alpha.dim() == 1 or (alpha.dim() == 3 and alpha.shape[1:3] == (3, 3)):
            alpha = alpha.unsqueeze(1)
        if alpha.dim() == 2:  # isotropic
            return e_field * alpha.unsqueeze(2)  # [N, n_q, 3]
        elif alpha.dim() == 4 and alpha.shape[2:4] == (3, 3):  # anisotropic
            # sum_c e_field[i,q,c] alpha[i,q,c,d]; broadcast rather than einsum,
            # which breaks AOTInductor export
            return (e_field.unsqueeze(-1) * alpha).sum(dim=2)  # [N, n_q, 3]
        else:
            raise ValueError('alpha dimension error')

    def _get_epsilon_r(self, alpha, volume, batch, B: int, n_q: int):
        """Susceptibility-based relative permittivity, per configuration."""
        epsilon_0 = 0.00552635  # e^2 eV^{-1} A^{-1}
        alpha = alpha.to(dtype=volume.dtype)
        if alpha.dim() == 1 or (alpha.dim() == 3 and alpha.shape[1:3] == (3, 3)):
            alpha = alpha.unsqueeze(1)
        if alpha.dim() == 2:  # isotropic, [N, n_q]
            per_atom = alpha
        elif alpha.dim() == 4 and alpha.shape[2:4] == (3, 3):  # anisotropic
            per_atom = alpha.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 3.0  # [N, n_q]
        else:
            raise ValueError('alpha dimension error')
        index_q = batch.view(-1, 1).expand(-1, n_q)
        summed = torch.zeros(B, n_q, device=alpha.device, dtype=volume.dtype)
        summed.scatter_add_(0, index_q, per_atom)
        return summed / volume.view(B, 1) / epsilon_0 + 1.0  # [B, n_q]

    def compute_potential_realspace(self, r, q, cell, batch,
                                    u: Optional[torch.Tensor] = None,
                                    quad: Optional[torch.Tensor] = None,
                                    kappa: Optional[torch.Tensor] = None,
                                    alpha: Optional[torch.Tensor] = None,
                                    e_ext: Optional[torch.Tensor] = None,
                                    compute_field: bool = False):
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
        same_batch = batch.unsqueeze(1) == batch.unsqueeze(0)            # [N, N]
        keep_pair = same_batch & (idx.unsqueeze(1) != idx.unsqueeze(0))  # [N, N] bool
        keep = keep_pair.to(dtype)

        # r_ij = r_j - r_i, the convention the reference kernels are written in
        # (the sign matters for the odd-in-r_ij charge-dipole term).
        r_ij = r.unsqueeze(0) - r.unsqueeze(1)                           # [N, N, 3]
        # Excluded pairs (i == j, and pairs from different configurations) are
        # masked out by `keep` below, but they must not blow up on the way there:
        # i == j has zero separation, and a clamped 1e-12 gives rinv**5 ~ 1e30,
        # which overflows float32 once the quadrupole kernels multiply it out. The
        # mask then turns inf into NaN, and the NaN reaches the gradients -- which
        # is what made compiled non-periodic training diverge while eager did not.
        # Hand those entries a harmless unit separation instead.
        dist_sq = torch.where(keep_pair,
                              r_ij.pow(2).sum(dim=-1).clamp(min=1e-12),
                              torch.ones_like(r_ij[..., 0]))            # [N, N]
        dist = dist_sq.pow(0.5)                                          # [N, N]
        rinv = dist.pow(-1)                                              # [N, N]
        a = 1.0 / (self.sigma * (2.0 ** 0.5))
        erf_val = torch.special.erf(dist * a)                            # [N, N]

        # monopole-monopole: 1/2 Σ_{i≠j} q_i q_j erf(a r)/r
        qq_pair = (q.unsqueeze(1) * q.unsqueeze(0)).sum(dim=-1)           # [N, N]
        pot_per_pair = 0.5 * qq_pair * erf_val * rinv                    # [N, N]

        # dipole/quadrupole terms, contracted per pair (see make_kernels_vectorized)
        if u is not None or quad is not None:
            pot_per_pair = pot_per_pair + multipole_pair_energy(
                q=q,
                u=u.to(dtype=dtype) if u is not None else None,
                quad=quad.to(dtype=dtype) if quad is not None else None,
                r_ij=r_ij, dist=dist, dist_sq=dist_sq, rinv=rinv, erf_val=erf_val,
                sigma=self.sigma,
            )

        # where, not a 0/1 multiply: an excluded pair can carry an inf and
        # inf * 0 is NaN (ChengUCB/les#11), in the backward as much as the forward
        pot_per_pair = torch.where(keep_pair, pot_per_pair,
                                   torch.zeros((), device=device, dtype=dtype))

        pair_batch = batch.unsqueeze(1).expand(N, N)                     # [N, N]
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

        # --- induced charges / dipoles from the real-space potential and field ---
        q_induced = torch.zeros(N, n_q, device=device, dtype=dtype)
        u_induced = torch.zeros(N, n_q, 3, device=device, dtype=dtype)
        e_phi = torch.zeros(N, n_q, device=device, dtype=dtype)
        e_field = torch.zeros(N, n_q, 3, device=device, dtype=dtype)

        if kappa is not None or alpha is not None or compute_field:
            norm_const = self.norm_factor / self.twopi
            e_phi, e_field = multipole_potential_field(
                q=q,
                u=u.to(dtype=dtype) if u is not None else None,
                quad=quad.to(dtype=dtype) if quad is not None else None,
                r_ij=r_ij, dist=dist, dist_sq=dist_sq, rinv=rinv, erf_val=erf_val,
                sigma=self.sigma, norm_const=norm_const,
                keep_pair=keep_pair,
            )
            # the real-space sum already excludes i == j, so the self terms are
            # added back here when they are meant to be kept
            if not self.remove_self_interaction:
                e_phi = e_phi + q * (2.0 / (self.sigma * self.twopi ** 1.5)) * self.norm_factor
                if u is not None:
                    c_self = (4.0 / (3.0 * torch.pi ** 0.5)) * (a ** 3) * norm_const
                    e_field = e_field - c_self * u.to(dtype=dtype)

            # one-hot segment sum: a reduction feeding scatter_add_ is miscompiled
            onehot = (batch.unsqueeze(0)
                      == torch.arange(B, device=device).unsqueeze(1)).to(dtype)  # [B, N]
            if kappa is not None:
                q_induced = self._get_induced_q(e_phi, kappa)
                # pot_per_batch is in units of physical energy / norm_factor
                pot_per_batch = pot_per_batch + 0.5 * (
                    onehot @ (e_phi * q_induced)).sum(dim=1) / self.norm_factor
            if alpha is not None:
                u_induced = self._get_induced_u(e_field, alpha, e_ext)
                pot_per_batch = pot_per_batch - 0.5 * (
                    onehot @ (e_field * u_induced).sum(dim=-1)).sum(dim=1) / self.norm_factor

        return {'pot': pot_per_batch * self.norm_factor,  # [B]
                'q_induced': q_induced,
                'u_induced': u_induced,
                'phi': e_phi,
                'field': e_field}
    

    def compute_potential_triclinic(self, r, q, cell, batch,
                                    u: Optional[torch.Tensor] = None,
                                    quad: Optional[torch.Tensor] = None,
                                    kappa: Optional[torch.Tensor] = None,
                                    alpha: Optional[torch.Tensor] = None,
                                    e_ext: Optional[torch.Tensor] = None,
                                    compute_field: bool = False):

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

        volume = torch.det(cell)  # [B]
        if alpha is not None and self.use_epsilon_r_scaling:
            epsilon_r = self._get_epsilon_r(alpha, volume, batch, B, n_q)  # [B, n_q]
        else:
            epsilon_r = torch.ones(B, n_q, device=device, dtype=dtype)

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
        # Broadcast per-configuration quantities onto the atoms with a one-hot matmul
        # instead of indexing with `batch`. The backward of such a gather is an
        # atomic_add scatter, and the inductor CPU backend cannot vectorise it once the
        # atom count is dynamic ("assert index.is_vec") -- which is how LAMMPS calls an
        # AOTInductor-exported model.
        onehot = (batch.unsqueeze(0)
                  == torch.arange(B, device=device).unsqueeze(1)).to(dtype)  # [B, N]
        onehot_t = onehot.transpose(0, 1)                                    # [N, B]
        kvec_for_atoms = torch.matmul(
            onehot_t, kvec.reshape(B, K * 3)).reshape(N, K, 3)               # [N, K, 3]
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
        # Sum the per-atom structure factors into their configuration with a one-hot
        # matmul rather than scatter_add_ over an [N, n_q, K] index. The scatter is
        # what the physics asks for, but the inductor CPU backend cannot vectorise
        # the atomic_add it emits ("assert index.is_vec") once the atom count is
        # dynamic, which is exactly how LAMMPS calls an AOTInductor-exported model.
        S_real = torch.matmul(onehot,
                              S_k_real_per_atom.reshape(N, n_q * K)).reshape(B, n_q, K)
        S_imag = torch.matmul(onehot,
                              S_k_imag_per_atom.reshape(N, n_q * K)).reshape(B, n_q, K)
        S_k_sq = S_real.pow(2) + S_imag.pow(2)  # [B, n_q, K]


        # --- 6. Compute potential, (2π/volume)* sum(factors * kfac * |S(k)|^2)---
        w = weight.unsqueeze(1)  # [B, 1, K]
        contrib = w * S_k_sq  # [B, n_q, K]

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

        # --- induced charges / dipoles from the reciprocal-space potential ---
        q_induced = torch.zeros(N, n_q, device=device, dtype=dtype)
        u_induced = torch.zeros(N, n_q, 3, device=device, dtype=dtype)
        e_phi = torch.zeros(N, n_q, device=device, dtype=dtype)
        e_field = torch.zeros(N, n_q, 3, device=device, dtype=dtype)

        if kappa is not None or alpha is not None or compute_field:
            # prefactor = factors * 2 kfac / V, broadcast onto the atoms (`onehot` and
            # `onehot_t` come from the structure-factor sum above; a matmul with them
            # also serves as the segment sum for the induced energies, where a
            # reduction feeding scatter_add_ is miscompiled by inductor)
            prefactor = torch.matmul(
                onehot_t, weight * 2.0 / volume.view(B, 1))               # [N, K]
            # broadcast each configuration's structure factor back to its atoms with
            # the transposed one-hot rather than S_real[batch]: the backward of that
            # gather is the same atomic_add scatter the inductor CPU backend cannot
            # vectorise under dynamic shapes
            S_real_at = torch.matmul(
                onehot_t, S_real.reshape(B, n_q * K)).reshape(N, n_q, K)
            S_imag_at = torch.matmul(
                onehot_t, S_imag.reshape(B, n_q * K)).reshape(N, n_q, K)

            if kappa is not None or compute_field:
                # Re[e^{-ikr} S(k)] = cos(kr) S_real + sin(kr) S_imag
                term_real = S_real_at * cos_exp + S_imag_at * sin_exp         # [N, n_q, K]
                # bmm rather than (prefactor * term).sum(-1): the fused reduction over
                # the k-grid is miscompiled on the Metal backend
                e_phi = torch.bmm(term_real, prefactor.unsqueeze(-1)).squeeze(-1) * self.norm_factor
                if self.remove_self_interaction:
                    e_phi = e_phi - q * (2.0 / (self.sigma * self.twopi ** 1.5)) * self.norm_factor

                if kappa is not None:
                    q_induced = self._get_induced_q(e_phi, kappa)
                    induced_per_batch = onehot @ (e_phi * q_induced)           # [B, n_q]
                    # pot is scaled by norm_factor at the end, e_phi already carries it
                    pot_per_batch_per_q = pot_per_batch_per_q + 0.5 * induced_per_batch / self.norm_factor

            if alpha is not None or compute_field:
                # Im[e^{-ikr} S(k)] contributes to the field
                term_imag = S_real_at * sin_exp - S_imag_at * cos_exp         # [N, n_q, K]
                wf = prefactor.unsqueeze(1) * term_imag                       # [N, n_q, K]
                # bmm rather than a broadcast product: keeps the memory at the output
                # size and stays friendly to AOTInductor export
                e_field = torch.bmm(wf, kvec_for_atoms) * self.norm_factor    # [N, n_q, 3]
                if self.remove_self_interaction and u is not None:
                    a = 1.0 / (self.sigma * (2.0 ** 0.5))
                    c_self = (4.0 / (3.0 * torch.pi ** 0.5)) * (a ** 3) / self.twopi * self.norm_factor
                    e_field = e_field + c_self * u

                if alpha is not None:
                    u_induced = self._get_induced_u(e_field, alpha, e_ext)
                    induced_u_per_batch = onehot @ (e_field * u_induced).sum(dim=-1)  # [B, n_q]
                    pot_per_batch_per_q = pot_per_batch_per_q - 0.5 * induced_u_per_batch / self.norm_factor

        pot_per_batch = pot_per_batch_per_q.sum(dim=1)  # [B]
        return {'pot': pot_per_batch * self.norm_factor,  # [B]
                'q_induced': q_induced,
                'u_induced': u_induced,
                'phi': e_phi,
                'field': e_field,
                'epsilon_r': epsilon_r}
