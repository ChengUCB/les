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
        self._build_k_grid(N_max, torch.device('cpu'))

    def _build_k_grid(self, N_max: int, device):
        """(Re)build the integer k-grid buffers for a given extent.

        They are registered non-persistently, so they never reach a state dict and can
        be rebuilt at any time -- see set_N_max.
        """
        nvec_all = torch.stack(
            torch.meshgrid(
                torch.arange(-N_max, N_max + 1, device=device),
                torch.arange(-N_max, N_max + 1, device=device),
                torch.arange(-N_max, N_max + 1, device=device),
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

    @torch.jit.unused
    def set_N_max(self, N_max: int):
        """Resize the k-grid on an existing module.

        N_max is not a learned quantity and is absent from the state dict: it only has
        to make the integer grid big enough to reach the cutoff dl already sets, so
        changing it leaves the physics alone as long as N_max >= max|a_i| / dl. Below
        that the reciprocal sum is truncated; above it the extra vectors are masked out
        and the energy is unchanged to the last bit.
        """
        self._build_k_grid(int(N_max), self.nvec_all.device)
        self.N_max = int(N_max)

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

    def compute_A_matrix(self, r, cell, batch: Optional[torch.Tensor] = None,
                         self_interaction: bool = True) -> torch.Tensor:
        """
        Charge-charge coupling matrix, A = d phi / d q, so that the monopole part of the
        long-range energy is exactly E = 1/2 q^T A q per structure. This is the matrix a
        charge equilibration solves against.

        Built from the same closed forms the potential itself is assembled from -- real
        space in make_kernels_vectorized.multipole_potential_field, reciprocal space in
        compute_potential_triclinic -- so norm_factor and every factor of two agree with
        the energy by construction rather than by convention. test_qeq.py asserts it.

        Cross-structure pairs are masked out, so A is block diagonal.

        self_interaction controls whether each Gaussian's interaction with itself sits on
        the diagonal, A_ii += norm_factor * 2/(sigma (2 pi)^{3/2}). Note the two paths
        start from opposite places: the real-space pair grid excludes the diagonal, while
        the reciprocal sum over k already contains it. Passing
        self_interaction=not self.remove_self_interaction reproduces this module's own
        energy; charge equilibration instead wants it present either way (Ko et al. 2021
        Eq. 7 puts it there, and it is what makes A + J positive definite).
        """
        device = r.device
        dtype = r.dtype
        N = r.shape[0]
        if batch is None:
            batch = torch.zeros(N, dtype=torch.int64, device=device)
        else:
            batch = batch.to(device=device, dtype=torch.long)
        same_batch = batch.unsqueeze(1) == batch.unsqueeze(0)                # [N, N]
        zero = torch.zeros((), device=device, dtype=dtype)
        self_diag = self.norm_factor * 2.0 / (self.sigma * self.twopi ** 1.5)

        if not self.is_periodic:
            idx = torch.arange(N, device=device, dtype=torch.long)
            keep_pair = same_batch & (idx.unsqueeze(1) != idx.unsqueeze(0))  # [N, N]
            r_ij = r.unsqueeze(0) - r.unsqueeze(1)                           # [N, N, 3]
            dist_sq = torch.where(keep_pair,
                                  r_ij.pow(2).sum(dim=-1).clamp(min=1e-12),
                                  torch.ones_like(r_ij[..., 0]))             # [N, N]
            dist = dist_sq.pow(0.5)
            a = 1.0 / (self.sigma * (2.0 ** 0.5))
            A = torch.where(keep_pair, torch.special.erf(dist * a) / dist, zero)
            A = A * (self.norm_factor / self.twopi)
            if self_interaction:
                A = A + torch.eye(N, device=device, dtype=dtype) * self_diag
        else:
            cell = cell.to(dtype=dtype)
            B = cell.shape[0]
            nvec = self.nvec_all.to(device=device, dtype=dtype)              # [K, 3]
            K = nvec.shape[0]
            # reciprocal lattice, k-grid and k-factor: steps 1-4 of compute_potential_triclinic
            G = 2 * torch.pi * torch.linalg.inv(cell).transpose(-2, -1)      # [B, 3, 3]
            volume = torch.det(cell)                                         # [B]
            kvec = torch.bmm(nvec.unsqueeze(0).expand(B, -1, -1), G)         # [B, K, 3]
            k_sq = (kvec ** 2).sum(dim=-1)                                   # [B, K]
            valid = ((k_sq > 0) & (k_sq <= self.k_sq_max)
                     & self.hemisphere_mask.unsqueeze(0))                    # [B, K]
            factors = self.factors.to(device=device, dtype=dtype)            # [K]
            weight = torch.exp(-self.sigma_sq_half * k_sq) / (k_sq + 1e-12)  # [B, K]
            weight = weight * factors.unsqueeze(0) * valid.to(dtype=dtype)   # [B, K]

            onehot = (batch.unsqueeze(0)
                      == torch.arange(B, device=device).unsqueeze(1)).to(dtype)  # [B, N]
            onehot_t = onehot.transpose(0, 1)                                # [N, B]
            kvec_at = torch.matmul(onehot_t, kvec.reshape(B, K * 3)).reshape(N, K, 3)
            k_dot_r = (kvec_at * r.unsqueeze(1)).sum(dim=-1)                 # [N, K]
            cos_kr = torch.cos(k_dot_r)                                      # [N, K]
            sin_kr = torch.sin(k_dot_r)                                      # [N, K]
            # the same per-atom prefactor the potential uses, 2 * weight / volume
            pref = torch.matmul(onehot_t, weight * 2.0 / volume.view(B, 1))  # [N, K]
            # phi_i = norm * sum_k pref_k [cos_i cos_j + sin_i sin_j] q_j
            A = (torch.matmul(cos_kr * pref, cos_kr.transpose(0, 1))
                 + torch.matmul(sin_kr * pref, sin_kr.transpose(0, 1)))      # [N, N]
            A = A * self.norm_factor
            A = torch.where(same_batch, A, zero)
            if not self_interaction:
                A = A - torch.eye(N, device=device, dtype=dtype) * self_diag
        return A

    def compute_A_matrix_padded(self, r, batch, slot, valid,
                                self_interaction: bool = True) -> torch.Tensor:
        """
        compute_A_matrix laid out per structure, [n_batch, n_max, n_max], for the
        non-periodic path.

        A batch's cross-structure blocks are zero, so the flat [n_atoms, n_atoms] form
        spends B times the memory it needs. Here every intermediate carries the batch
        axis instead, which for B structures of n atoms turns O((B n)^2) into O(B n^2).

        Only real space. The reciprocal path is dominated by its [n_atoms, K] structure
        factors rather than by A, so laying A out per structure would not pay there.

        Padded slots hold no atom: they are excluded from keep_pair and get a zero
        diagonal, so they contribute nothing.
        """
        device = r.device
        dtype = r.dtype
        B = valid.shape[0]
        n_max = valid.shape[1]
        zero = torch.zeros((), device=device, dtype=dtype)

        r_pad = torch.zeros(B, n_max, 3, device=device, dtype=dtype)
        r_pad[batch, slot] = r
        # same-structure pairs, minus the diagonal, minus anything padded
        off_diag = ~torch.eye(n_max, dtype=torch.bool, device=device).unsqueeze(0)
        keep_pair = valid.unsqueeze(1) & valid.unsqueeze(2) & off_diag    # [B, n, n]

        r_ij = r_pad.unsqueeze(1) - r_pad.unsqueeze(2)                    # [B, n, n, 3]
        dist_sq = torch.where(keep_pair,
                              r_ij.pow(2).sum(dim=-1).clamp(min=1e-12),
                              torch.ones_like(r_ij[..., 0]))              # [B, n, n]
        dist = dist_sq.pow(0.5)
        a = 1.0 / (self.sigma * (2.0 ** 0.5))
        A = torch.where(keep_pair, torch.special.erf(dist * a) / dist, zero)
        A = A * (self.norm_factor / self.twopi)
        if self_interaction:
            self_diag = self.norm_factor * 2.0 / (self.sigma * self.twopi ** 1.5)
            A = A + torch.diag_embed(valid.to(dtype) * self_diag)
        return A

    def energy_from_A_padded(self, A: torch.Tensor, q: torch.Tensor,
                             batch: torch.Tensor, slot: torch.Tensor,
                             valid: torch.Tensor) -> torch.Tensor:
        """energy_from_A for the per-structure layout: E = 1/2 q^T A q, [n_batch]."""
        device = A.device
        dtype = A.dtype
        B = A.shape[0]
        n_max = A.shape[1]
        q_pad = torch.zeros(B, n_max, device=device, dtype=dtype)
        q_pad[batch, slot] = q.reshape(-1).to(dtype=dtype)
        pot = 0.5 * (q_pad * torch.bmm(A, q_pad.unsqueeze(-1)).squeeze(-1)).sum(dim=-1)
        if self.remove_self_interaction:
            self_diag = self.norm_factor * 2.0 / (self.sigma * self.twopi ** 1.5)
            pot = pot - 0.5 * self_diag * (q_pad * q_pad).sum(dim=-1)
        return pot

    def energy_from_A(self, A: torch.Tensor, q: torch.Tensor,
                      batch: torch.Tensor, n_batch: int) -> torch.Tensor:
        """
        Monopole long-range energy read off a matrix from compute_A_matrix, without a
        second pass over the geometry: E = 1/2 q^T A q per structure.

        A is expected in the charge-equilibration convention, self_interaction=True. This
        module's own energy uses self_interaction = not remove_self_interaction, and the
        two differ by norm_factor * 2/(sigma (2 pi)^{3/2}) on the diagonal, so that shift
        is undone here when remove_self_interaction is set.

        Only the monopole term is covered. With dipoles, quadrupoles or polarizabilities
        in play, call forward instead.
        """
        device = A.device
        dtype = A.dtype
        B = n_batch          # passed in from cell.shape[0]: a shape, not tensor data
        q = q.reshape(-1).to(dtype=dtype)
        onehot = (batch.unsqueeze(0)
                  == torch.arange(B, device=device).unsqueeze(1)).to(dtype)   # [B, N]
        pot = 0.5 * torch.matmul(onehot, q * torch.matmul(A, q))               # [B]
        if self.remove_self_interaction:
            self_diag = self.norm_factor * 2.0 / (self.sigma * self.twopi ** 1.5)
            pot = pot - 0.5 * self_diag * torch.matmul(onehot, q * q)
        return pot

    def compute_potential_realspace(self, r, q, cell, batch,
                                    u: Optional[torch.Tensor] = None,
                                    quad: Optional[torch.Tensor] = None,
                                    kappa: Optional[torch.Tensor] = None,
                                    alpha: Optional[torch.Tensor] = None,
                                    e_ext: Optional[torch.Tensor] = None,
                                    compute_field: bool = False):
        """
        Compute the realspace (non-periodic) Ewald potential over an [N, N] pair grid (N = total
        atoms).
        Masks cross-batch pairs, cost is O(N^2). Handles latent monopoles and,
        when given, latent dipoles (u) and quadrupoles (quad), reproducing the
        corresponding terms of the loop-based reference module.
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
        dist_sq = torch.where(keep_pair,
                              r_ij.pow(2).sum(dim=-1).clamp(min=1e-12),
                              torch.ones_like(r_ij[..., 0]))            # [N, N]
        dist = dist_sq.pow(0.5)                                          # [N, N]
        rinv = dist.pow(-1)                                              # [N, N]
        a = 1.0 / (self.sigma * (2.0 ** 0.5))
        erf_val = torch.special.erf(dist * a)                            # [N, N]

        # monopole-monopole: 1/2 \Sum_{i!=j} q_i q_j erf(a r)/r
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

        pot_per_pair = torch.where(keep_pair, pot_per_pair,
                                   torch.zeros((), device=device, dtype=dtype))

        pair_batch = batch.unsqueeze(1).expand(N, N)                     # [N, N]
        pot_per_batch = torch.zeros(B, device=device, dtype=dtype)
        pot_per_batch.scatter_add_(0, pair_batch.reshape(-1), pot_per_pair.reshape(-1))
        pot_per_batch = pot_per_batch / self.twopi                       # 1/(4\pi\epsilon0) prefactor

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
            if not self.remove_self_interaction:
                e_phi = e_phi + q * (2.0 / (self.sigma * self.twopi ** 1.5)) * self.norm_factor
                if u is not None:
                    c_self = (4.0 / (3.0 * torch.pi ** 0.5)) * (a ** 3) * norm_const
                    e_field = e_field - c_self * u.to(dtype=dtype)

            onehot = (batch.unsqueeze(0)
                      == torch.arange(B, device=device).unsqueeze(1)).to(dtype)  # [B, N]
            if kappa is not None:
                q_induced = self._get_induced_q(e_phi, kappa)
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
        dtype = r.dtype

        N, n_q = q.shape
        B = cell.shape[0]
        cell = cell.to(dtype=dtype)
        q = q.to(dtype=dtype)
        nvec = self.nvec_all.to(device=device, dtype=dtype)  # [K, 3]
        K = nvec.shape[0] # K = (2*N_max+1)^3

        # --- 1. Reciprocal lattice G_b = 2\pi (M_b^{-1})^T ---
        cell_inv = torch.linalg.inv(cell) # [B, 3, 3]
        G = 2 * torch.pi * cell_inv.transpose(-2, -1)  # [B, 3, 3], G = 2\pi(M^{-1}).T

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
        onehot = (batch.unsqueeze(0)
                  == torch.arange(B, device=device).unsqueeze(1)).to(dtype)  # [B, N]
        onehot_t = onehot.transpose(0, 1)                                    # [N, B]
        kvec_for_atoms = torch.matmul(
            onehot_t, kvec.reshape(B, K * 3)).reshape(N, K, 3)               # [N, K, 3]
        k_dot_r = (kvec_for_atoms * r.unsqueeze(1)).sum(dim=-1)  # [N, K]
        #for torchscript compatibility, to avoid dtype mismatch, only use real part
        cos_k_dot_r = torch.cos(k_dot_r) # [N, K]
        sin_k_dot_r = torch.sin(k_dot_r) # [N, K]
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
            prefactor = torch.matmul(
                onehot_t, weight * 2.0 / volume.view(B, 1))               # [N, K]

            S_real_at = torch.matmul(
                onehot_t, S_real.reshape(B, n_q * K)).reshape(N, n_q, K)
            S_imag_at = torch.matmul(
                onehot_t, S_imag.reshape(B, n_q * K)).reshape(N, n_q, K)

            if kappa is not None or compute_field:
                # Re[e^{-ikr} S(k)] = cos(kr) S_real + sin(kr) S_imag
                term_real = S_real_at * cos_exp + S_imag_at * sin_exp         # [N, n_q, K]
                e_phi = torch.bmm(term_real, prefactor.unsqueeze(-1)).squeeze(-1) * self.norm_factor
                if self.remove_self_interaction:
                    e_phi = e_phi - q * (2.0 / (self.sigma * self.twopi ** 1.5)) * self.norm_factor

                if kappa is not None:
                    q_induced = self._get_induced_q(e_phi, kappa)
                    induced_per_batch = onehot @ (e_phi * q_induced)           # [B, n_q]
                    pot_per_batch_per_q = pot_per_batch_per_q + 0.5 * induced_per_batch / self.norm_factor

            if alpha is not None or compute_field:
                # Im[e^{-ikr} S(k)] contributes to the field
                term_imag = S_real_at * sin_exp - S_imag_at * cos_exp         # [N, n_q, K]
                wf = prefactor.unsqueeze(1) * term_imag                       # [N, n_q, K]
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
