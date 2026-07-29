import torch
import torch.nn as nn
from itertools import product
from typing import Dict, Optional, Tuple, List

from les.module.make_kernels import make_kernels

__all__ = ['Ewald', 'Ewald_vectorized']


class _ExpSaveInput(torch.autograd.Function):
    """
    exp() whose backward recomputes from the saved *input*.
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return grad_out * torch.exp(x)

class Ewald(nn.Module):
    def __init__(self,
                 dl=2.0,  # grid resolution
                 sigma=1.0,  # width of the Gaussian on each atom
                 remove_self_interaction=True,
                 norm_factor=90.4756,
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
        self.use_epsilon_r_scaling = use_epsilon_r_scaling

    def forward(self,
                q: torch.Tensor,  # [n_atoms, n_q] or [n_atoms]
                r: torch.Tensor, # [n_atoms, 3]
                cell: torch.Tensor, # [batch_size, 3, 3]
                batch: Optional[torch.Tensor] = None,
                u: Optional[torch.Tensor] = None, # [n_atoms, n_q, 3] or [natoms, 3]
                quad: Optional[torch.Tensor] = None, # [natoms,3,3]
                kappa: Optional[torch.Tensor] = None, # [n_atoms, n_q] or [n_atoms]
                alpha: Optional[torch.Tensor] = None, # [n_atoms, n_q] or [n_atoms, n_q, 3, 3] or [n_atoms] or [n_atoms, 3, 3]
                e_ext: Optional[torch.Tensor] = None,
                compute_field: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Check the input dimension
        n, d = r.shape
        assert d == 3, 'r dimension error'
        assert n == q.size(0), 'q dimension error'
        if batch is None:
            batch = torch.zeros(n, dtype=torch.int64, device=r.device)

        unique_batches = torch.unique(batch)  # Get unique batch indices

        results: List[torch.Tensor] = []
        q_induced_results: List[torch.Tensor] = []
        u_induced_results: List[torch.Tensor] = []
        for i in unique_batches.long():
            mask = batch == i  # Create a mask for the i-th configuration
            # Calculate the potential energy for the i-th configuration
            r_raw_now, q_now = r[mask], q[mask]

            u_now = u[mask] if u is not None else None
            quad_now = quad[mask] if quad is not None else None
            kappa_now = kappa[mask] if kappa is not None else None
            alpha_now = alpha[mask] if alpha is not None else None
            box_now = cell[i] if cell is not None else None # Get the box for the i-th configuration

            # check if the box is periodic or not
            if box_now is None or torch.linalg.det(box_now) < 1e-6:
                # the box is not periodic, we use the direct sum
                result = self.compute_potential_realspace(r_raw=r_raw_now, q=q_now, u=u_now,
                                                          quad=quad_now,
                                                          kappa=kappa_now,
                                                          alpha=alpha_now,
                                                          compute_field=compute_field,
                                                          e_ext=e_ext,
                                                          )
            else:
                # the box is periodic, we use the reciprocal sum
                result = self.compute_potential_triclinic(r_raw=r_raw_now, q=q_now,
                                                          cell_now=box_now, u=u_now, quad=quad_now,
                                                          kappa=kappa_now, alpha=alpha_now,
                                                          compute_field=compute_field,
                                                          e_ext=e_ext,
                                                          )
            results.append(result['pot'])
            q_induced_results.append(result['q_induced'])
            u_induced_results.append(result['u_induced'])

        return torch.cat(results), torch.cat(q_induced_results), torch.cat(u_induced_results)

    def compute_potential_realspace(self, r_raw, q,
                                    u: Optional[torch.Tensor]=None,
                                    quad: Optional[torch.Tensor]=None,
                                    kappa: Optional[torch.Tensor]=None,
                                    alpha: Optional[torch.Tensor]=None,
                                    compute_field: bool=False,
                                    e_ext: Optional[torch.Tensor]=None
                                    ):

        # this is 1/(4pi epsilon_0)
        norm_const = self.norm_factor / self.twopi

        if q.dim() == 1:
            one_dim_input = True
            q = q.unsqueeze(1)
        else:
            one_dim_input = False
        q = q.to(r_raw.dtype)
        n_node, n_q = q.shape
        device = r_raw.device

        e_field = torch.zeros((n_node, n_q, 3), device=device, dtype=r_raw.dtype)
        q_induced = torch.zeros((n_node, n_q), device=device, dtype=r_raw.dtype)
        u_induced = torch.zeros((n_node, n_q, 3), device=device, dtype=r_raw.dtype)

        if u is not None:
            u = u.to(r_raw.dtype)
            if u.dim() == 2 and u.shape[1] == 3:
                u = u.unsqueeze(1)
            assert u.shape == (n_node, n_q, 3), 'u dimension error'

        if quad is not None:
            quad = quad.to(r_raw.dtype)
            if quad.dim() == 3 and quad.shape[1] == 3:
                quad = quad.unsqueeze(1)
            assert quad.shape == (n_node, n_q, 3, 3), 'quad dimension error'

        # f_qu/f_uu also feed the charge-induced e_field when compute_field
        # or alpha is set, so compute_u covers those cases too.
        compute_u = (u is not None) or compute_field or (alpha is not None)
        compute_Q = quad is not None
        f_qq, f_qu, f_uu, f_Qu, f_QQ = make_kernels(
            r_raw, self.sigma, norm_const,
            compute_u=compute_u,
            compute_Q=compute_Q,
        )

        # electric potential at r_j due to q at r_i, sum over i
        e_phi = torch.einsum('iq,ij->jq', q, f_qq)
        pot = 0.5 * torch.einsum('iq,iq->q', e_phi, q)

        if u is not None:
            assert f_qu is not None
            e_phi_u = torch.einsum('iqc,ijc->jq', u, f_qu)
            e_phi = e_phi + e_phi_u
            pot_qu = torch.einsum('iq,iq->q', e_phi_u, q)
            pot += pot_qu

            # field at j induced by dipoles at all i (kernel excludes i=j via mask_off)
            assert f_uu is not None
            E_u = torch.einsum('ijcd,iqc->jqd', f_uu, u)
            pot_uu = -0.5 * torch.einsum('iqc,iqc->q', u, E_u)
            pot += pot_uu
        else:  # for torchscript compatibility
            E_u = torch.zeros((n_node, n_q, 3), device=device, dtype=r_raw.dtype)

        if quad is not None:
            # potential and field at j induced by quadrupoles at all i
            # (sign convention matches the triclinic structure factor S_Q = -(1/2)(k·Q·k))
            assert f_uu is not None
            e_phi_Q = 0.5 * torch.einsum('iqab,ijab->jq', quad, f_uu)
            assert f_Qu is not None
            E_Q = 0.5 * torch.einsum('iqab,ijabc->jqc', quad, f_Qu)
            e_phi = e_phi + e_phi_Q

            pot_Qq = torch.einsum('iq,iq->q', q, e_phi_Q)
            assert f_QQ is not None
            pot_QQ = 0.125 * torch.einsum('iqab,ijabcd,jqcd->q', quad, f_QQ, quad)
            pot = pot + pot_Qq + pot_QQ
            if u is not None:
                pot_Qu = -torch.einsum('iqc,iqc->q', u, E_Q)
                pot = pot + pot_Qu
        else:  # for torchscript compatibility
            E_Q = torch.zeros((n_node, n_q, 3), device=device, dtype=r_raw.dtype)

        # because this realspace sum already removed self-interaction, we need to add it back if needed
        # note this is the *opposite* behavior of triclinic -- triclinic gets the self-interaction automatically
        if not self.remove_self_interaction:
            pot += (q ** 2).sum(dim=0) / (self.sigma * self.twopi**1.5) * self.norm_factor
            e_phi = e_phi + q * (2 / (self.sigma * self.twopi**1.5)) * self.norm_factor
            if u is not None:
                pot += (u**2).sum(dim=(0,2)) / ( 3 * self.sigma**3. * self.twopi**1.5) * self.norm_factor
                a = 1.0 / (self.sigma * (2.0 ** 0.5))
                c_self = (4.0 / (3.0 * torch.pi**0.5)) * (a**3) * norm_const
                E_u = E_u - c_self * u
            if quad is not None:
                pot += (quad**2).sum(dim=(0,2,3)) / (10 * self.sigma**5. * self.twopi**1.5) * self.norm_factor

        # for computing induced charges
        if kappa is not None:
            q_induced = self._get_induced_q(e_phi, kappa)
            pot_induced = 0.5 * (e_phi * q_induced).sum(dim=0)
            pot += pot_induced

        # for computing electric field
        if compute_field or alpha is not None:
            assert f_qu is not None
            e_field = torch.einsum('iq,ijc->jqc', q, f_qu)

            if u is not None:
                e_field = e_field + E_u

            if quad is not None:
                e_field = e_field + E_Q

            if alpha is not None:
                u_induced = self._get_induced_u(e_field, alpha, e_ext)
                pot_u_induced = - 0.5 * (e_field * u_induced).sum(dim=(0,2)) # [n_q]
                pot += pot_u_induced

        output = {
                 'pot': pot.sum().view(-1),
                 'q_induced': q_induced.squeeze(dim=1) if one_dim_input else q_induced,
                 'u_induced': u_induced.squeeze(dim=1) if one_dim_input else u_induced,
                 'phi': e_phi,
                 'field': e_field,
                 }
        return output

    def _get_induced_q(self, e_phi, kappa):
        if kappa.dim() == 1:
            kappa = kappa.unsqueeze(1)
        assert kappa.dim() == 2, 'kappa dimension error'
        q_induced = - kappa * e_phi # [n, n_q]
        return q_induced

    def _get_induced_u(self, e_field, alpha, e_ext: Optional[torch.Tensor]=None):
        if e_ext is not None:
            e_field = e_field + e_ext[None,None,:]
        if alpha.dim() == 1 or (alpha.dim() == 3 and alpha.shape[1:3] == (3,3)):
            alpha = alpha.unsqueeze(1)
        if alpha.dim() == 2:
            u_induced = e_field * alpha.unsqueeze(2) # [n, n_q, 3]
        elif alpha.dim() == 4 and alpha.shape[2:4] == (3,3):
            # e_field: [n, n_q, 3], alpha: [n, n_q, 3, 3]
            u_induced = torch.einsum('iqc,iqcd->iqd', e_field, alpha)
        else:
            raise ValueError('alpha dimension error')
        return u_induced

    def _get_epsilon_r(self, alpha, volume):
        epsilon_0 = 0.00552635  # e^2 eV^{-1} A^{-1}
        if alpha.dim() == 1 or (alpha.dim() == 3 and alpha.shape[1:3] == (3,3)):
            alpha = alpha.unsqueeze(1)
        if alpha.dim() == 2: # isotropic alpha
            epsilon_r = alpha.sum(dim=0) / volume / epsilon_0 + 1.
        elif alpha.dim() == 4 and alpha.shape[2:4] == (3,3): # anisotropic alpha
            epsilon_r = torch.einsum('iqcc->q', alpha) / 3. / volume / epsilon_0 + 1.
        else:
            raise ValueError('alpha dimension error')
        return epsilon_r

    # Triclinic box(could be orthorhombic)
    def compute_potential_triclinic(self, r_raw, q, cell_now,
                                    u: Optional[torch.Tensor]=None,
                                    quad: Optional[torch.Tensor]=None,
                                    kappa:Optional[torch.Tensor]=None,
                                    alpha:Optional[torch.Tensor]=None,
                                    compute_potential:bool =False,
                                    compute_field: bool=False,
                                    e_ext: Optional[torch.Tensor]=None):
        device = r_raw.device
        if q.dim() == 1:
            one_dim_input = True
            q = q.unsqueeze(1)
        else:
            one_dim_input = False
        n_node, n_q = q.shape

        # pre-allocate tensors for torchscript compatibility
        e_phi = torch.zeros((n_node, n_q), device=device, dtype=r_raw.dtype)
        e_field = torch.zeros((n_node, n_q, 3), device=device, dtype=r_raw.dtype)
        q_induced = torch.zeros((n_node, n_q), device=device, dtype=r_raw.dtype)
        u_induced = torch.zeros((n_node, n_q, 3), device=device, dtype=r_raw.dtype)

        if u is not None:
            u = u.to(r_raw.dtype)
            if u.dim() == 2 and u.shape[1] == 3:
                u = u.unsqueeze(1)
            assert u.shape == (n_node, n_q, 3), 'u dimension error'

        if quad is not None:
            if quad.dim() == 3 and quad.shape[1] == 3:
                quad = quad.unsqueeze(1)
            assert quad.shape == (n_node, n_q, 3, 3), 'quad dimension error'

        volume = torch.det(cell_now)
        cell_inv = torch.linalg.inv(cell_now)
        G = 2 * torch.pi * cell_inv.T  # Reciprocal lattice vectors [3,3], G = 2π(M^{-1}).T

        if alpha is not None and hasattr(self, 'use_epsilon_r_scaling') and self.use_epsilon_r_scaling:
            epsilon_r = self._get_epsilon_r(alpha, volume) #[n_q]
        else:
            epsilon_r = torch.ones(n_q, device=device, dtype=r_raw.dtype)

        # max Nk for each axis
        norms = torch.norm(cell_now, dim=1)
        Nk = [max(1, int(n.item() / self.dl)) for n in norms]
        n1 = torch.arange(-Nk[0], Nk[0] + 1, device=device)
        n2 = torch.arange(-Nk[1], Nk[1] + 1, device=device)
        n3 = torch.arange(-Nk[2], Nk[2] + 1, device=device)

        # Create nvec grid and compute k vectors
        nvec = torch.stack(torch.meshgrid(n1, n2, n3, indexing="ij"), dim=-1).reshape(-1, 3).to(G.dtype)
        kvec = nvec @ G  # [N_total, 3]

        # Apply k-space cutoff and filter
        k_sq = torch.sum(kvec ** 2, dim=1)
        mask = (k_sq > 0) & (k_sq <= self.k_sq_max)
        kvec = kvec[mask] # [M, 3]
        k_sq = k_sq[mask] # [M]
        nvec = nvec[mask] # [M, 3]

        # Determine symmetry factors (handle hemisphere to avoid double-counting)
        # Include nvec if first non-zero component is positive
        non_zero = (nvec != 0).to(torch.int)
        first_non_zero = torch.argmax(non_zero, dim=1)
        sign = torch.gather(nvec, 1, first_non_zero.unsqueeze(1)).squeeze()
        hemisphere_mask = (sign > 0) | ((nvec == 0).all(dim=1))
        kvec = kvec[hemisphere_mask]
        k_sq = k_sq[hemisphere_mask]
        factors = torch.where((nvec[hemisphere_mask] == 0).all(dim=1), 1.0, 2.0)

        # Compute structure factor S(k), Σq*e^(ikr)
        k_dot_r = torch.matmul(r_raw, kvec.T)  # [n, M]
        # exp_ikr = torch.exp(1j * k_dot_r) # [n, M]
        # S_k = (q.unsqueeze(2) * exp_ikr.unsqueeze(1)).sum(dim=0) # [n_q, M]
        cos_kr = torch.cos(k_dot_r) # [n, M]
        sin_kr = torch.sin(k_dot_r) # [n, M]
        S_k_real = (q.unsqueeze(2) * cos_kr.unsqueeze(1)).sum(dim=0) # [n_q, M]
        S_k_imag = (q.unsqueeze(2) * sin_kr.unsqueeze(1)).sum(dim=0) # [n_q, M]

        if u is not None:
            uk = u @ kvec.T # [n, n_q, 3] @ [M, 3] -> [n_node, n_q, M]
            S_k_real_u = - (uk * sin_kr.unsqueeze(1)).sum(dim=0) # [n_q, M]
            S_k_real = S_k_real + S_k_real_u
            S_k_imag_u = (uk * cos_kr.unsqueeze(1)).sum(dim=0)
            S_k_imag = S_k_imag + S_k_imag_u

        if quad is not None:
            qk2 = torch.einsum("mi,ncij,mj->ncm",kvec, quad, kvec)
            S_k_real_Q = -0.5 * (qk2 * cos_kr.unsqueeze(1)).sum(dim=0)
            S_k_real = S_k_real + S_k_real_Q
            S_k_imag_Q = -0.5 * (qk2 * sin_kr.unsqueeze(1)).sum(dim=0)
            S_k_imag = S_k_imag + S_k_imag_Q

        S_k_sq = S_k_real**2 + S_k_imag**2  # [n_q, M]

        # Compute kfac,  exp(-σ^2/2 k^2) / k^2 for exponent = 1
        kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq

        # Compute potential energy, (2π/volume)* sum(factors * kfac * |S(k)|^2)
        pot = (factors * kfac * S_k_sq).sum(dim=1) / volume * self.norm_factor # [n_q]

        # Remove self-interaction if applicable
        if self.remove_self_interaction:
            pot -= torch.sum(q**2, dim=0) / (self.sigma * self.twopi**1.5) * self.norm_factor
            if u is not None:
                pot -= torch.sum(u**2, dim=(0,2)) / ( 3 * self.sigma**3. * self.twopi**1.5) * self.norm_factor
            if quad is not None:
                pot -= torch.sum(quad**2, dim=(0,2,3)) / (10 * self.sigma**5. * self.twopi**1.5) * self.norm_factor

        # for computing electric field or potential
        if compute_field or kappa is not None or alpha is not None:
            #S_k = S_k_real + 1j * S_k_imag
            #exp_ikr = cos_k_dot_r + 1j * sin_k_dot_r
            # sk_field = 2 * kfac * torch.conj(S_k)   # [n_q, M]
            prefactor = (factors * 2.0 * kfac) / volume * self.norm_factor # [M]
        else:
            prefactor = torch.zeros(kvec.shape[0], device=device, dtype=kvec.dtype)

        # for computing electric potential
        if compute_potential or kappa is not None:
            # real part of exp(-ik*r) * S(k) is the contribution to the potential,
            # Real part -> cos(k*r)*S_real + sin(k*r)*S_imag
            term_real = S_k_real.unsqueeze(0) * cos_kr.unsqueeze(1) + S_k_imag.unsqueeze(0) * sin_kr.unsqueeze(1) # [n, n_q, M]
            e_phi = (prefactor.unsqueeze(0) * term_real).sum(dim=2) # [n, n_q]

            if self.remove_self_interaction:
                e_phi -= q * (2 / (self.sigma * self.twopi**1.5)) * self.norm_factor # [n, n_q]

            if kappa is not None: # compute induced charges
                q_induced = self._get_induced_q(e_phi, kappa)
                pot_induced = 0.5 * (e_phi * q_induced).sum(dim=0) # [n_q]
                pot += pot_induced

        # for computing electric field
        if compute_field or alpha is not None:
            # imaginary part of exp(-ik*r) * S(k) contributes to the field
            # Imaginary part -> cos(k*r)*S_imag - sin(k*r)*S_real
            term_imag = S_k_real.unsqueeze(0) * sin_kr.unsqueeze(1) - S_k_imag.unsqueeze(0) * cos_kr.unsqueeze(1) # [n, n_q, M]
            e_field = (prefactor.unsqueeze(0).unsqueeze(0).unsqueeze(3)
                       * term_imag.unsqueeze(3) * kvec.unsqueeze(0).unsqueeze(0)).sum(dim=2) # [n, n_q, 3]

            if self.remove_self_interaction and u is not None:
                a = 1.0 / (self.sigma * (2.0 ** 0.5))
                c_self = (4.0 / (3.0 * torch.pi**0.5)) * (a**3) / self.twopi * self.norm_factor
                e_field += c_self * u

            # compute induced dipoles
            if alpha is not None:
                u_induced = self._get_induced_u(e_field, alpha, e_ext)
                pot_induced = - 0.5 * (e_field * u_induced).sum(dim=(0,2)) # [n_q]
                pot += pot_induced

        output = {
                 'pot': pot.sum().view(-1), # sum over the energy contributions from different nq channels
                 'q_induced': q_induced.squeeze(dim=1) if one_dim_input else q_induced,
                 'u_induced': u_induced.squeeze(dim=1) if one_dim_input else u_induced,
                 'phi': e_phi,
                 'field': e_field,
                 'epsilon_r': epsilon_r,
                 }
        return output

    def __repr__(self):
        return f"Ewald(dl={self.dl}, sigma={self.sigma}, remove_self_interaction={self.remove_self_interaction})"







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
        if quad is not None or kappa is not None or alpha is not None or e_ext is not None or compute_field:
            raise NotImplementedError(
                "the vectorized Ewald currently supports latent charges and dipoles; "
                "omit 'is_periodic' to fall back to the legacy module for quadrupoles, "
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

        if not self.is_periodic: # non-periodic
            assert cell is not None, 'fake cell needed for non-periodic case (ex. torch.zeros(n_batch, 3,3))'
            results = self.compute_potential_realspace(r, q, cell, batch, u=u)
        else: # periodic
            results = self.compute_potential_triclinic(r, q, cell, batch, u=u)

        # same (pot, q_induced, u_induced) interface as the legacy module
        q_induced = torch.zeros_like(q)
        u_induced = torch.zeros((n, q.shape[1], 3), device=r.device, dtype=r.dtype)
        return results, q_induced, u_induced
    


    def compute_potential_realspace(self, r, q, cell, batch,
                                    u: Optional[torch.Tensor] = None):
        """
        Realspace (non-periodic) Ewald over an [N, N] pair grid (N = total
        atoms).

        Masks cross-batch pairs, cost is O(N^2). Handles latent monopoles and,
        when `u` is given, latent dipoles, reproducing the charge-dipole and
        dipole-dipole terms of the loop-based reference module.
        """
        device = r.device
        dtype = r.dtype
        N = r.shape[0]
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

        if u is not None:
            u = u.to(dtype=dtype)
            sqrt_pi = torch.pi ** 0.5
            rinv2 = rinv * rinv
            rinv3 = rinv2 * rinv
            # Gaussian damping: its argument depends on the positions, so route it
            # through the export-safe exp (see _ExpSaveInput).
            gauss = _ExpSaveInput.apply(-(a * a) * dist_sq)              # [N, N]
            s1 = erf_val * rinv3 - (2.0 * a / sqrt_pi) * gauss * rinv2
            s2 = (3.0 * erf_val * rinv3
                  - (6.0 * a / sqrt_pi) * gauss * rinv2
                  - (4.0 * a ** 3 / sqrt_pi) * gauss)

            u_i = u[pair_i]                                              # [N, N, n_q, 3]
            u_j = u[pair_j_safe]                                         # [N, N, n_q, 3]
            q_j = q[pair_j_safe]                                         # [N, N, n_q]

            r_ij_e = r_ij.unsqueeze(2)                                   # [N, N, 1, 3]

            # charge-dipole: Σ_{i≠j} q_j (u_i · f_qu),  f_qu = s1 r_ij
            ui_dot_r = (u_i * r_ij_e).sum(dim=-1)                        # [N, N, n_q]
            pot_per_pair = pot_per_pair + (s1.unsqueeze(-1) * ui_dot_r * q_j).sum(dim=-1)

            # dipole-dipole: -1/2 Σ_{i≠j} u_j · f_uu · u_i,
            # with f_uu = s2 (r_ij ⊗ r_ij)/r^2 - s1 I
            uj_dot_r = (u_j * r_ij_e).sum(dim=-1)                        # [N, N, n_q]
            ui_dot_uj = (u_i * u_j).sum(dim=-1)                          # [N, N, n_q]
            uu_term = (s2.unsqueeze(-1) * ui_dot_r * uj_dot_r * rinv2.unsqueeze(-1)
                       - s1.unsqueeze(-1) * ui_dot_uj)                   # [N, N, n_q]
            pot_per_pair = pot_per_pair - 0.5 * uu_term.sum(dim=-1)

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
                u_sq_per_atom = (u ** 2).sum(dim=(1, 2))   # [N]
                u_self_per_batch = torch.zeros(B, device=device, dtype=dtype)
                u_self_per_batch.scatter_add_(0, batch, u_sq_per_atom)
                pot_per_batch = pot_per_batch + u_self_per_batch / (
                    3.0 * self.sigma ** 3.0 * self.twopi ** (3.0 / 2.0))

        return pot_per_batch * self.norm_factor
    

    def compute_potential_triclinic(self, r, q, cell, batch,
                                    u: Optional[torch.Tensor] = None):

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

        pot_per_batch = pot_per_batch_per_q.sum(dim=1)  # [B]
        return pot_per_batch * self.norm_factor  # [B]