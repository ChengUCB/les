import torch
import torch.nn as nn
from itertools import product
from typing import Dict, Optional, Tuple, List
import numpy as np

__all__ = ['Ewald']

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
                kappa: Optional[torch.Tensor] = None, # [n_atoms, n_q] or [n_atoms]
                alpha: Optional[torch.Tensor] = None, # [n_atoms, n_q] or [n_atoms, n_q, 3, 3] or [n_atoms] or [n_atoms, 3, 3]
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
            kappa_now = kappa[mask] if kappa is not None else None
            alpha_now = alpha[mask] if alpha is not None else None
            box_now = cell[i] if cell is not None else None # Get the box for the i-th configuration
            
            # check if the box is periodic or not
            if box_now is None or torch.linalg.det(box_now) < 1e-6:
                # the box is not periodic, we use the direct sum
                result = self.compute_potential_realspace(r_raw=r_raw_now, q=q_now, u=u_now, 
                                                          kappa=kappa_now, alpha=alpha_now, 
                                                          compute_field=compute_field
                                                          )
            else:
                # the box is periodic, we use the reciprocal sum
                result = self.compute_potential_triclinic(r_raw=r_raw_now, q=q_now, 
                                                          cell_now=box_now, u=u_now, 
                                                          kappa=kappa_now, alpha=alpha_now, 
                                                          compute_field=compute_field
                                                          )
            results.append(result['pot'])
            q_induced_results.append(result['q_induced'])
            u_induced_results.append(result['u_induced'])

        return torch.cat(results), torch.cat(q_induced_results), torch.cat(u_induced_results)

    def compute_potential_realspace(self, r_raw, q, 
                                    u: Optional[torch.Tensor]=None, 
                                    kappa: Optional[torch.Tensor]=None, 
                                    alpha: Optional[torch.Tensor]=None, 
                                    compute_field: bool=False):

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

        # mask out self terms cleanly
        mask_off = ~torch.eye(n_node, dtype=torch.bool, device=device)

        # r_ij[i,j] = r_raw[0,j] - r_raw[i,0] = r_j - r_i
        r_ij = r_raw.unsqueeze(0) - r_raw.unsqueeze(1) # [n_node, n_node, 3]
        r_ij_norm = torch.norm(r_ij, dim=-1) # [n_node, n_node], pairwise distance
        rinv = torch.zeros_like(r_ij_norm) # inverse distance
        rinv[mask_off] = 1.0 / r_ij_norm[mask_off]
        rinv2 = rinv * rinv
        rinv3 = rinv2 * rinv

        a = 1.0 / (self.sigma * (2.0 ** 0.5))         # 1/(sqrt(2)*sigma)
        # Error function scaling for long-range interactions
        erf = torch.zeros_like(r_ij_norm)
        erf[mask_off] = torch.special.erf(r_ij_norm[mask_off] * a) # the s0 term

        # charge-charge kernel, diagonal terms are zero
        f_qq = (erf * rinv * norm_const).to(r_raw.dtype) # [n_node, n_node]
        # electric potential at r_j due to q at r_i, sum over i
        e_phi = torch.einsum('iq,ij->jq', q, f_qq) # [n_node, n_q]
        pot = 0.5 * torch.einsum('iq,iq->q', e_phi, q) # [n_q], potential energy

        # charge-dipole kernel
        if compute_field or u is not None or alpha is not None:
            gauss = torch.exp(-(a * r_ij_norm) ** 2) * mask_off
            # this is actually the s1/r3 term
            s1 = erf * rinv3 - (2.0 * a / (torch.pi ** 0.5)) * gauss * rinv2 # [n_node, n_node]
            # f_qu[i, j] * q_i computes the electric field at r_j due to q at r_i
            # f_qu[i, j] * u_i is also equal to the potential at r_j due to u at r_i
            f_qu = s1.unsqueeze(2) * r_ij * norm_const # ij\alpha = [n_node, n_node, 3]
        else: # for torchscript compatibility
            f_qu = torch.zeros((n_node, n_node, 3), device=device, dtype=r_raw.dtype)
            gauss = torch.zeros_like(r_ij_norm)
            s1 = torch.zeros_like(r_ij_norm)


        if u is not None:
            # pot_qu = - torch.einsum('iq,ijc,jqc->q', q, f_qu, u) # - E * u
            e_phi_u = torch.einsum('iqc,ijc->jq', u, f_qu) # [n_node, n_q]
            e_phi = e_phi + e_phi_u
            pot_qu = torch.einsum('iq,iq->q', e_phi_u, q) # [n_q]
            pot += pot_qu

            # dipole-dipole kernel for f(r)=erf(ar)/r
            s2 = 3.0 * erf * rinv3 \
                - (6.0 * a / torch.pi**0.5) * gauss * rinv2 \
                - (4.0 * a**3 / torch.pi**0.5) * gauss
            rhat = r_ij * rinv[..., None]
            outer = rhat[..., :, None] * rhat[..., None, :] # [n,n,3,3]
            f_uu_1 = s2[:, :, None, None] * outer # ij\alpha\beta = [node, node, 3 , 3] 
            I3 = torch.eye(3, device=device, dtype=r_raw.dtype)[None, None, :, :]      # [1,1,3,3]
            f_uu_2 = - s1[:, :, None, None] * I3 
            # f_uu[i,j,...] is for the electric field at r_j due to u_i at r_i  
            f_uu = (f_uu_1 + f_uu_2) * norm_const
            pot_uu = -0.5 * torch.einsum('iqc,ijcd,jqd->q', u, f_uu, u)
            pot += pot_uu
        else: # for torchscript compatibility
            f_uu = torch.zeros((n_node, n_node, 3, 3), device=device, dtype=r_raw.dtype)
            

        # because this realspace sum already removed self-interaction, we need to add it back if needed
        if not self.remove_self_interaction:
            pot += (q ** 2).sum(dim=0) / (self.sigma * self.twopi**1.5) * self.norm_factor
            e_phi += q * (2 / (self.sigma * self.twopi**1.5)) * self.norm_factor  # [n, n_q]
            if u is not None:
                pot += (u**2).sum(dim=(0,2)) / ( 3 * self.sigma**3. * self.twopi**1.5) * self.norm_factor

        # for computing induced charges
        if kappa is not None:
            q_induced = self._get_induced_q(e_phi, kappa)
            pot_induced = 0.5 * (e_phi * q_induced).sum(dim=0) # [n_q]
            pot += pot_induced

        # for computing electric field
        if compute_field or alpha is not None:
            e_field = torch.einsum('iq,ijc->jqc', q, f_qu) # [n, n_q, 3]

            if u is not None:
                # Field from dipoles: E_u_j = sum_i (T_ij * u_i)
                # f_uu is the Hessian [n, n, 3, 3]
                E_u = torch.einsum('ijcd,iqc->jqd', f_uu, u)  # [n, n_q, 3]
                if not self.remove_self_interaction: 
                    c_self = (4.0 / (3.0 * torch.pi**0.5)) * (a**3) * norm_const
                    E_u = E_u - c_self * u
                e_field = e_field + E_u

            # compute induced dipoles
            if alpha is not None:
                u_induced = self._get_induced_u(e_field, alpha)
                pot_u_induced = - 0.5 * (e_field * u_induced).sum(dim=(0,2)) # [n_q]
                pot += pot_u_induced

        output = {
                 'pot': pot.sum().view(-1), # sum over the energy contributions from different nq channels
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

    def _get_induced_u(self, e_field, alpha):
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
            epsilon_r = alpha.sum(axis=0) / volume / epsilon_0 + 1.
        elif alpha.dim() == 4 and alpha.shape[2:4] == (3,3): # anisotropic alpha
            epsilon_r = torch.einsum('iqcc->q', alpha) / 3. / volume / epsilon_0 + 1.
        else:
            raise ValueError('alpha dimension error')
        return epsilon_r

    # Triclinic box(could be orthorhombic)
    def compute_potential_triclinic(self, r_raw, q, cell_now, 
                                    u: Optional[torch.Tensor]=None, 
                                    kappa: Optional[torch.Tensor]=None, 
                                    alpha: Optional[torch.Tensor]=None, 
                                    compute_potential: bool=False, compute_field: bool=False):
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

        volume = torch.det(cell_now)
        cell_inv = torch.linalg.inv(cell_now)
        G = 2 * torch.pi * cell_inv.T  # Reciprocal lattice vectors [3,3], G = 2π(M^{-1}).T

        # use epsilon_r to scale the potential and field if alpha is provided
        if alpha is not None and hasattr(self, 'use_epsilon_r_scaling') and self.use_epsilon_r_scaling:
            epsilon_r = self._get_epsilon_r(alpha, volume) #[n_q]
            #print(f"Using epsilon_r scaling with epsilon_r = {epsilon_r}")
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
                u_induced = self._get_induced_u(e_field, alpha)
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
