import torch
import torch.nn as nn
from itertools import product
from typing import Dict, Optional
import numpy as np

__all__ = ['Ewald']

class Ewald(nn.Module):
    def __init__(self,
                 dl=2.0,  # grid resolution
                 sigma=1.0,  # width of the Gaussian on each atom
                 remove_self_interaction=True,
                 norm_factor=90.0474,
                 ):
        super().__init__()
        self.dl = dl
        self.sigma = sigma
        self.sigma_sq_half = sigma ** 2 / 2.0
        self.twopi = 2.0 * torch.pi
        self.twopi_sq = self.twopi ** 2
        self.remove_self_interaction = remove_self_interaction
        # 1/2\epsilon_0, where \epsilon_0 is the vacuum permittivity
        # \epsilon_0 = 5.55263*10^{-3} e^2 eV^{-1} A^{-1}
        self.norm_factor = norm_factor
        self.k_sq_max = (self.twopi / self.dl) ** 2

    def forward(self,
                q: torch.Tensor,  # [n_atoms, n_q] or [n_atoms]
                r: torch.Tensor, # [n_atoms, 3]
                cell: torch.Tensor, # [batch_size, 3, 3]
                batch: Optional[torch.Tensor] = None,
                u: Optional[torch.Tensor] = None, # [n_atoms, n_q, 3] or [natoms, 3]
                alpha: Optional[torch.Tensor] = None, # [n_atoms, 3]
                compute_field: bool = False
                ) -> torch.Tensor:
        
        # Check the input dimension
        n, d = r.shape
        assert d == 3, 'r dimension error'
        assert n == q.size(0), 'q dimension error'
        if batch is None:
            batch = torch.zeros(n, dtype=torch.int64, device=r.device)

        unique_batches = torch.unique(batch)  # Get unique batch indices

        results = []
        for i in unique_batches:
            mask = batch == i  # Create a mask for the i-th configuration
            # Calculate the potential energy for the i-th configuration
            r_raw_now, q_now = r[mask], q[mask]
            if u is not None:
                u_now = u[mask]
            else:
                u_now = None
            if alpha is not None:
                alpha_now = alpha[mask]
            else:
                alpha_now = None

            if cell is not None:
                box_now = cell[i]  # Get the box for the i-th configuration
            
            # check if the box is periodic or not
            if cell is None or torch.linalg.det(box_now) < 1e-6:
                # the box is not periodic, we use the direct sum
                pot = self.compute_potential_realspace(r_raw_now, q_now, u=u_now, alpha=alpha_now, compute_field=compute_field)
            else:
                # the box is periodic, we use the reciprocal sum
                pot = self.compute_potential_triclinic(r_raw_now, q_now, box_now, u=u_now, alpha=alpha_now, compute_field=compute_field)
            results.append(pot)

        # here we sum over all n_q dimensions
        return torch.stack(results, dim=0).sum(dim=1)

    def compute_potential_realspace(self, r_raw, q, u=None, alpha=None, compute_field=False):

        # this is 1/(4pi epsilon_0)
        norm_const = self.norm_factor / self.twopi

        if q.dim() == 1:
            # [n_node, n_q]
            q = q.unsqueeze(1)
        n_node, n_q = q.shape

        if u is not None:
            if u.dim() == 2 and u.shape[1] == 3:
                u = u.unsqueeze(1)
            assert u.shape == (n_node, n_q, 3), 'u dimension error'

        # mask out self terms cleanly (avoid eps-hacks for fields)
        mask_off = ~torch.eye(n_node, dtype=torch.bool, device=r_raw.device)

        # r_ij[i,j] = r_raw[0,j] - r_raw[i,0] = r_j - r_i
        r_ij = r_raw.unsqueeze(0) - r_raw.unsqueeze(1) # [n_node, n_node, 3]
        # Compute pairwise distances (norm of vector differences)
        r_ij_norm = torch.norm(r_ij, dim=-1) # [n_node, n_node]
        # Compute inverse distance
        rinv = torch.zeros_like(r_ij_norm)
        rinv[mask_off] = 1.0 / r_ij_norm[mask_off]
        rinv2 = rinv * rinv
        rinv3 = rinv2 * rinv

        a = 1.0 / (self.sigma * (2.0 ** 0.5))          # 1/(sqrt(2)*sigma)
        # Error function scaling for long-range interactions
        erf = torch.zeros_like(r_ij_norm)
        erf[mask_off] = torch.special.erf(r_ij_norm[mask_off] * a) # the s0 term

        # charge-charge kernel
        f_qq = erf * rinv * norm_const # [n_node, n_node]
        # Compute potential energy
        # [1, n_node, n_q] * [n_node, 1, n_q] * [n_node, n_node, 1] 
        pot_qq = q.unsqueeze(0) * q.unsqueeze(1) * f_qq.unsqueeze(2) # [n_node, n_node, n_q]
        # diagonal terms are zero
        pot = 0.5 * pot_qq.sum(dim=(0,1)) # [n_q]

        if compute_field or u is not None or alpha is not None:
            # charge-dipole kernel
            gauss = torch.zeros_like(r_ij_norm)
            gauss[mask_off] = torch.exp(-(a * r_ij_norm[mask_off]) ** 2)
            # this is actually the s1/r3 term
            s1 = erf * rinv3 - (2.0 * a / (torch.pi ** 0.5)) * gauss * rinv2 # [n_node, n_node]
            # f_qu[i, j] * q_i computes the electric field at r_j due to q at r_i
            f_qu = s1.unsqueeze(2) * r_ij * norm_const # ij\alpha = [n_node, n_node, 3]
            # print((f_qu + f_qu.transpose(0,1))[mask].abs().max()) should be zero
        if u is not None:
            # - E * u
            pot_qu = - torch.einsum('iq,ijc,jqc->q', q, f_qu, u)
            pot = pot + pot_qu

            # dipole-dipole kernel for f(r)=erf(ar)/r
            s2 = 3.0 * erf * rinv3 \
                - (6.0 * a / torch.pi**0.5) * gauss * rinv2 \
                - (4.0 * a**3 / torch.pi**0.5) * gauss
            rhat = r_ij * rinv[..., None]
            outer = rhat[..., :, None] * rhat[..., None, :] # [n,n,3,3]
            f_uu_1 = s2[:, :, None, None] * outer # ij\alpha\beta = [node, node, 3 , 3] 
            I3 = torch.eye(3, device=r_raw.device, dtype=r_raw.dtype)[None, None, :, :]      # [1,1,3,3]
            f_uu_2 = - s1[:, :, None, None] * I3 
            # f_uu[i,j,...] is for the electric field at r_j due to u_i at r_i  
            f_uu = (f_uu_1 + f_uu_2) * norm_const
            pot_uu = -0.5 * torch.einsum('iqc,ijcd,jqd->q', u, f_uu, u)
            pot += pot_uu

        # because this realspace sum already removed self-interaction, we need to add it back if needed
        if self.remove_self_interaction == False:
            pot += (q ** 2).sum(dim=0) / (self.sigma * self.twopi**(3./2.)) * self.norm_factor
            if u is not None:
                pot += (u**2).sum(dim=(0,2)) / ( 3 * self.sigma**3. * (2*torch.pi)**1.5) * self.norm_factor

        if compute_field or alpha is not None:
            E_q = torch.einsum('iq,ijc->jqc', q, f_qu)

            if u is not None:
                # Field from dipoles: E_u_j = sum_i (T_ij * u_i)
                # f_uu is the Hessian [n, n, 3, 3]
                E_u = torch.einsum('ijcd,iqc->jqd', f_uu, u)  # [n, n_q, 3]
                if self.remove_self_interaction == False: 
                    c_self = (4.0 / (3.0 * torch.pi**0.5)) * (a**3) / self.twopi * self.norm_factor # [1/length^3] / (2π)
                    E_u = E_u - c_self * u
                q_field = E_q + E_u
            else:
                q_field = E_q

            if alpha is not None:
                pot_induced = - 0.5 * ((q_field ** 2).sum(dim=2) * alpha).sum(dim=0) # [n_q]
                pot = pot + pot_induced

        if compute_field:
            return pot, q_field
        if not compute_field:
            return pot

    # Triclinic box(could be orthorhombic)
    def compute_potential_triclinic(self, r_raw, q, cell_now, u=None, alpha=None, compute_field=False):
        device = r_raw.device

        cell_inv = torch.linalg.inv(cell_now)
        G = 2 * torch.pi * cell_inv.T  # Reciprocal lattice vectors [3,3], G = 2π(M^{-1}).T

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
        if q.dim() == 1:  
            q = q.unsqueeze(1)
        # Compute potential energy
        n_node, n_q = q.shape

        if u is not None:
            if u.dim() == 2 and u.shape[1] == 3:
                u = u.unsqueeze(1)
            assert u.shape == (n_node, n_q, 3), 'u dimension error'

        """
        # for torchscript compatibility, to avoid dtype mismatch, only use real part
        cos_k_dot_r = torch.cos(k_dot_r) # [n, M]
        sin_k_dot_r = torch.sin(k_dot_r)
        S_k_real = (q.unsqueeze(2) * cos_k_dot_r.unsqueeze(1)).sum(dim=0) # [n_q, M]
        S_k_imag = (q.unsqueeze(2) * sin_k_dot_r.unsqueeze(1)).sum(dim=0)
        if u is not None:
            uk = u @ kvec.T # [n, n_q, 3] @ [M, 3] -> [n_node, n_q, M]
            # [n, n_q, 3] * [n, 1, M] -> [n_q, M]
            S_k_real_u = - (uk * sin_k_dot_r.unsqueeze(1)).sum(dim=0) # [n_q, M]
            S_k_real = S_k_real + S_k_real_u
            S_k_imag_u = (uk * cos_k_dot_r.unsqueeze(1)).sum(dim=0)
            S_k_imag = S_k_imag + S_k_imag_u
        S_k_sq = S_k_real**2 + S_k_imag**2  # [M]
        """

        exp_ikr = torch.exp(1j * k_dot_r)
        S_k = (q.unsqueeze(2) * exp_ikr.unsqueeze(1)).sum(dim=0) # [n_q, M]

        if u is not None:
            uk = u @ kvec.T
            S_k_u = 1j * (uk * exp_ikr.unsqueeze(1)).sum(dim=0) # [n_q, M]
            S_k = S_k + S_k_u

        S_k_sq = torch.real(S_k * torch.conj(S_k)) # [n_q, M]

        # Compute kfac,  exp(-σ^2/2 k^2) / k^2 for exponent = 1
        kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq
        
        # Compute potential, (2π/volume)* sum(factors * kfac * |S(k)|^2)
        volume = torch.det(cell_now)
        pot = (factors * kfac * S_k_sq).sum(dim=1) / volume * self.norm_factor # [n_q]

        # Remove self-interaction if applicable
        if self.remove_self_interaction:
            pot -= torch.sum(q**2, dim=0) / (self.sigma * (2*torch.pi)**1.5) * self.norm_factor
            if u is not None:
                pot -= torch.sum(u**2, dim=(0,2)) / ( 3 * self.sigma**3. * (2*torch.pi)**1.5) * self.norm_factor

        if compute_field or alpha is not None:
            #S_k = S_k_real + 1j * S_k_imag
            #exp_ikr = cos_k_dot_r + 1j * sin_k_dot_r
            sk_field = 2 * kfac * torch.conj(S_k)   # [n_q, M]
            q_field = torch.real(
                      -1j *
                      factors[None, None, :, None]      # (1, 1,  M, 1)
                      * exp_ikr[:, None, :, None]       # (n, 1,  M, 1)
                      * kvec[None, None, :, :]          # (1, 1,  M, 3)
                      * sk_field[None, :, :, None]      # (1, n_q, M, 1)
                      ).sum(dim=2)
            q_field = q_field / volume * self.norm_factor # [n, n_q, 3]

            if self.remove_self_interaction and u is not None:
                a = 1.0 / (self.sigma * (2.0 ** 0.5))
                c_self = (4.0 / (3.0 * torch.pi**0.5)) * (a**3) / self.twopi * self.norm_factor
                q_field = q_field + c_self * u
            if alpha is not None:
                pot_induced = - 0.5 * ((q_field ** 2).sum(dim=2) * alpha).sum(dim=0)
                pot = pot + pot_induced

        if compute_field:
            return pot, q_field
        if not compute_field:
            return pot

    def __repr__(self):
        return f"Ewald(dl={self.dl}, sigma={self.sigma}, remove_self_interaction={self.remove_self_interaction})"
