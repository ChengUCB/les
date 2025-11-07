import torch
import torch.nn as nn
from itertools import product
from typing import Dict, Optional
import numpy as np

__all__ = ['Ewald', 'Ewald_develop']

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
                q: torch.Tensor,  # [n_atoms, n_q]
                r: torch.Tensor, # [n_atoms, 3]
                cell: torch.Tensor, # [batch_size, 3, 3]
                batch: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        
        if q.dim() == 1:
            q = q.unsqueeze(1)

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
            if cell is not None:
                box_now = cell[i]  # Get the box for the i-th configuration
            
            # check if the box is periodic or not
            if cell is None or torch.linalg.det(box_now) < 1e-6:
                # the box is not periodic, we use the direct sum
                pot = self.compute_potential_realspace(r_raw_now, q_now)
            else:
                # the box is periodic, we use the reciprocal sum
                pot = self.compute_potential_triclinic(r_raw_now, q_now, box_now)
            results.append(pot)

        return torch.stack(results, dim=0).sum(dim=1)

    def compute_potential_realspace(self, r_raw, q):
        # Compute pairwise distances (norm of vector differences)
        # Add epsilon for safe Hessian compute
        epsilon = 1e-6
        r_ij = r_raw.unsqueeze(0) - r_raw.unsqueeze(1)
        torch.diagonal(r_ij).add_(epsilon)
        r_ij_norm = torch.norm(r_ij, dim=-1)
 
        # Error function scaling for long-range interactions
        convergence_func_ij = torch.special.erf(r_ij_norm / self.sigma / (2.0 ** 0.5))
   
        # Compute inverse distance
        r_p_ij = 1.0 / (r_ij_norm)

        if q.dim() == 1:
            # [n_node, n_q]
            q = q.unsqueeze(1)
    
        # Compute potential energy
        n_node, n_q = q.shape
        # [1, n_node, n_q] * [n_node, 1, n_q] * [n_node, n_node, 1] * [n_node, n_node, 1]
        pot = q.unsqueeze(0) * q.unsqueeze(1) * r_p_ij.unsqueeze(2) * convergence_func_ij.unsqueeze(2)

        #Exclude diagonal terms from energy
        mask = ~torch.eye(pot.shape[0], device=pot.device).to(torch.bool).unsqueeze(-1)
        mask = torch.vstack([mask.transpose(0,-1)]*pot.shape[-1]).transpose(0,-1)
        pot = pot[mask].sum().view(-1) / self.twopi / 2.0

        # because this realspace sum already removed self-interaction, we need to add it back if needed
        if self.remove_self_interaction == False:
            pot += torch.sum(q ** 2) / (self.sigma * self.twopi**(3./2.))
    
        return pot * self.norm_factor
 
    # Triclinic box(could be orthorhombic)
    def compute_potential_triclinic(self, r_raw, q, cell_now):
        device = r_raw.device

        cell_inv = torch.linalg.inv(cell_now)
        G = 2 * torch.pi * cell_inv.T  # Reciprocal lattice vectors [3,3], G = 2π(M^{-1}).T
        #print('G', G.type())

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

         #for torchscript compatibility, to avoid dtype mismatch, only use real part
        cos_k_dot_r = torch.cos(k_dot_r)
        sin_k_dot_r = torch.sin(k_dot_r)
        S_k_real = (q.unsqueeze(2) * cos_k_dot_r.unsqueeze(1)).sum(dim=0)
        S_k_imag = (q.unsqueeze(2) * sin_k_dot_r.unsqueeze(1)).sum(dim=0)
        S_k_sq = S_k_real**2 + S_k_imag**2  # [M]

        # Compute kfac,  exp(-σ^2/2 k^2) / k^2 for exponent = 1
        kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq
        
        # Compute potential, (2π/volume)* sum(factors * kfac * |S(k)|^2)
        volume = torch.det(cell_now)
        pot = (factors * kfac * S_k_sq).sum(dim=1) / volume

        # Remove self-interaction if applicable
        if self.remove_self_interaction:
            pot -= torch.sum(q**2) / (self.sigma * (2*torch.pi)**1.5)

        return pot * self.norm_factor

    def __repr__(self):
        return f"Ewald(dl={self.dl}, sigma={self.sigma}, remove_self_interaction={self.remove_self_interaction})"


class Ewald_develop(nn.Module):
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
                q: torch.Tensor,  # [n_atoms, n_q]
                r: torch.Tensor, # [n_atoms, 3]
                cell: torch.Tensor, # [batch_size, 3, 3]
                batch: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        
        if q.dim() == 1:
            q = q.unsqueeze(1)

        # Check the input dimension
        n, d = r.shape
        assert d == 3, 'r dimension error'
        assert n == q.size(0), 'q dimension error'
        if batch is None:
            batch = torch.zeros(n, dtype=torch.int64, device=r.device)

        unique_batches = torch.unique(batch)  # Get unique batch indices

        results = self.potential_full_ewald_batched(
            pos=r,
            q=q,
            cell=cell,
            batch=batch,
        )

        return results

 
    def potential_full_ewald_batched(
        self,
        pos: torch.Tensor,
        q: torch.Tensor,
        cell: torch.Tensor,
        dl: float = 2.0,
        sigma: float = 1.0,
        epsilon: float = 1e-6,
        return_bec: bool = False,
        batch: torch.Tensor | None = None,
    ):
        """
        Get the potential energy for each atom in the batch using Ewald summation.
        Takes:
            pos: position matrix of shape (n_atoms, 3)
            q: charge vector of shape (n_atoms, 1)
            cell: cell matrix of shape (batch_size, 3, 3)
            sigma: sigma parameter for the error function
            epsilon: epsilon parameter for the error function
            dl: grid resolution
            k_sq_max: maximum k^2 value
            twopi: 2 * pi
            max_num_neighbors: maximum number of neighbors for each atom
            batch: batch vector of shape (n_atoms,)
        Returns:
            potential_dict: dictionary of potential energy for each atom
        """
        
        device = pos.device
        sigma_sq_half = sigma ** 2 / 2.0
        k_sq_max = (2 * np.pi / dl) ** 2
        norm_factor = 90.0474
        
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.int64, device=device)
        
        # Compute reciprocal lattice vectors for each batch
        cell_inv = torch.linalg.inv(cell)  # [B, 3, 3]
        G = 2 * torch.pi * cell_inv.transpose(-2, -1)  # [B, 3, 3]

        # Determine maximum Nk for each axis in each batch
        norms = torch.norm(cell, dim=2)  # [B, 3]
        Nk = torch.clamp(torch.floor(norms / dl).int(), min=1)  # [B, 3]
        
        # Pre-allocate maximum grid size to reuse memory
        # data-dependent, -> problem with torch.compile?
        max_Nk = Nk.max()
        max_grid_size = (2 * max_Nk + 1) ** 3
        
        # Pre-allocate reusable tensors
        nvec_buffer = torch.empty((max_grid_size, 3), device=device, dtype=G.dtype)
        kvec_buffer = torch.empty((max_grid_size, 3), device=device, dtype=G.dtype)
        k_sq_buffer = torch.empty(max_grid_size, device=device, dtype=G.dtype)
        
        # Process each batch separately to minimize memory usage
        unique_batches = torch.unique(batch)
        result_potentials = torch.zeros(pos.shape[0], device=device)
        
        result = []
        for b_idx in unique_batches:
            # Get atoms and cell for this batch
            atom_mask = batch == b_idx
            pos_b = pos[atom_mask]  # [n_atoms_b, 3]
            q_b = q[atom_mask]  # [n_atoms_b, 1] or [n_atoms_b]
            
            if q_b.dim() == 3:
                q_b = q_b.squeeze(-1)
            
            # Generate k-vectors only for this batch
            G_b = G[b_idx]  # [3, 3]
            Nk_b = Nk[b_idx]  # [3]
            
            # Calculate actual grid size for this batch
            grid_size = (2 * Nk_b[0] + 1) * (2 * Nk_b[1] + 1) * (2 * Nk_b[2] + 1)
            
            # Use views of pre-allocated buffers instead of creating new tensors
            nvec = nvec_buffer[:grid_size]
            kvec = kvec_buffer[:grid_size]
            k_sq = k_sq_buffer[:grid_size]
            
            # Generate grid indices efficiently using broadcasting
            n1_range = torch.arange(-Nk_b[0], Nk_b[0] + 1, device=device, dtype=G_b.dtype) # [2*Nk_b[0]+1]
            n2_range = torch.arange(-Nk_b[1], Nk_b[1] + 1, device=device, dtype=G_b.dtype) # [2*Nk_b[1]+1]
            n3_range = torch.arange(-Nk_b[2], Nk_b[2] + 1, device=device, dtype=G_b.dtype) # [2*Nk_b[2]+1]
            
            # Use meshgrid but reshape directly into nvec buffer
            n1_grid, n2_grid, n3_grid = torch.meshgrid(n1_range, n2_range, n3_range, indexing="ij")
            nvec[:, 0] = n1_grid.flatten()
            nvec[:, 1] = n2_grid.flatten()
            nvec[:, 2] = n3_grid.flatten()
            
            # Compute k vectors in-place using matrix multiplication
            torch.mm(nvec, G_b, out=kvec) # [grid_size, 3]
            
            # Compute k_sq in-place
            torch.sum(kvec ** 2, dim=1, out=k_sq) # [grid_size]
            
            # Apply filters using boolean indexing (creates views, not copies)
            valid_mask = (k_sq > 0) & (k_sq <= k_sq_max) # [grid_size] bool
            valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0] # [M], masked indices

            assert valid_indices.numel() != 0, "No valid k-vectors found for batch {}".format(b_idx)
                
            # Use advanced indexing to get valid vectors (still creates copies, but smaller)
            nvec_valid = nvec[valid_indices] # [M, 3]
            kvec_valid = kvec[valid_indices] # [M, 3]
            k_sq_valid = k_sq[valid_indices] # [M]
            
            # Hemisphere masking - use existing logic but optimize
            non_zero_mask = nvec_valid != 0 # [M, 3] bool
            has_non_zero = non_zero_mask.any(dim=1) # [M] bool
            first_non_zero_idx = torch.argmax(non_zero_mask.float(), dim=1) # [M]
            
            # Use gather efficiently
            sign = torch.gather(nvec_valid, 1, first_non_zero_idx.unsqueeze(1)).squeeze() # [M]
            hemisphere_mask = (sign > 0) | ~has_non_zero # [M] bool
            
            # Final filtering
            final_indices = torch.nonzero(hemisphere_mask, as_tuple=True)[0] # [M']

            assert final_indices.numel() != 0, "No k-vectors remain after hemisphere filtering for batch {}".format(b_idx)
                
            kvec_final = kvec_valid[final_indices] # [M', 3]
            nvec_final = nvec_valid[final_indices] # [M', 3]
            k_sq_final = k_sq_valid[final_indices] # [M']
            
            # Symmetry factors - compute in-place
            is_origin = (nvec_final == 0).all(dim=1) # [M'] bool
            factors = torch.where(is_origin, 1.0, 2.0)  # [M']
            
            # Compute kfac in-place
            kfac = torch.exp(-sigma_sq_half * k_sq_final) # [M']
            kfac.div_(k_sq_final + epsilon) # avoid division by zero
            
            # Structure factor computation - reuse intermediate results
            k_dot_r = torch.mm(pos_b, kvec_final.T) # [n_atoms_b, M']
            
            # Compute S_k components without creating intermediate tensors
            cos_k_dot_r = torch.cos(k_dot_r) # [n_atoms_b, M']
            sin_k_dot_r = torch.sin(k_dot_r) # [n_atoms_b, M']
            
            # Multiply q_b in-place and sum
            cos_k_dot_r *= q_b#.unsqueeze(1) # [n_atoms_b, M']
            sin_k_dot_r *= q_b#.unsqueeze(1) # [n_atoms_b, M']
            
            S_k_real = torch.sum(cos_k_dot_r, dim=0) # [M']
            S_k_imag = torch.sum(sin_k_dot_r, dim=0) # [M']
            
            # Compute S_k_sq in-place
            S_k_real.pow_(2) 
            S_k_imag.pow_(2)
            S_k_sq = S_k_real + S_k_imag # [M']
            
            # Compute potential for this batch
            volume = torch.det(cell[b_idx])
            
            # Combine factors, kfac, and S_k_sq efficiently
            factors *= kfac # [M']
            factors *= S_k_sq # [M']
            pot_b = torch.sum(factors) / volume # scalar
            
            # Remove self-interaction
            pot_b -= torch.sum(q_b**2) / (sigma * (2 * torch.pi)**1.5)
            
            # Assign to result
            result.append(pot_b * norm_factor)
            # result_potentials[atom_mask] = pot_b * norm_factor
        results = torch.stack(result)  # [num_batches]  
        # results = {"potential": result_potentials}
        return results

    def __repr__(self):
        return f"Ewald(dl={self.dl}, sigma={self.sigma}, remove_self_interaction={self.remove_self_interaction})"
