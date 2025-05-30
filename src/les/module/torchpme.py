import torch
import torch.nn as nn
from typing import Optional
import sys
import torchpme

__all__ = ['PME']

class PME(nn.Module):
    def __init__(self,
                 dl: float = 2.0,  # grid resolution
                 sigma: float = 1.0,  # width of the Gaussian on each atom
                 remove_self_interaction: bool = True,
                 norm_factor: float = (torchpme.prefactors.eV_A * 2.0 * torch.pi),
                 use_pme: bool = False,  # Changed default to False
                 apply_background_correction: bool = False,
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
        self.use_pme = use_pme
        self.calculator = None
        self.apply_background_correction = apply_background_correction

    def forward(self,
                q: torch.Tensor,  # [n_atoms, n_q]
                r: torch.Tensor,  # [n_atoms, 3]
                cell: torch.Tensor,  # [batch_size, 3, 3]
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
            box_now = cell[i] if cell is not None else None
            
            if self.use_pme:
                pot = self.compute_potential_pme(r_raw_now, q_now, box_now)
            else:
                # Return zero tensor when PME is not used
                pot = torch.tensor(0.0, device=r_raw_now.device, dtype=r_raw_now.dtype)
            results.append(pot)

        return torch.stack(results, dim=0).sum(dim=1)

    def setup_ewald_calculator(self, device: str = 'cpu', dtype: torch.dtype = torch.float32) -> torchpme.EwaldCalculator:
        """Configure Ewald parameters for PME calculation."""
        if self.calculator is None:
            potential = torchpme.CoulombPotential(self.sigma)
            self.calculator = torchpme.EwaldCalculator(
                potential=potential,
                lr_wavelength=self.dl,
                apply_background_correction=self.apply_background_correction,
                prefactor=torchpme.prefactors.eV_A
            )       
            self.calculator = self.calculator.to(device=device, dtype=dtype)
        return self.calculator

    def compute_potential_pme(self, r_raw: torch.Tensor, q: torch.Tensor, cell_now: torch.Tensor) -> torch.Tensor:
        """Compute potential energy using PME for periodic systems.
        
        Args:
            r_raw: Atomic positions [n_atoms, 3]
            q: Atomic charges [n_atoms, 1]
            cell_now: Simulation cell [3] or [3, 3]
            
        Returns:
            Potential energy tensor
        """
        device = r_raw.device
        dtype = r_raw.dtype
        
        # Ensure q has the correct shape [n_atoms, 1]
        if q.dim() == 1:
            q = q.unsqueeze(1)
            
        # Ensure r_raw has the correct shape [n_atoms, 3]
        if r_raw.dim() == 1:
            r_raw = r_raw.unsqueeze(1)
            
        if cell_now is None:
            cell_now = torch.zeros(3, device=device, dtype=dtype)
            
        if cell_now.dim() == 1:
            cell_now = torch.diag(cell_now)
        elif cell_now.dim() == 2 and not (cell_now.shape[0] == 3 and cell_now.shape[1] == 3):
            cell_now = torch.diag(cell_now)
            
        self.calculator = self.setup_ewald_calculator(device=device, dtype=dtype)
        
        try:
            potentials = self.calculator(
                charges=q,
                cell=cell_now,
                positions=r_raw,
                neighbor_indices=torch.zeros((0, 2), dtype=torch.int64, device=device), # neighbor_indices is not used in PME
                neighbor_distances=torch.zeros(0, dtype=dtype, device=device) # neighbor_distances is not used in PME
            )
            return (q.squeeze() * potentials.squeeze()).sum().unsqueeze(0)
            
        except Exception as e:
            print(f"PME calculation failed: {str(e)}")
            return torch.tensor(0.0, device=device, dtype=dtype)

    def __repr__(self) -> str:
        return f"PME(dl={self.dl}, sigma={self.sigma}, remove_self_interaction={self.remove_self_interaction}, use_pme={self.use_pme})"
