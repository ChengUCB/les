import sys
sys.path.append('../')

import torch
from les.module import AlphaDerivative

torch.manual_seed(0)

N = 10
box_full = torch.tensor([[10., 0, 0], [0, 10., 0], [0, 0, 10.]])
cell = box_full.unsqueeze(0)

# --- anisotropic alpha [N, 3, 3] ---
r = torch.rand(N, 3) * 10
r.requires_grad_(True)
# alpha as a function of r to create a valid autograd path
alpha_aniso = torch.einsum('ni,nj->nij', r, r)  # [N, 3, 3]

module = AlphaDerivative()
result_aniso = module(alpha=alpha_aniso, r=r, cell=cell, batch=None)

print("alpha_derivative (anisotropic) shape:", result_aniso.shape)
assert result_aniso.shape == (N, 3, 3, 3), f"Expected ({N}, 3, 3, 3), got {result_aniso.shape}"

# velocity contraction: d(alpha_ij)/dt = einsum('nijk,nk->ij', alpha_deriv, v) → [3, 3]
velocity = torch.rand(N, 3)
dalpha_rate = torch.einsum('nijk,nk->nij', result_aniso, velocity)  # [N, 3, 3]
total_dalpha = dalpha_rate.sum(0)                                     # [3, 3]
print("velocity contraction (dalpha/dt) shape:", total_dalpha.shape)
assert total_dalpha.shape == (3, 3), f"Expected (3, 3), got {total_dalpha.shape}"

# --- isotropic alpha [N] ---
r2 = torch.rand(N, 3) * 10
r2.requires_grad_(True)
alpha_iso = r2.pow(2).sum(dim=1)  # [N]

result_iso = module(alpha=alpha_iso, r=r2, cell=cell, batch=None)
print("alpha_derivative (isotropic) shape:", result_iso.shape)
assert result_iso.shape == (N, 3), f"Expected ({N}, 3), got {result_iso.shape}"

# --- multi-batch anisotropic ---
N_multi = 12
batch = torch.tensor([0]*6 + [1]*6)
r3 = torch.rand(N_multi, 3) * 10
r3.requires_grad_(True)
alpha_multi = torch.einsum('ni,nj->nij', r3, r3)
cell_multi = box_full.unsqueeze(0).expand(2, -1, -1).contiguous()

result_multi = module(alpha=alpha_multi, r=r3, cell=cell_multi, batch=batch)
print("alpha_derivative (multi-batch anisotropic) shape:", result_multi.shape)
assert result_multi.shape == (N_multi, 3, 3, 3), f"Expected ({N_multi}, 3, 3, 3), got {result_multi.shape}"

print("\nAll shape tests passed.")
