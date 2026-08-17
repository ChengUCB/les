from les import Les
import logging, traceback
import torch
import tempfile

############################## data generation ##############################
device = "cpu"
dtype = torch.float32

torch.manual_seed(42)

# config 0: simple cubic box
box0 = torch.tensor([
    [10.0, 0.0, 0.0],
    [0.0, 10.0, 0.0],
    [0.0, 0.0, 10.0],
], device=device, dtype=dtype)

# config 1: triclinic box
box_tric = torch.tensor([
    [10.0, 2.0, 1.0],
    [0.0, 9.0, 1.5],
    [0.0, 0.0, 10.0],
], device=device, dtype=dtype)

# config 2: orthorhombic box
box2 = torch.tensor([
    [8.0, 0.0, 0.0],
    [0.0, 12.0, 0.0],
    [0.0, 0.0, 9.0],
], device=device, dtype=dtype)

cells = torch.stack([box0, box_tric, box2], dim=0)   # [3, 3, 3]

n_atoms_per_config = [5, 6, 7] 
n_configs = len(n_atoms_per_config)
n_atoms = sum(n_atoms_per_config)

rs = []
for cfg_idx, (n_i, cell_i) in enumerate(zip(n_atoms_per_config, cells)):
    # fractional coordinates
    s_rand = torch.rand(n_i, 3, device=device, dtype=dtype)
    # actual positions r = s * cell (matrix multiply)
    r_i = torch.matmul(s_rand, cell_i)  # [n_i, 3]
    rs.append(r_i)

r_all = torch.cat(rs, dim=0)          # [n_atoms, 3]
r_all.requires_grad_(True)

q_all = torch.rand(n_atoms, device=device, dtype=dtype) * 2.0 - 1.0  # [-1, 1] 
q_all = q_all.unsqueeze(-1)  # [n_atoms, 1]

#2 channel q
# q_all = torch.rand(n_atoms, 2, device=device, dtype=dtype) * 2.0 - 1.0  # [-1, 1]


batch_list = []
for cfg_idx, n_i in enumerate(n_atoms_per_config):
    batch_list.append(
        torch.full((n_i,), cfg_idx, device=device, dtype=torch.long)
    )

batch_all = torch.cat(batch_list, dim=0)  # [n_atoms]

# print("n_atoms:", n_atoms)
# print("r_all shape     :", r_all.shape)      # [n_atoms, 3]
# print("q_all shape     :", q_all.shape)      # [n_atoms]
# print("cells shape     :", cells.shape)      # [3, 3, 3]
# print("batch_all shape :", batch_all.shape)  # [n_atoms]
# print("batch_all      :", batch_all)


############################## checking ##############################
FAILURES = []


def check(name, got, ref, rtol=1e-4, atol=1e-5):
    print(f"{name}\n  reference : {ref}\n  obtained  : {got}")
    try:
        torch.testing.assert_close(got, ref, rtol=rtol, atol=atol)
        print("  -> match")
    except AssertionError as e:
        print(f"  -> MISMATCH: {str(e).splitlines()[0]}")
        FAILURES.append(name)


############################## Ewald test ##############################
print("############################## Ewald test ##############################")
from les.module import Ewald, Ewald_vectorized

org_ewald = Ewald(dl=2.0, sigma=1.0, remove_self_interaction=True)
vec_ewald = Ewald_vectorized(dl=2.0, sigma=1.0, remove_self_interaction=True)

### non-periodic case (real space) ###
unique_batches = torch.unique(batch_all)
results = []
for i in unique_batches:
    mask = batch_all == i  # Create a mask for the i-th configuration
    r_raw_now, q_now = r_all[mask], q_all[mask]
    pot = org_ewald.compute_potential_realspace(r_raw_now, q_now)['pot']
    results.append(pot)
org_result = torch.stack(results, dim=0).sum(dim=1)
vec_result = vec_ewald.compute_potential_realspace(r_all, q_all, cells, batch_all)['pot']
check("real space: vectorized vs legacy", vec_result, org_result)

### periodic case (reciprocal space) ###
unique_batches = torch.unique(batch_all)
results = []
for i in unique_batches:
    mask = batch_all == i  # Create a mask for the i-th configuration
    r_raw_now, q_now = r_all[mask], q_all[mask]
    box_now = cells[i]  # Get the box for the i-th configuration
    pot = org_ewald.compute_potential_triclinic(r_raw_now, q_now, box_now)['pot']
    results.append(pot)

org_result = torch.stack(results, dim=0).sum(dim=1)
vec_result = vec_ewald.compute_potential_triclinic(r_all, q_all, cells, batch_all)['pot']
check("reciprocal space: vectorized vs legacy", vec_result, org_result)


############################## Ewald compilation test ##############################
print("############################## Ewald compilation test ##############################")

updated_les_non_periodic = Les({'is_periodic': False})
updated_les_periodic = Les({'is_periodic': True})

for les_module, name in ((updated_les_non_periodic, "real space"),
                         (updated_les_periodic, "reciprocal space")):
    kw = dict(positions=r_all, latent_charges=q_all, cell=cells, batch=batch_all)
    eager = les_module(**kw)['E_lr']
    compiled = torch.compile(les_module, dynamic=True, fullgraph=True)(**kw)['E_lr']
    check(f"{name}: compiled LES vs eager LES", compiled, eager)


############################## summary ##############################
print("\n############################## SUMMARY ##############################")
if FAILURES:
    for name in FAILURES:
        print("  FAILED:", name)
    raise SystemExit(f"{len(FAILURES)} comparison(s) failed")
print("all comparisons matched")