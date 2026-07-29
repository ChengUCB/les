import torch
import sys
torch.set_default_dtype(torch.float32)

import les
from les.module import Ewald

ep = Ewald(dl=1.5,
          sigma=1,
          remove_self_interaction=True,
          )

# set the same random seed for reproducibility
torch.manual_seed(sys.argv[1])
r = torch.rand(10, 3) * 8  # Random positions in a 10x10x10 box
q = torch.rand(10) * 2 # Random charges

q -= torch.mean(q)
box = torch.tensor([[40.0, 0.0, 0.0], [ 0.0, 40.0, 0.0], [0.0, 0.0, 40.0]], dtype=torch.float32)  # Box dimensions


result = ep.compute_potential_triclinic(torch.tensor(r), torch.tensor(q).unsqueeze(1), torch.tensor(box), compute_field=True, compute_potential=True)
ew_1, phi_1, field_1 = result['pot'], result['phi'], result['field']
print(ew_1)
result_r = ep.compute_potential_realspace(torch.tensor(r), torch.tensor(q), compute_field=True)
ew_1_s, phi_1_s, field_1_s = result_r['pot'], result_r['phi'], result_r['field']
print(ew_1_s)

print("compare electric fields")
print(field_1)
print(field_1_s)

print("compare electric potential")
print(phi_1)
print(phi_1_s)

print(torch.sum(q.unsqueeze(1) * phi_1 / 2), torch.sum(q.unsqueeze(1) * phi_1_s / 2))
