import torch
import sys
torch.set_default_dtype(torch.float32)

import les
from les.module import Ewald

ep = Ewald(dl=1.0,
          sigma=2
          )

# set the same random seed for reproducibility
torch.manual_seed(sys.argv[1])
ri = torch.rand(10, 3) * 8  # Random positions in a 10x10x10 box
r = torch.tensor(ri, requires_grad=True, dtype=torch.float32)

q = torch.rand(10) * 2 # Random charges

q -= torch.mean(q)

u = torch.rand(10, 3) * 2.
u -= torch.mean(u, 0)

box = torch.tensor([[40.0, 0.0, 0.0], [ 0.0, 40.0, 0.0], [0.0, 0.0, 40.0]], dtype=torch.float32)  # Box dimensions


ew_1, field_1 = ep.compute_potential_triclinic(r, q, torch.tensor(box),  u=u, compute_field=True)
print(ew_1)
print('reciprocal', field_1)
# Numerical Calculation (Force / charge)
# field_numeric = - dPot / dr_i / q_i
#grad_r = torch.autograd.grad(ew_1[0], r)[0]
#field_numeric = -grad_r / q.view(-1, 1)
#print(field_numeric)
ew_1_s, field_1_s = ep.compute_potential_realspace(r, q, u=u, compute_field=True)
print(ew_1_s)
print('real', field_1_s)

#grad_r = torch.autograd.grad(ew_1_s, r)[0]
#field_numeric = -grad_r / q.view(-1, 1)
#print(field_numeric)

print('dif in E_field')
print(field_1-field_1_s)

