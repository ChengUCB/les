import torch
import sys
torch.set_default_dtype(torch.float32)

import les
from les.module import Ewald

ep = Ewald(dl=2.0,
          sigma=2,
          remove_self_interaction=False,
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

kappa = torch.ones(10) * 0.5
alpha = torch.ones(10) * 0.5

result = ep.compute_potential_triclinic(r, q, torch.tensor(box),  u=u, kappa=kappa, alpha=alpha, compute_field=True, compute_potential=True)
ew_1, phi_1, field_1 = result['pot'], result['phi'], result['field']
print(ew_1)
print('reciprocal', field_1)
# Numerical Calculation (Force / charge)
# field_numeric = - dPot / dr_i / q_i
#grad_r = torch.autograd.grad(ew_1[0], r)[0]
#field_numeric = -grad_r / q.view(-1, 1)
#print(field_numeric)

result_r = ep.compute_potential_realspace(r, q, u=u, kappa=kappa, alpha=alpha, compute_field=True)
ew_1_s, phi_1_s, field_1_s = result_r['pot'], result_r['phi'], result_r['field']
print(ew_1_s)
print('real', field_1_s)

#print('dif in E_field')
#print(field_1-field_1_s)

print("compare electric potential")
print(phi_1)
print(phi_1_s)

print("compare induced q")
print(result['q_induced'])
print(result_r['q_induced'])

print("compare induced u")
print(result['u_induced'])
print(result_r['u_induced'])
