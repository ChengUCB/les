import torch
import sys
torch.set_default_dtype(torch.float32)

import les
from les.module import Ewald

# set the same random seed for reproducibility
torch.manual_seed(int(sys.argv[1]))
ri = torch.rand(10, 3) * 8  # Random positions in a 10x10x10 box
r = torch.tensor(ri, requires_grad=True, dtype=torch.float32)

q = torch.rand(10) * 2 # Random charges
q -= torch.mean(q)

u = torch.rand(10, 3) * 2.
u -= torch.mean(u, 0)

Qr = torch.rand(10, 3, 3) * 2.
Qr = 0.5 * (Qr + Qr.transpose(-1, -2))
trace = torch.einsum('iaa->i', Qr)
eye = torch.eye(3)
Q = Qr - trace[:, None, None] * eye[None, :, :] / 3.0

box = torch.tensor([[40.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]], dtype=torch.float32)

kappa = torch.ones(10) * 0.5
alpha = torch.ones(10) * 0.5


def run(rsi):
    ep = Ewald(dl=2.0, sigma=2, remove_self_interaction=rsi)
    print("=" * 72)
    print(f"  remove_self_interaction = {rsi}")
    print("=" * 72)

    res_t = ep.compute_potential_triclinic(
        r, q, torch.tensor(box), u=u, quad=Q,
        kappa=kappa, alpha=alpha,
        compute_field=True, compute_potential=True,
    )
    res_r = ep.compute_potential_realspace(
        r, q, u=u, quad=Q,
        kappa=kappa, alpha=alpha,
        compute_field=True,
    )

    print("pot:")
    print("  triclinic:", res_t['pot'])
    print("  real     :", res_r['pot'])

    print("field (triclinic):")
    print(res_t['field'])
    print("field (real):")
    print(res_r['field'])

    print("phi (triclinic):")
    print(res_t['phi'])
    print("phi (real):")
    print(res_r['phi'])

    print("q_induced (triclinic):", res_t['q_induced'])
    print("q_induced (real)     :", res_r['q_induced'])

    print("u_induced (triclinic):")
    print(res_t['u_induced'])
    print("u_induced (real):")
    print(res_r['u_induced'])
    print()


for rsi in (False, True):
    run(rsi)
