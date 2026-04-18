"""
Human-readable sanity check: runs the test_q_u_Q_induced scenario through
both remove_self_interaction={False, True} and prints the realspace vs
triclinic outputs side by side.

Uses the same ATOMS / seed / (q,u,Q,kappa,alpha) generation as
test_real_vs_triclinic.py so the numbers here match those tests.
"""
import torch

from les.module import Ewald


ATOMS = torch.tensor([
    [ 0.06633400,  0.00000000,  0.00370100],
    [-0.52638300, -0.76932700, -0.02936600],
    [-0.52638300,  0.76932700, -0.02936600],
    [-0.06632200,  0.00000000,  2.38157100],
    [ 0.48094200, -0.00000000,  3.18454900],
    [ 0.57162900, -0.00000000,  1.64857000],
], dtype=torch.float64)

BOX_LEN = 100.0
SIGMA   = 2.0
DL      = 1.0
SEED    = 34


def _gen():
    torch.manual_seed(SEED)
    n = ATOMS.shape[0]
    q = torch.rand(n, dtype=torch.float64) * 2.0 - 1.0
    q -= q.mean()
    u = torch.rand(n, 3, dtype=torch.float64) * 2.0 - 1.0
    Qr = torch.rand(n, 3, 3, dtype=torch.float64) * 2.0 - 1.0
    Qr = 0.5 * (Qr + Qr.transpose(-1, -2))
    trace = torch.einsum('iaa->i', Qr)
    eye = torch.eye(3, dtype=torch.float64)
    Q = Qr - trace[:, None, None] * eye[None, :, :] / 3.0
    kappa = torch.rand(n, dtype=torch.float64) * 0.5
    alpha = torch.rand(n, dtype=torch.float64) * 0.5
    return q, u, Q, kappa, alpha


def _run(rsi):
    n = ATOMS.shape[0]
    ep = Ewald(dl=DL, sigma=SIGMA, remove_self_interaction=rsi)
    box = torch.tensor(
        [[BOX_LEN, 0.0, 0.0], [0.0, BOX_LEN, 0.0], [0.0, 0.0, BOX_LEN]],
        dtype=torch.float64,
    )
    q, u, Q, kappa, alpha = _gen()
    r_b     = torch.cat([ATOMS, ATOMS], dim=0)
    q_b     = torch.cat([q, q], dim=0)
    u_b     = torch.cat([u, u], dim=0)
    Q_b     = torch.cat([Q, Q], dim=0)
    kappa_b = torch.cat([kappa, kappa], dim=0)
    alpha_b = torch.cat([alpha, alpha], dim=0)
    batch_b = torch.cat([torch.zeros(n, dtype=torch.int64),
                         torch.ones (n, dtype=torch.int64)])
    cell_tri = torch.stack([box, box], dim=0)

    pot_re, q_ind_re, u_ind_re = ep(
        q=q_b, r=r_b, cell=None, batch=batch_b,
        u=u_b, quad=Q_b, kappa=kappa_b, alpha=alpha_b,
        compute_field=True,
    )
    pot_tri, q_ind_tri, u_ind_tri = ep(
        q=q_b, r=r_b, cell=cell_tri, batch=batch_b,
        u=u_b, quad=Q_b, kappa=kappa_b, alpha=alpha_b,
        compute_field=True,
    )
    return pot_re, q_ind_re, u_ind_re, pot_tri, q_ind_tri, u_ind_tri


def _print_block(label, pot_re, q_re, u_re, pot_tri, q_tri, u_tri):
    torch.set_printoptions(precision=6, sci_mode=False, linewidth=120)
    print("=" * 78)
    print(f"  {label}")
    print("=" * 78)
    print(f"\npot (realspace):   {pot_re.detach().numpy()}")
    print(f"pot (triclinic):   {pot_tri.detach().numpy()}")
    print(f"pot abs diff    :   {(pot_tri - pot_re).abs().max().item():.3e}")

    print("\nq_induced (realspace):")
    print(q_re.detach())
    print("q_induced (triclinic):")
    print(q_tri.detach())
    print(f"q_induced abs diff: {(q_tri - q_re).abs().max().item():.3e}")

    print("\nu_induced (realspace):")
    print(u_re.detach())
    print("u_induced (triclinic):")
    print(u_tri.detach())
    print(f"u_induced abs diff: {(u_tri - u_re).abs().max().item():.3e}")
    print()


if __name__ == "__main__":
    for rsi in (False, True):
        out = _run(rsi)
        _print_block(f"test_q_u_Q_induced  —  remove_self_interaction={rsi}", *out)
