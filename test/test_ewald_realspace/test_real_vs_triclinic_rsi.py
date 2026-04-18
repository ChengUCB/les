"""
Compares the Ewald realspace vs triclinic paths with remove_self_interaction=True.

  - cell=None      → forward() dispatches both batch members to realspace.
  - cell=big_box   → forward() dispatches both batch members to triclinic.

Under remove_self_interaction=True:
  - realspace's kernels naturally exclude i=j, so no self-term is present.
  - triclinic's reciprocal sum naturally includes self-terms, so it must
    subtract q², u², and Q² self-energies.
Both should therefore agree (modulo image-tail error).

Q is generated traceless, so there is no Tr(Q) self-phi contribution to
worry about. All 14 (q_on, u_on, Q_on, kappa_on, alpha_on) combinations
from the RSI=False test are reproduced here with RSI=True.

Tolerance note: subtracting the self-energy shrinks |pot|, so the
absolute image-tail error (~1e-4) becomes a larger fraction of |pot|
than in the RSI=False test. We use atol=5e-4 (non-induction) and
atol=3e-3 (induction) to absorb the floor.
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


def _ewald_and_box():
    ep = Ewald(dl=DL, sigma=SIGMA, remove_self_interaction=True)
    box = torch.tensor(
        [[BOX_LEN, 0.0, 0.0], [0.0, BOX_LEN, 0.0], [0.0, 0.0, BOX_LEN]],
        dtype=torch.float64,
    )
    return ep, box


def _gen(q_on, u_on, Q_on, kappa_on, alpha_on, seed=SEED):
    torch.manual_seed(seed)
    n = ATOMS.shape[0]
    q = torch.zeros(n, dtype=torch.float64)
    u = torch.zeros(n, 3, dtype=torch.float64)
    Q = torch.zeros(n, 3, 3, dtype=torch.float64)
    if q_on:
        q = torch.rand(n, dtype=torch.float64) * 2.0 - 1.0
        q -= q.mean()
    if u_on:
        u = torch.rand(n, 3, dtype=torch.float64) * 2.0 - 1.0
    if Q_on:
        Qr = torch.rand(n, 3, 3, dtype=torch.float64) * 2.0 - 1.0
        Qr = 0.5 * (Qr + Qr.transpose(-1, -2))
        trace = torch.einsum('iaa->i', Qr)
        eye = torch.eye(3, dtype=torch.float64)
        Q = Qr - trace[:, None, None] * eye[None, :, :] / 3.0
    kappa = torch.rand(n, dtype=torch.float64) * 0.5 if kappa_on else None
    alpha = torch.rand(n, dtype=torch.float64) * 0.5 if alpha_on else None
    return q, u, Q, kappa, alpha


def _make_batch_of_two(q, u, Q, kappa, alpha, u_on, Q_on):
    """Stack two identical copies of the ATOMS system into a single batch."""
    n = ATOMS.shape[0]
    r_b     = torch.cat([ATOMS, ATOMS], dim=0)
    q_b     = torch.cat([q, q], dim=0)
    u_b     = torch.cat([u, u], dim=0)     if u_on     else None
    Q_b     = torch.cat([Q, Q], dim=0)     if Q_on     else None
    kappa_b = torch.cat([kappa, kappa])    if kappa is not None else None
    alpha_b = torch.cat([alpha, alpha])    if alpha is not None else None
    batch_b = torch.cat([torch.zeros(n, dtype=torch.int64),
                         torch.ones (n, dtype=torch.int64)])
    return r_b, q_b, u_b, Q_b, kappa_b, alpha_b, batch_b


def _check(a, b, atol, rtol):
    abs_err = (a - b).abs().max().item()
    scale   = max(a.abs().max().item(), b.abs().max().item(), 1e-12)
    rel_err = abs_err / scale
    ok = abs_err <= atol + rtol * scale
    return abs_err, rel_err, ok


def _compare(q_on, u_on, Q_on, kappa_on=False, alpha_on=False,
             atol=5e-4, rtol=1e-3):
    ep, box = _ewald_and_box()
    q, u, Q, kappa, alpha = _gen(q_on, u_on, Q_on, kappa_on, alpha_on)
    r_b, q_b, u_b, Q_b, kappa_b, alpha_b, batch_b = _make_batch_of_two(
        q, u, Q, kappa, alpha, u_on, Q_on)

    cell_tri = torch.stack([box, box], dim=0)   # [2, 3, 3]

    pot_re, q_ind_re, u_ind_re = ep(
        q=q_b, r=r_b, cell=None,
        batch=batch_b, u=u_b, quad=Q_b,
        kappa=kappa_b, alpha=alpha_b,
        compute_field=True,
    )
    pot_tri, q_ind_tri, u_ind_tri = ep(
        q=q_b, r=r_b, cell=cell_tri,
        batch=batch_b, u=u_b, quad=Q_b,
        kappa=kappa_b, alpha=alpha_b,
        compute_field=True,
    )

    tag = (f"[q={int(q_on)},u={int(u_on)},Q={int(Q_on)},"
           f"k={int(kappa_on)},a={int(alpha_on)}]")
    failures = []
    for key, a_, b_ in (
        ('pot',       pot_tri,   pot_re  ),
        ('q_induced', q_ind_tri, q_ind_re),
        ('u_induced', u_ind_tri, u_ind_re),
    ):
        abs_err, rel_err, ok = _check(a_.detach(), b_.detach(), atol, rtol)
        status = "OK  " if ok else "FAIL"
        print(f"  {tag} {key:<10s} abs={abs_err:.3e}  rel={rel_err:.3e}  [{status}]")
        if not ok:
            failures.append(f"{key}: abs={abs_err:.3e} rel={rel_err:.3e}")
    assert not failures, f"{tag} mismatches: " + "; ".join(failures)


# ═══════════════════════════════════════════════════════════════════════════
# No induction (kappa = alpha = None). Tight tolerance.
# ═══════════════════════════════════════════════════════════════════════════
def test_q_only():  _compare(q_on=True,  u_on=False, Q_on=False)
def test_u_only():  _compare(q_on=False, u_on=True,  Q_on=False)
def test_q_u():     _compare(q_on=True,  u_on=True,  Q_on=False)
def test_Q_only():  _compare(q_on=False, u_on=False, Q_on=True)
def test_q_Q():     _compare(q_on=True,  u_on=False, Q_on=True)
def test_u_Q():     _compare(q_on=False, u_on=True,  Q_on=True)
def test_q_u_Q():   _compare(q_on=True,  u_on=True,  Q_on=True)


# ═══════════════════════════════════════════════════════════════════════════
# With induction (random positive kappa, alpha). Mixed tolerance.
# ═══════════════════════════════════════════════════════════════════════════
_IND = dict(kappa_on=True, alpha_on=True, atol=3e-3, rtol=1e-3)

def test_q_only_induced():  _compare(q_on=True,  u_on=False, Q_on=False, **_IND)
def test_u_only_induced():  _compare(q_on=False, u_on=True,  Q_on=False, **_IND)
def test_q_u_induced():     _compare(q_on=True,  u_on=True,  Q_on=False, **_IND)
def test_Q_only_induced():  _compare(q_on=False, u_on=False, Q_on=True,  **_IND)
def test_q_Q_induced():     _compare(q_on=True,  u_on=False, Q_on=True,  **_IND)
def test_u_Q_induced():     _compare(q_on=False, u_on=True,  Q_on=True,  **_IND)
def test_q_u_Q_induced():   _compare(q_on=True,  u_on=True,  Q_on=True,  **_IND)
