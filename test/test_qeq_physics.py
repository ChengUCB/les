"""
Physics of charge equilibration, checked against closed-form answers.

  P1 the coupling matrix is Coulomb: A_ij -> 14.3996/r_ij (eV A / e^2) once the
     Gaussians stop overlapping, and A_ii is the Gaussian self-energy
  P2 a two-atom system has an exact solution; the solver must reproduce it
  P3 electronegativity ordering: the more electronegative atom goes negative,
     and the transfer grows with the electronegativity difference
  P4 equal electronegativity => no charge transfer, whatever the geometry
  P5 NONLOCAL charge transfer -- changing chi on one fragment moves charge on a
     fragment 15 A away. This is what a Qeq buys over a local (3G) model and it
     is the property Ko et al. 2021 built 4G-HDNNP around
  P6 charge transfer grows as the two atoms approach
  P7 large hardness pins every charge to Q_tot/N
  P8 invariance: translation, rotation, and permutation of the atom order
  P9 NQE: a bigger Fukui weight takes a bigger share, and the shares are exactly
     proportional to the weights

Run:
    PYTHONPATH=src python test/test_qeq_physics.py
"""
import sys

import torch

from les import Les
from les.module import Ewald_vectorized, Nqe, Qeq

DT = torch.float64
# 1/(4 pi eps0) in eV Angstrom / e^2. LES carries 1/(2 eps0) as norm_factor=90.4756,
# rounded to six figures, so the two agree to ~4e-7 rather than exactly -- P1 checks
# the code against its own constant and that constant against the physical one.
COULOMB_PHYS = 14.399645


def _chk(gate, ok, label, fails, note=""):
    print(f"[{gate:9s}] {'ok  ' if ok else 'FAIL'} {label:28s} {note}")
    if not ok:
        fails.append(f"{label}:{gate.strip()}")


def _dimer(sep, sigma=1.0):
    """Two atoms `sep` apart, non-periodic (zero cell)."""
    r = torch.tensor([[0.0, 0.0, 0.0], [sep, 0.0, 0.0]], dtype=DT)
    cell = torch.zeros(1, 3, 3, dtype=DT)
    batch = torch.zeros(2, dtype=torch.long)
    ew = Ewald_vectorized(sigma=sigma, is_periodic=False).to(DT)
    return r, cell, batch, ew


def check_coulomb_limit():
    """P1: A is the Coulomb matrix once the Gaussians no longer overlap."""
    fails = []
    _, _, _, ew0 = _dimer(1.0)
    k_code = ew0.norm_factor / ew0.twopi          # the Coulomb constant LES uses
    _chk("P1 const", abs(k_code - COULOMB_PHYS) / COULOMB_PHYS < 1e-6,
         "norm_factor/(2pi) is 1/(4 pi eps0)", fails,
         f"{k_code:.6f} vs {COULOMB_PHYS:.6f} eV A/e^2")

    for sep in (8.0, 12.0, 20.0):
        r, cell, batch, ew = _dimer(sep)
        A = ew.compute_A_matrix(r, cell, batch, self_interaction=True)
        got, want = A[0, 1].item(), k_code / sep
        rel = abs(got - want) / want
        _chk("P1 coul", rel < 1e-12, f"A_12 at r={sep} A", fails,
             f"{got:.6f} eV = k/r  rel={rel:.1e}")
    # the diagonal is the Gaussian self-energy, norm_factor * 2/(sigma (2pi)^1.5)
    r, cell, batch, ew = _dimer(20.0)
    A = ew.compute_A_matrix(r, cell, batch, self_interaction=True)
    want = ew.norm_factor * 2.0 / (ew.sigma * ew.twopi ** 1.5)
    _chk("P1 self", abs(A[0, 0].item() - want) < 1e-12, "A_11 self-energy", fails,
         f"{A[0, 0].item():.6f} eV")
    # strip the two self terms and the rest is the bare point-charge energy
    qpm = torch.tensor([1.0, -1.0], dtype=DT)
    E_pair = 0.5 * (qpm @ (A @ qpm)) - A[0, 0]
    _chk("P1 pair", abs(E_pair.item() + k_code / 20.0) < 1e-12,
         "E(+1,-1) at 20 A", fails, f"{E_pair.item():.6f} eV")
    return fails


def check_dimer_analytic():
    """P2: the two-atom Qeq has a closed form; the solver must match it.

    With Q_tot = 0 and q_1 = -q_2 = q the objective is
        1/2 q^2 (A_11 + A_22 - 2 A_12 + J_1 + J_2) + q (chi_1 - chi_2)
    so    q = -(chi_1 - chi_2) / (A_11 + A_22 - 2 A_12 + J_1 + J_2).
    """
    fails = []
    for sep in (2.0, 5.0, 12.0):
        r, cell, batch, ew = _dimer(sep)
        z = torch.tensor([11, 17])                       # two distinct elements
        chi = torch.tensor([0.3, 1.1], dtype=DT)
        qeq = Qeq(chi_positive=False).to(DT)             # use chi as given
        A = ew.compute_A_matrix(r, cell, batch, self_interaction=True)
        q, lam = qeq(A, chi, z, batch, torch.zeros(1, dtype=DT),
                     torch.zeros(2, dtype=DT), torch.ones(2, dtype=torch.bool))
        J = torch.nn.functional.softplus(qeq.J_raw.to(DT))[z] + qeq.eps_j
        denom = A[0, 0] + A[1, 1] - 2 * A[0, 1] + J[0] + J[1]
        want = -(chi[0] - chi[1]) / denom
        rel = abs((q[0] - want) / want).item()
        _chk("P2 exact", rel < 1e-10, f"dimer r={sep} A", fails,
             f"q1={q[0].item():+.6f} vs analytic {want.item():+.6f}")
    return fails


def check_electronegativity():
    """P3: the more electronegative atom becomes negative, monotonically in dchi."""
    fails = []
    r, cell, batch, ew = _dimer(3.0)
    z = torch.tensor([11, 17])
    A = ew.compute_A_matrix(r, cell, batch, self_interaction=True)
    qeq = Qeq(chi_positive=False).to(DT)
    args = (z, batch, torch.zeros(1, dtype=DT), torch.zeros(2, dtype=DT),
            torch.ones(2, dtype=torch.bool))
    prev = 0.0
    ok_mono = True
    for dchi in (0.0, 0.5, 1.0, 2.0):
        chi = torch.tensor([0.0, dchi], dtype=DT)        # atom 1 more electronegative
        q, _ = qeq(A, chi, *args)
        # atom 1 has the LOWER chi here, so atom 1 should end up positive
        if dchi > 0:
            ok_mono = ok_mono and q[1].item() < prev - 1e-12
        prev = q[1].item()
        print(f"            dchi={dchi:4.1f} -> q = [{q[0].item():+.5f}, {q[1].item():+.5f}]")
    _chk("P3 sign", ok_mono, "more electronegative -> negative", fails,
         "monotone in dchi")

    # and softplus (the default) preserves that ordering
    qeq_sp = Qeq(chi_positive=True).to(DT)
    q_lo, _ = qeq_sp(A, torch.tensor([0.0, 1.0], dtype=DT), *args)
    q_hi, _ = qeq_sp(A, torch.tensor([0.0, 2.0], dtype=DT), *args)
    _chk("P3 softpl", q_hi[1].item() < q_lo[1].item() < 0.0, "ordering under softplus",
         fails, f"q2: {q_lo[1].item():+.5f} -> {q_hi[1].item():+.5f}")
    return fails


def _solve(r, cell, periodic, chi, z, qtot, sigma=1.0, j_init=1.0):
    n = r.shape[0]
    batch = torch.zeros(n, dtype=torch.long)
    ew = Ewald_vectorized(sigma=sigma, is_periodic=periodic).to(DT)
    A = ew.compute_A_matrix(r, cell, batch, self_interaction=True)
    qeq = Qeq(j_init=j_init, chi_positive=False).to(DT)
    q, lam = qeq(A, chi, z, batch, torch.tensor([qtot], dtype=DT),
                 torch.zeros(n, dtype=DT), torch.ones(n, dtype=torch.bool))
    return q, lam, A


def check_no_transfer():
    """P4: identical atoms transfer no charge; symmetric sites share equally."""
    fails = []
    g = torch.Generator().manual_seed(5)

    # (a) neutral + identical chi => exactly zero charge, at ANY geometry
    for n, periodic in ((4, False), (6, True), (8, True)):
        r = torch.rand(n, 3, generator=g, dtype=DT) * 7.0
        cell = ((torch.eye(3, dtype=DT) * 10.0).unsqueeze(0) if periodic
                else torch.zeros(1, 3, 3, dtype=DT))
        q, _, _ = _solve(r, cell, periodic, torch.full((n,), 0.7, dtype=DT),
                         torch.full((n,), 6), 0.0)
        _chk("P4 zero", q.abs().max().item() < 1e-10,
             f"n={n} {'per' if periodic else 'real'} neutral", fails,
             f"max|q|={q.abs().max().item():.2e}")

    # (b) a NET charge only spreads evenly when the sites are equivalent. Random
    # positions are not, so this needs real symmetry: a dimer, a square, and a
    # simple-cubic lattice where every atom maps onto every other.
    sq = 3.0
    cases = {
        "dimer": (torch.tensor([[0., 0., 0.], [2.5, 0., 0.]], dtype=DT),
                  torch.zeros(1, 3, 3, dtype=DT), False),
        "square": (torch.tensor([[0., 0., 0.], [sq, 0., 0.],
                                 [sq, sq, 0.], [0., sq, 0.]], dtype=DT),
                   torch.zeros(1, 3, 3, dtype=DT), False),
        "sc lattice": (torch.stack([torch.tensor([x, y, z], dtype=DT)
                                    for x in (0., 5.) for y in (0., 5.)
                                    for z in (0., 5.)]),
                       (torch.eye(3, dtype=DT) * 10.0).unsqueeze(0), True),
    }
    for name, (r, cell, periodic) in cases.items():
        n = r.shape[0]
        q, _, _ = _solve(r, cell, periodic, torch.full((n,), 0.7, dtype=DT),
                         torch.full((n,), 6), 1.0)
        spread = (q - 1.0 / n).abs().max().item()
        _chk("P4 share", spread < 1e-10, f"{name} (n={n}) Q=+1", fails,
             f"max|q_i - Q/N|={spread:.2e}")
    return fails


def check_surface_charge():
    """P4b: net charge piles up on the outside, as it does on a conductor.

    A straight chain of identical atoms given a net charge: the ends see fewer
    neighbours to repel them, so they carry more of it than the middle. This is
    the collective behaviour a local charge model has no way to produce.
    """
    fails = []
    n = 7
    r = torch.stack([torch.tensor([2.5 * i, 0.0, 0.0], dtype=DT) for i in range(n)])
    q, _, _ = _solve(r, torch.zeros(1, 3, 3, dtype=DT), False,
                     torch.full((n,), 0.7, dtype=DT), torch.full((n,), 6), 1.0)
    print("            q = [" + ", ".join(f"{x:+.5f}" for x in q.tolist()) + "]")
    mid = n // 2
    ok = q[0] > q[1] > q[mid] and q[-1] > q[-2] > q[mid]
    _chk("P4b surf", bool(ok), "ends carry more than the middle", fails,
         f"end={q[0].item():+.5f} mid={q[mid].item():+.5f}")
    _chk("P4b symm", (q - q.flip(0)).abs().max().item() < 1e-12,
         "chain is symmetric end to end", fails)
    return fails


def check_nonlocal():
    """P5: charge moves between fragments that never touch.

    Two dimers 15 A apart. Raising chi on the LEFT fragment must pull charge onto
    it and leave the RIGHT fragment positive, even though no descriptor of any
    right-hand atom changed. A local charge model cannot do this.
    """
    fails = []
    r = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0],
                      [15.0, 0.0, 0.0], [16.5, 0.0, 0.0]], dtype=DT)
    cell = torch.zeros(1, 3, 3, dtype=DT)
    batch = torch.zeros(4, dtype=torch.long)
    ew = Ewald_vectorized(is_periodic=False).to(DT)
    A = ew.compute_A_matrix(r, cell, batch, self_interaction=True)
    qeq = Qeq(chi_positive=False).to(DT)
    z = torch.tensor([6, 6, 6, 6])
    args = (z, batch, torch.zeros(1, dtype=DT), torch.zeros(4, dtype=DT),
            torch.ones(4, dtype=torch.bool))

    base, _ = qeq(A, torch.zeros(4, dtype=DT), *args)
    pert, _ = qeq(A, torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=DT), *args)
    dq = pert - base
    left, right = dq[:2].sum().item(), dq[2:].sum().item()
    print(f"            dq = [{', '.join(f'{x:+.5f}' for x in dq.tolist())}]")
    print(f"            left fragment {left:+.5f} e, right fragment {right:+.5f} e")
    _chk("P5 nonloc", abs(right) > 1e-3, "far fragment responds", fails,
         f"|dq_right|={abs(right):.4f} e")
    _chk("P5 opp", left * right < 0 and abs(left + right) < 1e-10,
         "charge conserved, opposite signs", fails, f"sum={left + right:+.1e}")

    # the whole dq/dchi Jacobian: off-diagonal blocks must be non-negligible
    chi_g = torch.zeros(4, dtype=DT, requires_grad=True)
    q_g, _ = qeq(A, chi_g, *args)
    jac = torch.stack([torch.autograd.grad(q_g[i], chi_g, retain_graph=True)[0]
                       for i in range(4)])
    cross = jac[2:, :2].abs().max().item()
    diag = jac.diagonal().abs().max().item()
    _chk("P5 jac", cross > 0.05 * diag, "dq_far/dchi_near is real", fails,
         f"cross={cross:.4f} vs diag={diag:.4f}")
    return fails


def check_distance_trend():
    """P6: the closer the two atoms, the more charge is transferred."""
    fails = []
    z = torch.tensor([11, 17])
    chi = torch.tensor([0.0, 1.0], dtype=DT)
    qeq = Qeq(chi_positive=False).to(DT)
    prev, ok = None, True
    for sep in (2.0, 3.0, 4.0, 6.0, 10.0, 20.0):
        r, cell, batch, ew = _dimer(sep)
        A = ew.compute_A_matrix(r, cell, batch, self_interaction=True)
        q, _ = qeq(A, chi, z, batch, torch.zeros(1, dtype=DT),
                   torch.zeros(2, dtype=DT), torch.ones(2, dtype=torch.bool))
        mag = abs(q[0].item())
        print(f"            r={sep:5.1f} A -> |q| = {mag:.6f} e")
        if prev is not None:
            ok = ok and mag < prev - 1e-9
        prev = mag
    _chk("P6 trend", ok, "|q| decreases with separation", fails)
    return fails


def check_hardness_limit():
    """P7: a very large hardness pins every charge to Q_tot/N."""
    fails = []
    g = torch.Generator().manual_seed(9)
    n = 6
    r = torch.rand(n, 3, generator=g, dtype=DT) * 7.0
    cell = torch.zeros(1, 3, 3, dtype=DT)
    batch = torch.zeros(n, dtype=torch.long)
    ew = Ewald_vectorized(is_periodic=False).to(DT)
    A = ew.compute_A_matrix(r, cell, batch, self_interaction=True)
    chi = torch.randn(n, generator=g, dtype=DT)
    z = torch.arange(n) + 1
    prev = None
    ok = True
    for j in (1.0, 1e2, 1e4, 1e6):
        qeq = Qeq(j_init=j, chi_positive=False).to(DT)
        q, _ = qeq(A, chi, z, batch, torch.tensor([1.0], dtype=DT),
                   torch.zeros(n, dtype=DT), torch.ones(n, dtype=torch.bool))
        spread = (q - 1.0 / n).abs().max().item()
        print(f"            J={j:8.0e} -> max|q_i - Q/N| = {spread:.3e}")
        if prev is not None:
            ok = ok and spread < prev
        prev = spread
    _chk("P7 hard", ok and spread < 1e-4, "J -> inf pins q to Q/N", fails)
    return fails


def check_invariance():
    """P8: translation, rotation and permutation."""
    fails = []
    g = torch.Generator().manual_seed(13)
    n = 7
    r = torch.rand(n, 3, generator=g, dtype=DT) * 6.0
    z = torch.randint(1, 20, (n,), generator=g)
    chi = torch.randn(n, generator=g, dtype=DT)
    qtot = torch.tensor([0.5], dtype=DT)

    for periodic in (False, True):
        cell = ((torch.eye(3, dtype=DT) * 12.0).unsqueeze(0) if periodic
                else torch.zeros(1, 3, 3, dtype=DT))
        batch = torch.zeros(n, dtype=torch.long)
        les = Les({'is_periodic': periodic, 'use_qeq': True}).to(DT)
        tag = "periodic" if periodic else "realspace"

        def run(pos, cl=cell, zz=z, cc=chi):
            o = les(positions=pos, cell=cl, latent_chi=cc, atomic_numbers=zz,
                    batch=batch, total_charge=qtot)
            return o['latent_charges'], o['E_lr']

        q0, E0 = run(r)
        # translation (a lattice-compatible one in the periodic case)
        qt, Et = run(r + torch.tensor([1.7, -0.4, 2.3], dtype=DT))
        _chk("P8 transl", (q0 - qt).abs().max().item() < 1e-9
             and (E0 - Et).abs().max().item() < 1e-9, f"{tag} translation", fails,
             f"dq={(q0 - qt).abs().max().item():.1e} dE={(E0 - Et).abs().max().item():.1e}")

        # rotation: free in real space; in a periodic cell rotate the cell too
        ang = 0.7
        c, s = torch.cos(torch.tensor(ang, dtype=DT)), torch.sin(torch.tensor(ang, dtype=DT))
        R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=DT)
        qr, Er = run(r @ R.T, cl=(cell @ R.T))
        _chk("P8 rot", (q0 - qr).abs().max().item() < 1e-9
             and (E0 - Er).abs().max().item() < 1e-9, f"{tag} rotation", fails,
             f"dq={(q0 - qr).abs().max().item():.1e} dE={(E0 - Er).abs().max().item():.1e}")

        # permutation of the atom order
        perm = torch.randperm(n, generator=g)
        qp, Ep = run(r[perm], zz=z[perm], cc=chi[perm])
        _chk("P8 perm", (q0[perm] - qp).abs().max().item() < 1e-9
             and (E0 - Ep).abs().max().item() < 1e-9, f"{tag} permutation", fails,
             f"dq={(q0[perm] - qp).abs().max().item():.1e}")
    return fails


def check_nqe_weights():
    """P9: shares are exactly proportional to the Fukui weights, and monotone."""
    fails = []
    n = 5
    batch = torch.zeros(n, dtype=torch.long)
    nqe = Nqe(eps=0.0).to(DT)
    f_raw = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0], dtype=DT)
    f = torch.nn.functional.softplus(f_raw)
    # add exactly one electron to a neutral set of charges
    q0 = torch.zeros(n, dtype=DT)
    q = nqe(q0, f_raw, batch, torch.tensor([-1.0], dtype=DT))
    print(f"            f      = [{', '.join(f'{x:.4f}' for x in f.tolist())}]")
    print(f"            shares = [{', '.join(f'{x:+.4f}' for x in q.tolist())}]")
    ratio = (q / (-f / f.sum()))
    _chk("P9 prop", (ratio - 1.0).abs().max().item() < 1e-12,
         "shares proportional to f", fails)
    _chk("P9 mono", bool((q[1:] < q[:-1]).all()),
         "larger f takes more of the electron", fails)
    _chk("P9 sum", abs(q.sum().item() + 1.0) < 1e-12, "one electron added", fails,
         f"sum={q.sum().item():+.12f}")
    return fails


def main():
    fails = []
    for fn in (check_coulomb_limit, check_dimer_analytic, check_electronegativity,
               check_no_transfer, check_surface_charge, check_nonlocal,
               check_distance_trend,
               check_hardness_limit, check_invariance, check_nqe_weights):
        print(f"\n================ {fn.__name__} ================")
        print(f"  {fn.__doc__.strip().splitlines()[0]}")
        fails += fn()

    print("\n==================== SUMMARY ====================")
    if fails:
        print("FAILURES:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("All Qeq / NQE physics checks passed.")


if __name__ == "__main__":
    main()
