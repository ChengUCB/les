"""
Charge equilibration (Qeq) and neural charge equilibration (NQE).

  A  E_lr == 1/2 q^T A q, so compute_A_matrix and the Ewald energy share one
     unit convention -- the gate that pins norm_factor, every factor of two and
     the self-interaction term at once
  B  the solve really solves: residual of the bordered system, per-structure
     total charge with a DIFFERENT target per structure, and A + J positive
     definite so the solution is the minimum and not a saddle
  C  gauge invariance -- a uniform shift of chi is absorbed by lambda, which is
     why forcing chi > 0 is a reparameterization and not a physical constraint
  D  forces: autograd through linalg.solve == central differences. A depends on
     the positions, so dq/dr != 0 and the implicit-differentiation term has to
     come out of autograd correctly
  E  partial equilibration: frozen atoms keep their charge exactly and still
     field the equilibrated ones; a structure with nothing active passes through
  F  NQE: exact total charge, the remove_mean special case, and shares
     proportional to f rather than to exp(f)
  G  qeq_self_interaction=False constructs and solves
  K  the combinations that cannot work are refused, not silently wrong
  N  charge_scope: 'system' constrains the whole structure, 'masked' constrains
     only the selected atoms -- for Qeq and for NQE alike
  O  the per-structure (padded) layout agrees with the flat one, and the energy
     read straight off A agrees with a full Ewald pass
  I  the solution really is the constrained MINIMUM of the Qeq objective, not
     just a stationary point of it
  J  gradients reach the chi readout and the learned hardness, which is what
     training the thing actually requires
  H  TorchScript round-trip

Everything is float64 on the CPU, and swept over both boundary conditions and
non-unit sigma/dl -- at sigma=1 every sigma**k is 1, so a wrong exponent would be
invisible at the defaults.

Run:
    PYTHONPATH=src python test/test_qeq.py
"""
import sys

import torch

from les import Les
from les.module import Ewald_vectorized, Nqe, Qeq

DT = torch.float64
SIGMA_DL = ((1.0, 2.0), (1.3, 1.5), (0.7, 2.5))


def _chk(gate, ok, label, fails, note=""):
    print(f"[{gate:9s}] {'ok  ' if ok else 'FAIL'} {label} {note}")
    if not ok:
        fails.append(f"{label}:{gate.strip()}")


def _system(periodic, n_per=(5, 6, 7), seed=42):
    """A small batch of structures; zero cells are how LES marks non-periodic."""
    g = torch.Generator().manual_seed(seed)
    n = sum(n_per)
    batch = torch.cat([torch.full((k,), i) for i, k in enumerate(n_per)]).long()
    r = torch.rand(n, 3, generator=g, dtype=DT) * 8.0
    z = torch.randint(1, 30, (n,), generator=g)
    chi = torch.randn(n, generator=g, dtype=DT)
    q0 = torch.randn(n, generator=g, dtype=DT)
    if periodic:
        cells = torch.stack([
            torch.eye(3, dtype=DT) * 10.0,
            torch.tensor([[9.0, 0.4, 0.0], [0.3, 10.0, 0.2], [0.0, 0.1, 11.0]], dtype=DT),
            torch.diag(torch.tensor([9.0, 10.0, 12.0], dtype=DT)),
        ])[:len(n_per)]
    else:
        cells = torch.zeros(len(n_per), 3, 3, dtype=DT)
    onehot = (batch.unsqueeze(0) == torch.arange(len(n_per)).unsqueeze(1)).to(DT)
    return r, cells, batch, z, chi, q0, onehot


def check_units(periodic, rsi, sigma, dl, label):
    """A: 1/2 q^T A q reproduces the Ewald energy exactly.

    compute_A_matrix(self_interaction=not remove_self_interaction) is the setting
    that matches this module's own energy: the real-space pair grid excludes the
    diagonal while the reciprocal sum over k already contains it, so the two
    paths start from opposite places.
    """
    fails = []
    r, cells, batch, _, _, q, onehot = _system(periodic)
    ew = Ewald_vectorized(sigma=sigma, dl=dl, is_periodic=periodic,
                          remove_self_interaction=rsi, N_max=10).to(DT)
    E_ref, _, _ = ew(q=q, r=r, cell=cells, batch=batch)
    A = ew.compute_A_matrix(r, cells, batch, self_interaction=not rsi)
    E_A = 0.5 * (onehot @ (q * (A @ q)))
    rel = ((E_A - E_ref).abs().max() / E_ref.abs().max()).item()
    _chk("A units", rel < 1e-10, label, fails, f"rel={rel:.2e}")
    _chk("A symm", (A - A.T).abs().max().item() < 1e-12, label, fails)
    return fails


def check_solve(periodic, label):
    """B: residual, per-structure conservation, positive definiteness."""
    fails = []
    r, cells, batch, z, chi, _, onehot = _system(periodic)
    B = cells.shape[0]
    qtot = torch.tensor([0.0, 1.0, -2.0], dtype=DT)[:B]

    ew = Ewald_vectorized(is_periodic=periodic).to(DT)
    qeq = Qeq().to(DT)
    A = ew.compute_A_matrix(r, cells, batch, self_interaction=True)
    q, lam = qeq(A, chi, z, batch, qtot, torch.zeros_like(chi),
                 torch.ones_like(batch, dtype=torch.bool))

    J = torch.nn.functional.softplus(qeq.J_raw.to(DT))[z] + qeq.eps_j
    chi_pos = torch.nn.functional.softplus(chi)
    # (A + J) q + lambda 1 + chi = 0 on every atom
    resid = (A + torch.diag(J)) @ q + onehot.T @ lam + chi_pos
    _chk("B resid", resid.abs().max().item() < 1e-9, label, fails,
         f"max={resid.abs().max().item():.2e}")

    dq = (onehot @ q - qtot).abs().max().item()
    _chk("B sum-q", dq < 1e-10, label, fails, f"max|sum q - Q|={dq:.2e}")

    # block diagonal, so the whole spectrum is the union of the per-structure ones
    eig = torch.linalg.eigvalsh(A + torch.diag(J))
    _chk("B posdef", eig.min().item() > 0.0, label, fails, f"min eig={eig.min().item():.3f}")
    return fails


def check_gauge(periodic, label):
    """C: chi -> chi + c leaves q untouched and shifts lambda by -c."""
    fails = []
    r, cells, batch, z, chi, _, _ = _system(periodic)
    B = cells.shape[0]
    qtot = torch.tensor([0.0, 1.0, -2.0], dtype=DT)[:B]
    ew = Ewald_vectorized(is_periodic=periodic).to(DT)
    # chi_positive off, so the shift survives to the linear system
    qeq = Qeq(chi_positive=False).to(DT)
    A = ew.compute_A_matrix(r, cells, batch, self_interaction=True)
    args = (z, batch, qtot, torch.zeros_like(chi), torch.ones_like(batch, dtype=torch.bool))
    q1, l1 = qeq(A, chi, *args)
    q2, l2 = qeq(A, chi + 5.0, *args)
    _chk("C gauge", (q1 - q2).abs().max().item() < 1e-12, label, fails,
         f"max|dq|={(q1 - q2).abs().max().item():.2e}")
    _chk("C lambda", (l2 - l1 + 5.0).abs().max().item() < 1e-10, label, fails)
    return fails


def check_forces(periodic, label):
    """D: autograd through the solve == central differences."""
    fails = []
    r, cells, batch, z, chi, _, _ = _system(periodic, n_per=(4, 5))
    qtot = torch.tensor([0.0, 1.0], dtype=DT)
    les = Les({'is_periodic': periodic, 'use_qeq': True}).to(DT)

    def energy(pos):
        return les(positions=pos, cell=cells, latent_chi=chi, atomic_numbers=z,
                   batch=batch, total_charge=qtot)['E_lr'].sum()

    rg = r.clone().requires_grad_(True)
    F_auto = -torch.autograd.grad(energy(rg), rg)[0]

    h = 1e-6
    F_fd = torch.zeros_like(r)
    for i in range(r.shape[0]):
        for d in range(3):
            rp = r.clone(); rp[i, d] += h
            rm = r.clone(); rm[i, d] -= h
            F_fd[i, d] = -(energy(rp) - energy(rm)) / (2 * h)
    err = (F_auto - F_fd).abs().max().item()
    scale = max(F_fd.abs().max().item(), 1e-12)
    _chk("D forces", err / scale < 1e-6, label, fails, f"rel={err / scale:.2e}")
    return fails


def check_mask(periodic, label):
    """E: all-true mask == unmasked, partial mask freezes exactly, empty passes through."""
    fails = []
    r, cells, batch, z, chi, q0, onehot = _system(periodic)
    B = cells.shape[0]
    qtot = torch.tensor([0.0, 1.0, -2.0], dtype=DT)[:B]
    ew = Ewald_vectorized(is_periodic=periodic).to(DT)
    qeq = Qeq().to(DT)
    A = ew.compute_A_matrix(r, cells, batch, self_interaction=True)
    all_on = torch.ones_like(batch, dtype=torch.bool)

    q_full, _ = qeq(A, chi, z, batch, qtot, q0, all_on)
    q_zero, _ = qeq(A, chi, z, batch, qtot, torch.zeros_like(q0), all_on)
    _chk("E prior", (q_full - q_zero).abs().max().item() < 1e-12, label, fails,
         "an all-true mask ignores q_prior")

    mask = all_on.clone()
    mask[:3] = False
    q_m, lam_m = qeq(A, chi, z, batch, qtot, q0, mask)
    _chk("E frozen", torch.equal(q_m[:3], q0[:3]), label, fails, "frozen q bit-identical")
    dq = (onehot @ q_m - qtot).abs().max().item()
    _chk("E sum-q", dq < 1e-10, label, fails, f"max|sum q - Q|={dq:.2e}")
    # the frozen charges must actually be felt: their field enters the active rows
    q_m0, _ = qeq(A, chi, z, batch, qtot, torch.zeros_like(q0), mask)
    _chk("E field", (q_m[3:] - q_m0[3:]).abs().max().item() > 1e-6, label, fails,
         "frozen charges field the active ones")

    # a structure with nothing to equilibrate: pass through, lambda = 0
    empty = all_on.clone()
    empty[batch == 0] = False
    q_e, lam_e = qeq(A, chi, z, batch, qtot, q0, empty)
    _chk("E empty", torch.equal(q_e[batch == 0], q0[batch == 0])
         and abs(lam_e[0].item()) < 1e-12, label, fails)
    return fails


def check_nqe(label):
    """F: exact conservation, the remove_mean special case, and f/sum(f) shares."""
    fails = []
    _, _, batch, _, _, q0, onehot = _system(False)
    B = onehot.shape[0]
    qtot = torch.tensor([0.0, 1.0, -2.0], dtype=DT)[:B]
    nqe = Nqe(eps=0.0).to(DT)

    f_raw = torch.randn(q0.shape[0], generator=torch.Generator().manual_seed(7), dtype=DT)
    q = nqe(q0, f_raw, batch, qtot)
    dq = (onehot @ q - qtot).abs().max().item()
    _chk("F sum-q", dq < 1e-12, label, fails, f"max|sum q - Q|={dq:.2e}")

    # uniform weights with Q_tot = 0 is exactly what remove_mean does
    uniform = torch.zeros_like(q0)
    q_u = nqe(q0, uniform, batch, torch.zeros(B, dtype=DT))
    mean_free = q0 - (onehot.T @ (onehot @ q0)) / (onehot.T @ onehot.sum(dim=1))
    _chk("F remmean", (q_u - mean_free).abs().max().item() < 1e-12, label, fails)

    # with q~ = 0 and Q_tot = 1 the charges ARE the shares, so proportionality to f
    # is directly readable -- softmax(f) would also sum to 1 but be wrong here
    one = torch.zeros(B, dtype=DT); one[:] = 1.0
    shares = nqe(torch.zeros_like(q0), f_raw, batch, one)
    f = torch.nn.functional.softplus(f_raw)
    want = f / (onehot.T @ (onehot @ f))
    _chk("F fukui", (shares - want).abs().max().item() < 1e-12, label, fails)
    soft = torch.zeros_like(f)
    for b in range(B):
        m = batch == b
        soft[m] = torch.softmax(f[m], dim=0)
    _chk("F notsmax", (shares - soft).abs().max().item() > 1e-3, label, fails,
         "shares are f/sum(f), not softmax(f)")
    return fails


def check_no_self(periodic, label):
    """G: qeq_self_interaction=False builds and solves (the default is +self)."""
    fails = []
    r, cells, batch, z, chi, _, onehot = _system(periodic)
    qtot = torch.zeros(cells.shape[0], dtype=DT)
    les = Les({'is_periodic': periodic, 'use_qeq': True,
               'qeq_self_interaction': False}).to(DT)
    out = les(positions=r, cell=cells, latent_chi=chi, atomic_numbers=z, batch=batch,
              total_charge=qtot)
    q = out['latent_charges']
    ok = bool(torch.isfinite(q).all()) and (onehot @ q).abs().max().item() < 1e-9
    _chk("G no-self", ok, label, fails)
    return fails


def check_charge_scope(periodic, label):
    """N: both scopes hold whatever the frozen atoms carry.

    'system' pins sum over ALL atoms to total_charge, so the selected region takes
    up the remainder. 'masked' pins the sum over the SELECTED atoms instead, which
    is the same linear system with total_charge shifted by sum q_frozen. Either way
    the unselected atoms keep their incoming charge exactly.
    """
    fails = []
    n = 6
    r, cells, batch, z, chi, _, onehot = _system(periodic, n_per=(n,))
    f_raw = torch.randn(n, generator=torch.Generator().manual_seed(8), dtype=DT)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[n // 2:] = True                      # the second half absorbs
    qtot = torch.tensor([-1.0], dtype=DT)

    for feat in ('qeq', 'nqe'):
        for scope in ('system', 'masked'):
            les = Les({'is_periodic': periodic, f'use_{feat}': True,
                       f'{feat}_charge_scope': scope}).to(DT)
            ok_target, ok_frozen = True, True
            # a prior the constraint must be insensitive to
            for pf in (0.0, 0.3, 1.0, -0.8):
                prior = torch.zeros(n, dtype=DT)
                prior[~mask] = pf
                kw = dict(positions=r, cell=cells, atomic_numbers=z, batch=batch,
                          total_charge=qtot)
                if feat == 'qeq':
                    q = les(latent_chi=chi, latent_charges=prior,
                            qeq_mask=mask, **kw)['latent_charges']
                else:
                    q = les(latent_charges=prior, latent_fukui=f_raw,
                            nqe_mask=mask, **kw)['latent_charges']
                got = q.sum() if scope == 'system' else q[mask].sum()
                ok_target = ok_target and abs(got.item() - qtot.item()) < 1e-10
                ok_frozen = ok_frozen and torch.equal(q[~mask], prior[~mask])
            what = 'sum(all)' if scope == 'system' else 'sum(masked)'
            _chk(f"N {feat}", ok_target, f"{label} [{scope}]", fails,
                 f"{what} == Q_tot for every frozen prior")
            _chk(f"N frozen", ok_frozen, f"{label} [{scope}]", fails,
                 "unselected charges untouched")

    # with no mask at all the two scopes must coincide
    for feat in ('qeq', 'nqe'):
        outs = []
        for scope in ('system', 'masked'):
            les = Les({'is_periodic': periodic, f'use_{feat}': True,
                       f'{feat}_charge_scope': scope}).to(DT)
            kw = dict(positions=r, cell=cells, atomic_numbers=z, batch=batch,
                      total_charge=qtot)
            outs.append((les(latent_chi=chi, **kw) if feat == 'qeq'
                         else les(latent_charges=chi, latent_fukui=f_raw, **kw)
                         )['latent_charges'])
        _chk(f"N nomask", torch.allclose(outs[0], outs[1], atol=1e-12),
             f"{label} [{feat}]", fails, "no mask -> the scopes coincide")

    # nothing selected: charges pass through untouched, both features
    empty = torch.zeros(n, dtype=torch.bool)
    prior = torch.randn(n, generator=torch.Generator().manual_seed(2), dtype=DT)
    for feat in ('qeq', 'nqe'):
        les = Les({'is_periodic': periodic, f'use_{feat}': True,
                   f'{feat}_charge_scope': 'masked'}).to(DT)
        kw = dict(positions=r, cell=cells, atomic_numbers=z, batch=batch,
                  total_charge=qtot)
        q = (les(latent_chi=chi, latent_charges=prior, qeq_mask=empty, **kw)
             if feat == 'qeq' else
             les(latent_charges=prior, latent_fukui=f_raw, nqe_mask=empty, **kw)
             )['latent_charges']
        _chk(f"N empty", torch.equal(q, prior), f"{label} [{feat}]", fails,
             "nothing selected -> pass through")

    try:
        Les({'is_periodic': periodic, 'use_qeq': True, 'qeq_charge_scope': 'both'})
        _chk("N bad", False, label, fails, "an invalid scope was accepted")
    except ValueError:
        _chk("N bad", True, label, fails, "invalid scope raises")
    return fails


def check_energy_from_A(periodic, label):
    """O-1: E_lr taken from A equals E_lr from a full Ewald pass.

    With monopoles alone the long-range energy is 1/2 q^T A q, so the matrix built
    for the solve already carries it. Passing a zero dipole forces the Ewald pass
    instead, which is what this compares against.
    """
    fails = []
    for rsi in (True, False):
        for sigma, dl in SIGMA_DL[:2]:
            n = 8
            r, cells, batch, z, chi, _, _ = _system(periodic, n_per=(4, 4))
            args = {'is_periodic': periodic, 'sigma': sigma, 'dl': dl,
                    'remove_self_interaction': rsi, 'use_qeq': True}
            les = Les(args).to(DT)
            kw = dict(cell=cells, latent_chi=chi, atomic_numbers=z, batch=batch)
            rg = r.clone().requires_grad_(True)
            o1 = les(positions=rg, **kw)
            f1 = -torch.autograd.grad(o1['E_lr'].sum(), rg)[0]
            rg2 = r.clone().requires_grad_(True)
            o2 = les(positions=rg2, latent_dipoles=torch.zeros(n, 3, dtype=DT), **kw)
            f2 = -torch.autograd.grad(o2['E_lr'].sum(), rg2)[0]
            rel = ((o1['E_lr'] - o2['E_lr']).abs().max()
                   / o2['E_lr'].abs().max()).item()
            df = (f1 - f2).abs().max().item()
            _chk("O fromA", rel < 1e-10 and df < 1e-9,
                 f"{label} [rsi={rsi} s={sigma}]", fails, f"dE={rel:.1e} dF={df:.1e}")
    return fails


def check_padded(label):
    """O-2: the per-structure layout is the flat one, laid out differently.

    Non-periodic only -- see the note on compute_A_matrix_padded. Swept over uneven
    atom counts, exact and generous padding, both charge scopes, and with and
    without a mask, because padded slots are handled by the same identity-row trick
    the frozen atoms use and a bookkeeping slip would show up in exactly one of
    those corners.
    """
    fails = []
    bad = 0
    total = 0
    for n_per in ((5, 6, 7), (3, 9, 4, 8), (10, 10), (7,), (2, 11, 5)):
        n_max = max(n_per)
        for pad in (n_max, n_max + 3):
            for scope in ('system', 'masked'):
                for use_mask in (False, True):
                    total += 1
                    g = torch.Generator().manual_seed(11)
                    n = sum(n_per)
                    r, cells, batch, z, chi, _, _ = _system(False, n_per=n_per)
                    prior = torch.randn(n, generator=g, dtype=DT) * 0.3
                    qtot = torch.linspace(-1, 1, len(n_per)).to(DT)
                    mask = (torch.rand(n, generator=g) > 0.4) if use_mask else None
                    outs = []
                    for npad in (None, pad):
                        cfg = {'is_periodic': False, 'use_qeq': True,
                               'qeq_charge_scope': scope}
                        if npad is not None:
                            cfg['qeq_max_atoms_per_structure'] = npad
                        les = Les(cfg).to(DT)
                        rg = r.clone().requires_grad_(True)
                        o = les(positions=rg, cell=cells, latent_chi=chi,
                                atomic_numbers=z, batch=batch, latent_charges=prior,
                                total_charge=qtot, qeq_mask=mask)
                        f = -torch.autograd.grad(o['E_lr'].sum(), rg)[0]
                        outs.append((o['latent_charges'].detach(),
                                     o['E_lr'].detach(), f))
                    worst = max((outs[0][i] - outs[1][i]).abs().max().item()
                                for i in range(3))
                    if worst > 1e-11:
                        bad += 1
                        print(f"           mismatch n_per={n_per} pad={pad} "
                              f"{scope} mask={use_mask} -> {worst:.1e}")
    _chk("O padded", bad == 0, label, fails,
         f"{total} layouts (uneven counts x padding x scope x mask)")

    # a padding budget below the largest structure must fail loudly
    try:
        les = Les({'is_periodic': False, 'use_qeq': True,
                   'qeq_max_atoms_per_structure': 3}).to(DT)
        r, cells, batch, z, chi, _, _ = _system(False, n_per=(5, 6))
        les(positions=r, cell=cells, latent_chi=chi, atomic_numbers=z, batch=batch)
        _chk("O small", False, label, fails, "too small a padding was accepted")
    except (IndexError, RuntimeError):
        _chk("O small", True, label, fails, "too small a padding raises")

    # and it is refused on the periodic path, where it would not pay
    try:
        Les({'is_periodic': True, 'use_qeq': True,
             'qeq_max_atoms_per_structure': 8})
        _chk("O per", False, label, fails, "accepted on the periodic path")
    except ValueError:
        _chk("O per", True, label, fails, "refused on the periodic path")
    return fails


def check_guards(periodic, label):
    """K: refuse the combinations whose result would be meaningless.

    latent_kappas adds -kappa*phi to the charges *after* the solve, which breaks
    the total charge Qeq just enforced; Qeq is the self-consistent version of that
    same response, so the two together are both wrong and redundant.
    """
    fails = []
    r, cells, batch, z, chi, q0, _ = _system(periodic, n_per=(4, 5))
    kw = dict(positions=r, cell=cells, atomic_numbers=z, batch=batch)
    les = Les({'is_periodic': periodic, 'use_qeq': True}).to(DT)

    try:
        les(latent_chi=chi, latent_kappas=torch.full_like(chi, 0.3), **kw)
        _chk("K kappa", False, label, fails, "Qeq + kappa was accepted")
    except ValueError:
        _chk("K kappa", True, label, fails, "Qeq + kappa raises ValueError")

    try:
        les(latent_chi=chi, **{k: v for k, v in kw.items() if k != 'atomic_numbers'})
        _chk("K noZ", False, label, fails, "missing atomic_numbers was accepted")
    except ValueError:
        _chk("K noZ", True, label, fails, "missing atomic_numbers raises")

    # the multipoles that DO compose must keep working
    n = r.shape[0]
    quad = torch.rand(n, 3, 3, dtype=DT)
    out = les(latent_chi=chi, latent_dipoles=torch.rand(n, 3, dtype=DT),
              latent_quads=0.5 * (quad + quad.transpose(1, 2)),
              latent_alphas=torch.full((n,), 0.3, dtype=DT), **kw)
    _chk("K multi", bool(torch.isfinite(out['E_lr']).all()), label, fails,
         "dipole/quad/alpha still compose with Qeq")

    # mutually exclusive flags are refused at construction
    try:
        Les({'is_periodic': periodic, 'use_qeq': True, 'use_nqe': True})
        _chk("K both", False, label, fails, "use_qeq + use_nqe was accepted")
    except ValueError:
        _chk("K both", True, label, fails, "use_qeq + use_nqe raises")
    return fails


def check_minimum(periodic, label):
    """I: perturbing q along any charge-conserving direction must raise the objective.

    check_solve already shows the residual vanishes and A + J is positive definite;
    together with this, the solve is the constrained minimum rather than a saddle.
    """
    fails = []
    r, cells, batch, z, chi, _, onehot = _system(periodic)
    B = cells.shape[0]
    qtot = torch.tensor([0.0, 1.0, -2.0], dtype=DT)[:B]
    ew = Ewald_vectorized(is_periodic=periodic).to(DT)
    qeq = Qeq().to(DT)
    A = ew.compute_A_matrix(r, cells, batch, self_interaction=True)
    q, _ = qeq(A, chi, z, batch, qtot, torch.zeros_like(chi),
               torch.ones_like(batch, dtype=torch.bool))
    J = torch.nn.functional.softplus(qeq.J_raw.to(DT))[z] + qeq.eps_j
    chi_pos = torch.nn.functional.softplus(chi)

    def objective(qq):
        return 0.5 * (qq * ((A + torch.diag(J)) @ qq)).sum() + (chi_pos * qq).sum()

    base = objective(q)
    worst = float('inf')
    g = torch.Generator().manual_seed(1)
    for _ in range(20):
        # a random direction that keeps every structure's total charge fixed
        d = torch.randn(q.shape[0], generator=g, dtype=DT)
        counts = onehot.T @ onehot.sum(dim=1)
        d = d - onehot.T @ (onehot @ d) / counts
        worst = min(worst, (objective(q + 0.01 * d) - base).item())
    _chk("I minimum", worst > 0.0, label, fails, f"worst delta={worst:.3e}")
    return fails


def check_trainability(periodic, label):
    """J: dE/dchi and dE/dJ_raw are finite and nonzero."""
    fails = []
    r, cells, batch, z, chi, _, _ = _system(periodic, n_per=(4, 5))
    qtot = torch.zeros(cells.shape[0], dtype=DT)
    les = Les({'is_periodic': periodic, 'use_qeq': True}).to(DT)
    chi_g = chi.clone().requires_grad_(True)
    E = les(positions=r, cell=cells, latent_chi=chi_g, atomic_numbers=z, batch=batch,
            total_charge=qtot)['E_lr'].sum()
    E.backward()
    ok_chi = chi_g.grad is not None and bool(torch.isfinite(chi_g.grad).all()) \
        and chi_g.grad.abs().max().item() > 1e-12
    jg = les.qeq.J_raw.grad
    # only the elements present in the batch may receive a gradient
    present = torch.zeros_like(jg, dtype=torch.bool)
    present[z] = True
    ok_j = jg is not None and bool(torch.isfinite(jg).all()) \
        and jg[present].abs().max().item() > 1e-12 \
        and jg[~present].abs().max().item() == 0.0
    _chk("J grad-chi", ok_chi, label, fails)
    _chk("J grad-J", ok_j, label, fails, "only present elements get a gradient")
    return fails


def check_torchscript(periodic, use_qeq, label):
    """H: script, save, reload, and still agree with eager."""
    import tempfile, os
    fails = []
    r, cells, batch, z, chi, q0, _ = _system(periodic)
    qtot = torch.zeros(cells.shape[0], dtype=DT)
    args = {'is_periodic': periodic}
    args.update({'use_qeq': True} if use_qeq else {'use_nqe': True})
    les = Les(args).to(DT)
    kw = dict(positions=r, cell=cells, atomic_numbers=z, batch=batch, total_charge=qtot)
    if use_qeq:
        kw['latent_chi'] = chi
    else:
        kw['latent_charges'] = q0
        kw['latent_fukui'] = chi
    ref = les(**kw)
    try:
        sm = torch.jit.script(les)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.pt")
            torch.jit.save(sm, p)
            sm = torch.jit.load(p)
        got = sm(**kw)
        ok = set(got.keys()) == set(ref.keys())
        for k in ref:
            if ref[k] is None:
                ok = ok and got[k] is None
            else:
                ok = ok and got[k] is not None and torch.allclose(got[k], ref[k], atol=1e-9)
        _chk("H script", ok, label, fails)
    except Exception as e:
        print(f"[H script ] FAILED -> {type(e).__name__}: {str(e)[:200]}")
        fails.append(f"{label}:torchscript-error")
    return fails


def check_back_compat(label):
    """Old pickles have no use_qeq/use_nqe in their __dict__ and no submodule."""
    import io
    fails = []
    les = Les({'is_periodic': True})
    buf = io.BytesIO()
    torch.save(les, buf)
    buf.seek(0)
    obj = torch.load(buf, weights_only=False)
    for flag in ('use_qeq', 'use_nqe', 'qeq_self_interaction'):
        obj.__dict__.pop(flag, None)
    buf2 = io.BytesIO()
    torch.save(obj, buf2)
    buf2.seek(0)
    migrated = torch.load(buf2, weights_only=False)
    ok = all(hasattr(migrated, f) for f in ('use_qeq', 'use_nqe', 'qeq_self_interaction'))
    ok = ok and migrated.use_qeq is False and migrated.use_nqe is False
    r, cells, batch, _, _, q0, _ = _system(True)
    out = migrated(positions=r, cell=cells, latent_charges=q0, batch=batch)
    ok = ok and bool(torch.isfinite(out['E_lr']).all())
    # the output contract is unchanged: exactly the keys that existed before
    ok = ok and set(out.keys()) == {'E_lr', 'latent_charges', 'latent_dipoles',
                                    'latent_quads', 'latent_alphas', 'BEC'}
    _chk("setstate", ok, label, fails)
    return fails


def main():
    fails = []
    for periodic, pname in ((True, "periodic"), (False, "realspace")):
        print(f"\n================ {pname} ================")
        for rsi in (True, False):
            for sigma, dl in SIGMA_DL:
                fails += check_units(periodic, rsi, sigma, dl,
                                     f"{pname} [rsi={rsi} s={sigma} dl={dl}]")
        fails += check_solve(periodic, pname)
        fails += check_gauge(periodic, pname)
        fails += check_forces(periodic, pname)
        fails += check_mask(periodic, pname)
        fails += check_no_self(periodic, pname)
        fails += check_guards(periodic, pname)
        fails += check_charge_scope(periodic, pname)
        fails += check_energy_from_A(periodic, pname)
        fails += check_minimum(periodic, pname)
        fails += check_trainability(periodic, pname)
        fails += check_torchscript(periodic, True, f"{pname} [qeq]")
        fails += check_torchscript(periodic, False, f"{pname} [nqe]")
    print("\n================ shared ================")
    fails += check_padded("padded layout")
    fails += check_nqe("nqe")
    fails += check_back_compat("back-compat")

    print("\n==================== SUMMARY ====================")
    if fails:
        print("FAILURES:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("All Qeq / NQE checks passed.")


if __name__ == "__main__":
    main()
