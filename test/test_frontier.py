"""
Frontier charge allocation: Model 0 and Model B.

  A  per-structure normalization, on uneven and unsorted batches
  B  charge bookkeeping across constraint on/off, m, non-neutral Q_ref, epsilon_e
  C  carrier sign: an electron lowers the total by m, a hole raises it
  D  no NaN/Inf from large offsets, large beta, ties, or a single candidate
  E  Model 0 gauge: a common shift of epsilon leaves p, q, E and F alone
  F  Model B gauge: a common shift of a leaves p and q alone and moves E by m*C
  G  the candidate cost formula equals feeding z + d^(i) to the Ewald directly
  H  and it keeps doing so across boundary condition, self term and sigma
  I  Model B's two-step reference correction: r and p are distinct, the total exact
  J  no double counting: E_B is the smooth minimum over candidate TOTAL energies,
     which an implementation adding U(z_report) fails
  K  forces: full autograd against central differences, with geometry-dependent
     charges and heads so the whole chain is exercised
  L  gradients reach the charge, epsilon and a heads; gradcheck on a small case
  M  a distant charge reorders Model B's candidates at fixed local a
  N  spectator candidates dilute p and shift G by log(count)/beta
  O  the existing API is untouched: key set, positional call, state dict, old pickle
  P  TorchScript round-trip
  Q  torch.compile against eager

Run:
    PYTHONPATH=src python test/test_frontier.py
"""
import io
import math
import sys
import tempfile

import torch

from les import Les
from les.module import Model0, ModelB, quadratic_energy, segment_onehot

DT = torch.float64


def _chk(gate, ok, label, fails, note=""):
    print(f"[{gate:10s}] {'ok  ' if ok else 'FAIL'} {label:34s} {note}")
    if not ok:
        fails.append(f"{label}:{gate.strip()}")


def _system(periodic, n_per=(5, 6, 7), seed=42, spread=7.0):
    """A batch of structures; zero cells are how LES marks non-periodic."""
    g = torch.Generator().manual_seed(seed)
    n = sum(n_per)
    batch = torch.cat([torch.full((k,), i) for i, k in enumerate(n_per)]).long()
    r = torch.rand(n, 3, generator=g, dtype=DT) * spread
    if periodic:
        cells = torch.stack([
            torch.eye(3, dtype=DT) * 12.0,
            torch.tensor([[11., .4, 0.], [.3, 12., .2], [0., .1, 13.]], dtype=DT),
            torch.diag(torch.tensor([11., 12., 14.], dtype=DT)),
        ])[:len(n_per)]
    else:
        cells = torch.zeros(len(n_per), 3, 3, dtype=DT)
    z = torch.randn(n, generator=g, dtype=DT) * 0.3
    eps = torch.randn(n, generator=g, dtype=DT)
    a = torch.randn(n, generator=g, dtype=DT)
    return r, cells, batch, z, eps, a


def _onehot(batch, n_batch):
    return segment_onehot(batch, n_batch, DT)


def _run(les, r, cells, batch, z, head, m, q_ref=None, model_b=False):
    kw = dict(positions=r, cell=cells, latent_charges=z, batch=batch,
              carrier_count=m, reference_total_charge=q_ref)
    kw['latent_local_addition' if model_b else 'latent_frontier'] = head
    return les(**kw)


def _les(periodic, model_b, **cfg):
    args = {'is_periodic': periodic}
    args['use_model_b' if model_b else 'use_model0'] = True
    args.update(cfg)
    return Les(args).to(DT)


# ----------------------------------------------------------------------------
def check_normalization():
    """A: sum(p) is one per structure, with no leakage between them."""
    fails = []
    for n_per in ((5, 6, 7), (3, 9, 4, 8), (11,), (2, 2, 2, 2, 2)):
        for model_b in (False, True):
            r, cells, batch, z, eps, a = _system(False, n_per=n_per)
            B = len(n_per)
            les = _les(False, model_b)
            out = _run(les, r, cells, batch, z, a if model_b else eps,
                       torch.zeros(B, dtype=DT), model_b=model_b)
            p = out['occupation']
            sums = _onehot(batch, B) @ p
            ok = (sums - 1.0).abs().max().item() < 1e-12
            _chk(f"A {'B' if model_b else '0'}", ok, f"n_per={n_per}", fails,
                 f"max|sum(p)-1|={(sums - 1.0).abs().max().item():.1e}")

    # an unsorted batch must give the same answer as the sorted one, atom for atom
    for model_b in (False, True):
        r, cells, batch, z, eps, a = _system(False, n_per=(4, 5, 6))
        head = a if model_b else eps
        les = _les(False, model_b)
        m = torch.tensor([0.0, 1.0, 1.0], dtype=DT)
        ref = _run(les, r, cells, batch, z, head, m, model_b=model_b)
        perm = torch.randperm(len(batch), generator=torch.Generator().manual_seed(1))
        got = _run(les, r[perm], cells, batch[perm], z[perm], head[perm], m,
                   model_b=model_b)
        ok = (torch.allclose(got['occupation'], ref['occupation'][perm], atol=1e-12)
              and torch.allclose(got['latent_charges'],
                                 ref['latent_charges'][perm], atol=1e-12)
              and torch.allclose(got['E_lr'], ref['E_lr'], atol=1e-10))
        _chk(f"A unsort", ok, f"{'B' if model_b else '0'} shuffled atoms", fails)

    # moving one structure's scores must not touch another's charges
    for model_b in (False, True):
        r, cells, batch, z, eps, a = _system(False, n_per=(4, 5))
        head = (a if model_b else eps).clone()
        les = _les(False, model_b)
        m = torch.ones(2, dtype=DT)
        ref = _run(les, r, cells, batch, z, head, m, model_b=model_b)
        bumped = head.clone()
        bumped[batch == 0] += 3.0
        got = _run(les, r, cells, batch, z, bumped, m, model_b=model_b)
        far = batch == 1
        ok = torch.allclose(got['occupation'][far], ref['occupation'][far], atol=1e-12)
        _chk(f"A isolate", ok, f"{'B' if model_b else '0'} no cross-structure leak",
             fails)
    return fails


def check_charge_bookkeeping():
    """B: the total charge follows the formula in every combination.

    constrained: physical total is Q_ref + c*m.
    unconstrained: the total is whatever the model made, but the CHANGE from the
    carrier is exactly c*m, which is the part the constraint switch does not alter.
    """
    fails = []
    for model_b in (False, True):
        for periodic in (False, True):
            for eps_e in (1.0, 4.0):
                for constrain in (False, True):
                    r, cells, batch, z, eps, a = _system(periodic, n_per=(5, 6))
                    B, s = 2, math.sqrt(eps_e)
                    q_ref = torch.tensor([1.3, -2.0], dtype=DT)   # non-neutral
                    les = _les(periodic, model_b, frontier_epsilon_e=eps_e,
                               frontier_constrain_charge=constrain)
                    head = a if model_b else eps
                    oh = _onehot(batch, B)
                    tot = {}
                    for mv in (0.0, 1.0):
                        m = torch.full((B,), mv, dtype=DT)
                        out = _run(les, r, cells, batch, z, head, m,
                                   q_ref if constrain else None, model_b=model_b)
                        tot[mv] = s * (oh @ out['latent_charges'])
                    if constrain:
                        worst = max((tot[mv] - (q_ref - mv)).abs().max().item()
                                    for mv in (0.0, 1.0))
                        note = f"|total-(Q_ref+c m)|={worst:.1e}"
                    else:
                        worst = (tot[1.0] - tot[0.0] + 1.0).abs().max().item()
                        note = f"|delta-(c m)|={worst:.1e}"
                    _chk(f"B {'B' if model_b else '0'}", worst < 1e-11,
                         f"{'per' if periodic else 'real'} eps_e={eps_e} "
                         f"constr={constrain}", fails, note)
    return fails


def check_sign():
    """C: an electron lowers the physical total by m, a hole raises it."""
    fails = []
    for model_b in (False, True):
        r, cells, batch, z, eps, a = _system(False, n_per=(5, 6))
        oh = _onehot(batch, 2)
        head = a if model_b else eps
        for sign in (-1, +1):
            les = _les(False, model_b, frontier_carrier_sign=sign)
            base = _run(les, r, cells, batch, z, head, torch.zeros(2, dtype=DT),
                        model_b=model_b)['latent_charges']
            one = _run(les, r, cells, batch, z, head, torch.ones(2, dtype=DT),
                       model_b=model_b)['latent_charges']
            delta = (oh @ (one - base))          # LES units, epsilon_e = 1
            ok = (delta - sign).abs().max().item() < 1e-12
            _chk(f"C {'B' if model_b else '0'}", ok,
                 f"carrier_sign={sign:+d}", fails,
                 f"delta={delta[0].item():+.6f}")
    return fails


def check_stability():
    """D: extreme but legal inputs must not produce NaN or Inf."""
    fails = []
    cases = {
        "large offset": lambda h: h + 5.0e3,
        "large beta": None,
        "exact ties": lambda h: torch.zeros_like(h),
        "wide spread": lambda h: h * 500.0,
    }
    for model_b in (False, True):
        for name, tweak in cases.items():
            beta = 200.0 if name == "large beta" else 1.0
            r, cells, batch, z, eps, a = _system(False, n_per=(5, 6))
            head = a if model_b else eps
            if tweak is not None:
                head = tweak(head)
            les = _les(False, model_b, frontier_beta=beta)
            out = _run(les, r, cells, batch, z, head, torch.ones(2, dtype=DT),
                       model_b=model_b)
            vals = [out['occupation'], out['latent_charges'], out['E_lr']]
            ok = all(bool(torch.isfinite(v).all()) for v in vals)
            sums = _onehot(batch, 2) @ out['occupation']
            ok = ok and (sums - 1.0).abs().max().item() < 1e-10
            _chk(f"D {'B' if model_b else '0'}", ok, name, fails)

    # a structure with exactly one candidate: p must be 1, not NaN
    for model_b in (False, True):
        r, cells, batch, z, eps, a = _system(False, n_per=(1, 4))
        les = _les(False, model_b)
        out = _run(les, r, cells, batch, z, a if model_b else eps,
                   torch.ones(2, dtype=DT), model_b=model_b)
        p = out['occupation']
        ok = bool(torch.isfinite(p).all()) and abs(p[0].item() - 1.0) < 1e-12
        _chk(f"D single", ok, f"{'B' if model_b else '0'} one candidate", fails,
             f"p={p[0].item():.6f}")
    return fails


def check_gauge_model0():
    """E: epsilon is a selection score, so a per-structure offset changes nothing.

    This is why energy and force training cannot fix its absolute scale, and why
    calling it an electron affinity would overclaim.
    """
    fails = []
    for periodic in (False, True):
        for constrain in (False, True):
            r, cells, batch, z, eps, a = _system(periodic, n_per=(5, 6))
            q_ref = torch.tensor([0.5, -1.0], dtype=DT)
            les = _les(periodic, False, frontier_constrain_charge=constrain)
            m = torch.ones(2, dtype=DT)
            outs = []
            for shift in (0.0, 7.0):
                bumped = eps + shift * (batch == 0).to(DT) - 3.0 * (batch == 1).to(DT)
                rg = r.clone().requires_grad_(True)
                o = _run(les, rg, cells, batch, z, bumped, m,
                         q_ref if constrain else None)
                f = -torch.autograd.grad(o['E_lr'].sum(), rg)[0]
                outs.append((o['occupation'].detach(), o['latent_charges'].detach(),
                             o['E_lr'].detach(), f))
            worst = max((outs[0][i] - outs[1][i]).abs().max().item() for i in range(4))
            _chk("E gauge", worst < 1e-11,
                 f"{'per' if periodic else 'real'} constr={constrain}", fails,
                 f"max|d(p,q,E,F)|={worst:.1e}")
    return fails


def check_gauge_model_b():
    """F: shifting a by C leaves p and the charges alone and moves E by exactly m*C."""
    fails = []
    for periodic in (False, True):
        for constrain in (False, True):
            r, cells, batch, z, eps, a = _system(periodic, n_per=(5, 6))
            q_ref = torch.tensor([0.5, -1.0], dtype=DT)
            les = _les(periodic, True, frontier_constrain_charge=constrain)
            m = torch.tensor([0.0, 1.0], dtype=DT)
            shift = torch.tensor([2.0, -3.0], dtype=DT)
            oh = _onehot(batch, 2)
            outs = []
            for use_shift in (False, True):
                head = a + (oh.transpose(0, 1) @ shift if use_shift else 0.0)
                outs.append(_run(les, r, cells, batch, z, head, m,
                                 q_ref if constrain else None, model_b=True))
            d_pq = max((outs[0][k] - outs[1][k]).abs().max().item()
                       for k in ('occupation', 'latent_charges', 'reference_charges'))
            d_E = (outs[1]['E_lr'] - outs[0]['E_lr'] - m * shift).abs().max().item()
            _chk("F gauge", d_pq < 1e-11 and d_E < 1e-10,
                 f"{'per' if periodic else 'real'} constr={constrain}", fails,
                 f"|d(p,q)|={d_pq:.1e} |dE-mC|={d_E:.1e}")
    return fails


def check_candidate_energy():
    """G/H: a + k(Kz)_i + k^2 K_ii/2 is the exact Ewald cost of moving the carrier.

    Swept over boundary condition, the self-interaction switch and non-unit sigma,
    because the self term is where the two A-matrix conventions differ and sigma=1
    hides a wrong exponent.
    """
    fails = []
    for periodic in (False, True):
        for rsi in (True, False):
            for sigma in (1.0, 1.3):
                for eps_e in (1.0, 4.0):
                    r, cells, batch, z, eps, a = _system(periodic, n_per=(5,))
                    plain = Les({'is_periodic': periodic, 'sigma': sigma,
                                 'remove_self_interaction': rsi}).to(DT)
                    K = plain.ewald.compute_A_matrix(r, cells, batch, not rsi)
                    mb = ModelB(epsilon_e=eps_e).to(DT)
                    formula = mb.candidate_costs(a, K, z)
                    k = mb.carrier_sign / mb.scale
                    base = plain(positions=r, cell=cells, latent_charges=z,
                                 batch=batch)['E_lr']
                    direct = []
                    for i in range(len(z)):
                        zi = z.clone()
                        zi[i] = zi[i] + k
                        direct.append((a[i] + plain(positions=r, cell=cells,
                                                    latent_charges=zi,
                                                    batch=batch)['E_lr'] - base).item())
                    d = (formula - torch.tensor(direct, dtype=DT)).abs().max().item()
                    _chk("G/H cost", d < 1e-10,
                         f"{'per' if periodic else 'real'} rsi={rsi} s={sigma} "
                         f"eps_e={eps_e}", fails, f"max|d|={d:.1e}")
    return fails


def check_reference_correction():
    """I: the correction weight and the occupation are computed in a fixed order.

    They are different quantities -- one normalises charges the model got wrong, the
    other places a carrier -- so a non-zero deficit must make them differ while the
    total stays exact.
    """
    fails = []
    for periodic in (False, True):
        r, cells, batch, z, eps, a = _system(periodic, n_per=(5, 6))
        q_ref = torch.tensor([2.0, -1.5], dtype=DT)
        oh = _onehot(batch, 2)
        les = _les(periodic, True, frontier_constrain_charge=True)
        m = torch.ones(2, dtype=DT)
        out = _run(les, r, cells, batch, z, a, m, q_ref, model_b=True)

        # rebuild the two weights by hand from the documented recipe
        mb = les.model_b
        K = les.ewald.compute_A_matrix(r, cells, batch, not les.remove_self_interaction)
        from les.module import segment_softmax
        w = segment_softmax(-mb.beta * mb.candidate_costs(a, K, z), oh)
        deficit = q_ref / mb.scale - oh @ z
        z_ref = z + w * (oh.transpose(0, 1) @ deficit)
        p = segment_softmax(-mb.beta * mb.candidate_costs(a, K, z_ref), oh)

        d_ref = (out['reference_charges'] - z_ref).abs().max().item()
        d_p = (out['occupation'] - p).abs().max().item()
        differ = (w - p).abs().max().item()
        tot = (oh @ out['latent_charges'] - (q_ref - m)).abs().max().item()
        _chk("I order", d_ref < 1e-12 and d_p < 1e-12,
             f"{'per' if periodic else 'real'} recipe matches", fails,
             f"|dz_ref|={d_ref:.1e} |dp|={d_p:.1e}")
        _chk("I distinct", differ > 1e-6, f"{'per' if periodic else 'real'} r != p",
             fails, f"max|r-p|={differ:.2e}")
        _chk("I total", tot < 1e-11, f"{'per' if periodic else 'real'} exact total",
             fails, f"|d|={tot:.1e}")
    return fails


def check_no_double_counting():
    """J: E_B is the smooth minimum over what the carrier would cost on each site.

    With m = 1, U(z_ref) + softmin(epsilon) telescopes into softmin over the candidates'
    TOTAL energies. An implementation that also adds U(z_report) fails this by an
    amount far larger than any tolerance, so the check has teeth.
    """
    fails = []
    for periodic in (False, True):
        for beta in (1.0, 5.0):
            r, cells, batch, z, eps, a = _system(periodic, n_per=(2,), spread=3.0)
            les = _les(periodic, True, frontier_beta=beta)
            out = _run(les, r, cells, batch, z, a, torch.ones(1, dtype=DT),
                       model_b=True)
            plain = Les({'is_periodic': periodic}).to(DT)
            k = les.model_b.carrier_sign / les.model_b.scale
            totals = []
            for i in range(len(z)):
                zi = z.clone()
                zi[i] = zi[i] + k
                totals.append((a[i] + plain(positions=r, cell=cells, latent_charges=zi,
                                            batch=batch)['E_lr']).item())
            softmin = (-torch.logsumexp(-beta * torch.tensor(totals, dtype=DT), dim=0)
                       / beta)
            d = abs(out['E_lr'].item() - softmin.item())
            wrong = plain(positions=r, cell=cells,
                          latent_charges=out['latent_charges'],
                          batch=batch)['E_lr'] + out['E_frontier']
            gap = abs(wrong.item() - softmin.item())
            _chk("J softmin", d < 1e-10,
                 f"{'per' if periodic else 'real'} beta={beta}", fails,
                 f"|dE|={d:.1e}; a U(z_report) build is off by {gap:.2e}")
            _chk("J teeth", gap > 1e-3, f"{'per' if periodic else 'real'} beta={beta}",
                 fails, "the wrong build really is distinguishable")

            parts = (out['E_reference_lr'] + out['E_frontier'] - out['E_lr'])
            _chk("J parts", parts.abs().max().item() < 1e-12,
                 f"{'per' if periodic else 'real'} beta={beta}", fails,
                 "E_lr = E_reference_lr + E_frontier")
    return fails


def _geometry_coupled(r, batch, n_batch):
    """Charges and heads that actually depend on the positions.

    Differentiating a model whose charges were constants would miss every path the
    deployed graph takes, which is the failure the LES harness was built around.
    """
    d = torch.linalg.norm(r, dim=-1)
    z = 0.21 * torch.sin(1.3 * d) + 0.05 * r[:, 0]
    head = 0.7 * torch.cos(0.9 * d) + 0.11 * r[:, 1]
    return z, head


def check_forces():
    """K: autograd forces against central differences, charges and heads included."""
    fails = []
    for periodic in (False, True):
        for model_b in (False, True):
            for constrain in (False, True):
                r, cells, batch, _, _, _ = _system(periodic, n_per=(3, 4))
                q_ref = torch.tensor([0.5, -1.0], dtype=DT)
                les = _les(periodic, model_b, frontier_constrain_charge=constrain)
                m = torch.tensor([1.0, 1.0], dtype=DT)

                def energy(pos):
                    z, head = _geometry_coupled(pos, batch, 2)
                    return _run(les, pos, cells, batch, z, head, m,
                                q_ref if constrain else None,
                                model_b=model_b)['E_lr'].sum()

                rg = r.clone().requires_grad_(True)
                f_auto = -torch.autograd.grad(energy(rg), rg)[0]
                h = 1e-6
                f_fd = torch.zeros_like(r)
                for i in range(r.shape[0]):
                    for c in range(3):
                        rp = r.clone(); rp[i, c] += h
                        rm = r.clone(); rm[i, c] -= h
                        f_fd[i, c] = -(energy(rp) - energy(rm)) / (2 * h)
                err = (f_auto - f_fd).abs().max().item()
                scale = max(f_fd.abs().max().item(), 1e-12)
                _chk("K force", err / scale < 1e-6,
                     f"{'per' if periodic else 'real'} "
                     f"{'B' if model_b else '0'} constr={constrain}", fails,
                     f"rel={err / scale:.1e}")
    return fails


def check_training_gradients():
    """L: gradients reach every head, and gradcheck passes on a small case."""
    fails = []
    for model_b in (False, True):
        r, cells, batch, z, eps, a = _system(False, n_per=(3, 4))
        les = _les(False, model_b, frontier_constrain_charge=True)
        q_ref = torch.tensor([0.5, -1.0], dtype=DT)
        m = torch.tensor([0.0, 1.0], dtype=DT)
        zg = z.clone().requires_grad_(True)
        hg = (a if model_b else eps).clone().requires_grad_(True)
        rg = r.clone().requires_grad_(True)
        out = _run(les, rg, cells, batch, zg, hg, m, q_ref, model_b=model_b)
        # a force loss, so the second derivative path is exercised too
        f = -torch.autograd.grad(out['E_lr'].sum(), rg, create_graph=True)[0]
        (f ** 2).sum().backward()
        ok = all(t.grad is not None and bool(torch.isfinite(t.grad).all())
                 and t.grad.abs().max().item() > 0 for t in (zg, hg))
        _chk("L grads", ok, f"{'B' if model_b else '0'} q and head receive gradient",
             fails)

        # m = 0 still trains the head through the reference correction
        m0 = torch.zeros(2, dtype=DT)
        hg2 = (a if model_b else eps).clone().requires_grad_(True)
        o2 = _run(les, r, cells, batch, z, hg2, m0, q_ref, model_b=model_b)
        o2['E_lr'].sum().backward()
        _chk("L m=0", hg2.grad is not None and hg2.grad.abs().max().item() > 0,
             f"{'B' if model_b else '0'} head trains at m=0", fails,
             "via the reference correction")

    for model_b in (False, True):
        r, cells, batch, z, eps, a = _system(False, n_per=(3,), seed=5)
        les = _les(False, model_b, frontier_constrain_charge=True)
        q_ref = torch.tensor([0.4], dtype=DT)
        m = torch.ones(1, dtype=DT)

        def fn(zz, hh):
            return _run(les, r, cells, batch, zz, hh, m, q_ref,
                        model_b=model_b)['E_lr'].sum()

        args = (z.clone().requires_grad_(True),
                (a if model_b else eps).clone().requires_grad_(True))
        try:
            ok = torch.autograd.gradcheck(fn, args, eps=1e-6, atol=1e-7)
            ok = ok and torch.autograd.gradgradcheck(fn, args, atol=1e-5)
        except Exception as exc:
            ok = False
            print(f"             gradcheck raised {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:70]}")
        _chk("L gradchk", bool(ok), f"{'B' if model_b else '0'} gradcheck", fails)
    return fails


def check_remote_electrostatics():
    """M: at identical local a, a distant charge decides where the carrier goes.

    Two equivalent sites and one spectator carrying charge. Moving the spectator from
    one side to the other must flip which site the carrier prefers -- information a
    purely local epsilon head has no access to.
    """
    fails = []
    for periodic in (False, True):
        cells = ((torch.eye(3, dtype=DT) * 40.0).unsqueeze(0) if periodic
                 else torch.zeros(1, 3, 3, dtype=DT))
        batch = torch.zeros(3, dtype=torch.long)
        a = torch.zeros(3, dtype=DT)              # identical local chemistry
        prefs = []
        for spectator_x in (-6.0, 14.0):          # near site 0, then near site 1
            r = torch.tensor([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0],
                              [spectator_x, 0.0, 0.0]], dtype=DT)
            z = torch.tensor([0.0, 0.0, 0.6], dtype=DT)     # only the spectator charged
            les = _les(periodic, True, frontier_beta=5.0)
            out = _run(les, r, cells, batch, z, a, torch.ones(1, dtype=DT),
                       model_b=True)
            p = out['occupation']
            prefs.append(p[0].item() - p[1].item())
        flipped = prefs[0] * prefs[1] < 0 and min(abs(p) for p in prefs) > 1e-3
        _chk("M remote", flipped, f"{'per' if periodic else 'real'} ordering flips",
             fails, f"p0-p1: {prefs[0]:+.4f} -> {prefs[1]:+.4f}")

    # Model 0's epsilon is supplied by the host, so LES itself cannot react here
    r = torch.tensor([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [-6.0, 0.0, 0.0]], dtype=DT)
    cells = torch.zeros(1, 3, 3, dtype=DT)
    batch = torch.zeros(3, dtype=torch.long)
    les0 = _les(False, False)
    outs = [_run(les0, r, cells, batch, torch.tensor([0.0, 0.0, q], dtype=DT),
                 torch.zeros(3, dtype=DT), torch.ones(1, dtype=DT))['occupation']
            for q in (0.6, -0.6)]
    _chk("M local0", torch.allclose(outs[0], outs[1], atol=1e-12),
         "Model 0 p ignores remote charge", fails,
         "by construction: epsilon comes from the host")
    return fails


def check_spectators():
    """N: extra equivalent candidates dilute p and shift G by log(count)/beta."""
    fails = []
    beta = 4.0
    for n_extra in (1, 3, 9):
        # one favoured site plus n_extra equivalent ones, all far apart so the
        # electrostatic part of the cost is common and the algebra is clean
        n = 1 + n_extra
        r = torch.stack([torch.tensor([60.0 * i, 0.0, 0.0], dtype=DT)
                         for i in range(n)])
        cells = torch.zeros(1, 3, 3, dtype=DT)
        batch = torch.zeros(n, dtype=torch.long)
        z = torch.zeros(n, dtype=DT)
        delta = 0.35
        a = torch.zeros(n, dtype=DT)
        a[0] = -delta                       # site 0 cheaper by delta
        les = _les(False, True, frontier_beta=beta)
        out = _run(les, r, cells, batch, z, a, torch.ones(1, dtype=DT), model_b=True)
        p_best = 1.0 / (1.0 + n_extra * math.exp(-beta * delta))
        d = abs(out['occupation'][0].item() - p_best)
        _chk("N dilute", d < 1e-9, f"{n_extra} spectators", fails,
             f"p_best={out['occupation'][0].item():.6f} vs {p_best:.6f}")

    # replicating an identical candidate set K times shifts G by -log(K)/beta
    gs = []
    for K in (1, 2, 4):
        n = 2 * K
        r = torch.stack([torch.tensor([60.0 * i, 0.0, 0.0], dtype=DT)
                         for i in range(n)])
        cells = torch.zeros(1, 3, 3, dtype=DT)
        batch = torch.zeros(n, dtype=torch.long)
        a = torch.tensor([0.0, 0.4], dtype=DT).repeat(K)
        les = _les(False, True, frontier_beta=beta)
        out = _run(les, r, cells, batch, torch.zeros(n, dtype=DT), a,
                   torch.ones(1, dtype=DT), model_b=True)
        gs.append(out['E_frontier'].item())          # m = 1, so this is G
    shifts = [gs[0] - gs[i] - math.log(k) / beta
              for i, k in enumerate((1, 2, 4))]
    worst = max(abs(s) for s in shifts)
    _chk("N replicate", worst < 1e-9 and abs(gs[0] - gs[2]) > 1e-3,
         "G shifts by log(K)/beta", fails,
         f"G = {gs[0]:.6f} -> {gs[1]:.6f} -> {gs[2]:.6f}, max|d|={worst:.1e}")
    return fails


def check_mini_training(model_b):
    """R: the architecture can fit a closed-form target it is able to represent.

    A donor and an acceptor plus one remote charge. The target comes from the same
    model with a known linear head, so an exact fit exists and a failure means the
    gradients do not carry usable signal -- not that the chemistry is right. It says
    nothing about real redox data or transferability.

    900 steps rather than a looser tolerance: at beta = 3 the occupation saturates to
    roughly [1, 0, 0], so the gradient reaching the sites the carrier avoids is small
    and Adam needs the room. That slow direction is a property of the model worth
    knowing before picking beta, not a reason to accept a worse fit.
    """
    fails = []
    torch.manual_seed(0)
    cells = torch.zeros(1, 3, 3, dtype=DT)
    batch = torch.zeros(3, dtype=torch.long)
    m = torch.ones(1, dtype=DT)
    w_true = torch.tensor([0.8, -0.35], dtype=DT)

    def features(pos):
        """per atom: a constant, and how close the remote charge is"""
        d = torch.linalg.norm(pos - pos[2], dim=-1)
        return torch.stack([torch.ones_like(d), torch.exp(-0.12 * d)], dim=-1)

    def geometries(count, seed):
        g = torch.Generator().manual_seed(seed)
        out = []
        for _ in range(count):
            x = 5.0 + 3.0 * torch.rand(1, generator=g, dtype=DT).item()
            sx = -4.0 + 16.0 * torch.rand(1, generator=g, dtype=DT).item()
            out.append(torch.tensor([[0., 0., 0.], [x, 0., 0.], [sx, 0.6, 0.]],
                                    dtype=DT))
        return out

    les = _les(False, model_b, frontier_beta=3.0)

    def predict(pos, w):
        z = torch.tensor([0.0, 0.0, 0.55], dtype=DT)
        head = features(pos) @ w
        return _run(les, pos, cells, batch, z, head, m, model_b=model_b)

    def energy_force(pos, w):
        pg = pos.clone().requires_grad_(True)
        e = predict(pg, w)['E_lr'].sum()
        f = -torch.autograd.grad(e, pg, create_graph=w.requires_grad)[0]
        return e, f

    train = geometries(12, 1)
    held = geometries(4, 99)
    targets = {}
    for pos in train + held:
        e, f = energy_force(pos, w_true)
        targets[id(pos)] = (e.detach(), f.detach())

    w = torch.zeros(2, dtype=DT, requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.05)
    for step in range(900):
        opt.zero_grad()
        loss = torch.zeros((), dtype=DT)
        for pos in train:
            e, f = energy_force(pos, w)
            te, tf = targets[id(pos)]
            loss = loss + (e - te) ** 2 + ((f - tf) ** 2).sum()
        loss.backward()
        opt.step()

    held_err = 0.0
    for pos in held:
        e, f = energy_force(pos, w.detach())
        te, tf = targets[id(pos)]
        held_err = max(held_err, abs((e - te).item()),
                       (f - tf).abs().max().item())
    _chk("R train", loss.item() < 1e-8, f"{'B' if model_b else '0'} fits the target",
         fails, f"final train loss={loss.item():.2e}")
    _chk("R held", held_err < 1e-4, f"{'B' if model_b else '0'} withheld geometries",
         fails, f"max|dE|,|dF| = {held_err:.2e}")
    return fails


def check_existing_api():
    """O: nothing above changes for a model that does not switch the features on."""
    fails = []
    base_keys = {'E_lr', 'latent_charges', 'latent_dipoles', 'latent_quads',
                 'latent_alphas', 'BEC'}
    r, cells, batch, z, eps, a = _system(True, n_per=(4, 5))
    plain = Les({'is_periodic': True}).to(DT)
    out = plain(positions=r, cell=cells, latent_charges=z, batch=batch)
    _chk("O keys", set(out.keys()) == base_keys, "key set unchanged", fails,
         f"{sorted(out.keys())}")

    # a positional call still lands on the same arguments
    pos = plain(r, cells, None, None, z, None, None, None, None, None, batch)
    _chk("O positional", torch.allclose(pos['E_lr'], out['E_lr'], atol=0, rtol=0),
         "positional call unaffected", fails)

    for model_b in (False, True):
        les = _les(True, model_b)
        extra = {'occupation', 'reference_charges'}
        if model_b:
            extra |= {'E_reference_lr', 'E_frontier'}
        o = _run(les, r, cells, batch, z, a if model_b else eps,
                 torch.ones(2, dtype=DT), model_b=model_b)
        _chk("O extra", set(o.keys()) == base_keys | extra,
             f"{'B' if model_b else '0'} adds only its own keys", fails,
             f"+{sorted(extra)}")
        _chk("O sd", list(les.state_dict().keys()) == [],
             f"{'B' if model_b else '0'} adds no parameters", fails)

    # an old pickle predates the flags entirely
    buf = io.BytesIO()
    torch.save(plain, buf)
    buf.seek(0)
    obj = torch.load(buf, weights_only=False)
    for flag in ('use_model0', 'use_model_b'):
        obj.__dict__.pop(flag, None)
    buf2 = io.BytesIO()
    torch.save(obj, buf2)
    buf2.seek(0)
    migrated = torch.load(buf2, weights_only=False)
    ok = (all(hasattr(migrated, f) for f in ('use_model0', 'use_model_b'))
          and migrated.use_model0 is False and migrated.use_model_b is False)
    o = migrated(positions=r, cell=cells, latent_charges=z, batch=batch)
    ok = ok and set(o.keys()) == base_keys and torch.allclose(o['E_lr'], out['E_lr'])
    _chk("O pickle", ok, "old checkpoint migrates", fails)

    # combinations that cannot work are refused rather than silently ignored
    refusals = [
        ("two charge models", lambda: Les({'is_periodic': True, 'use_qeq': True,
                                           'use_model0': True})),
        ("legacy boundary", lambda: Les({'use_model0': True})),
    ]
    for name, fn in refusals:
        try:
            fn()
            _chk("O refuse", False, name, fails, "accepted")
        except ValueError:
            _chk("O refuse", True, name, fails, "ValueError")

    les = _les(True, False)
    runtime = [
        ("missing head", dict(latent_frontier=None, carrier_count=torch.ones(2, dtype=DT))),
        ("missing carrier", dict(latent_frontier=eps, carrier_count=None)),
        ("dipoles", dict(latent_frontier=eps, carrier_count=torch.ones(2, dtype=DT),
                         latent_dipoles=torch.zeros(len(z), 3, dtype=DT))),
        ("kappas", dict(latent_frontier=eps, carrier_count=torch.ones(2, dtype=DT),
                        latent_kappas=torch.zeros(len(z), dtype=DT))),
        ("bec", dict(latent_frontier=eps, carrier_count=torch.ones(2, dtype=DT),
                     compute_bec=True)),
    ]
    for name, kw in runtime:
        try:
            les(positions=r, cell=cells, latent_charges=z, batch=batch, **kw)
            _chk("O refuse", False, name, fails, "accepted")
        except ValueError:
            _chk("O refuse", True, name, fails, "ValueError")

    # compute_energy=False keeps the charges and drops every energy
    for model_b in (False, True):
        les = _les(True, model_b)
        kw = dict(positions=r, cell=cells, latent_charges=z, batch=batch,
                  carrier_count=torch.ones(2, dtype=DT), compute_energy=False)
        kw['latent_local_addition' if model_b else 'latent_frontier'] = \
            a if model_b else eps
        o = les(**kw)
        energies = [k for k in o if k.startswith('E_')]
        ok = (all(o[k] is None for k in energies)
              and o['occupation'] is not None and o['latent_charges'] is not None)
        _chk("O no-E", ok, f"{'B' if model_b else '0'} compute_energy=False", fails,
             f"{energies} all None, charges kept")

    les_c = _les(True, False, frontier_constrain_charge=True)
    try:
        _run(les_c, r, cells, batch, z, eps, torch.ones(2, dtype=DT), None)
        _chk("O refuse", False, "constraint without Q_ref", fails, "accepted")
    except ValueError:
        _chk("O refuse", True, "constraint without Q_ref", fails, "ValueError")
    return fails


def check_torchscript():
    """P: script, save, reload, and still agree with eager."""
    fails = []
    for periodic in (False, True):
        for model_b in (False, True):
            for constrain in (False, True):
                r, cells, batch, z, eps, a = _system(periodic, n_per=(4, 5))
                q_ref = torch.tensor([0.5, -1.0], dtype=DT)
                les = _les(periodic, model_b, frontier_constrain_charge=constrain)
                m = torch.tensor([0.0, 1.0], dtype=DT)
                kw = dict(positions=r, cell=cells, latent_charges=z, batch=batch,
                          carrier_count=m,
                          reference_total_charge=q_ref if constrain else None)
                kw['latent_local_addition' if model_b else 'latent_frontier'] = \
                    a if model_b else eps
                ref = les(**kw)
                try:
                    sm = torch.jit.script(les)
                    with tempfile.TemporaryDirectory() as d:
                        path = f"{d}/m.pt"
                        torch.jit.save(sm, path)
                        sm = torch.jit.load(path)
                    got = sm(**kw)
                    ok = set(got.keys()) == set(ref.keys())
                    for key in ref:
                        if ref[key] is None:
                            ok = ok and got[key] is None
                        else:
                            ok = ok and got[key] is not None and torch.allclose(
                                got[key], ref[key], atol=1e-10)
                    _chk("P script", ok,
                         f"{'per' if periodic else 'real'} "
                         f"{'B' if model_b else '0'} constr={constrain}", fails)
                except Exception as exc:
                    _chk("P script", False,
                         f"{'per' if periodic else 'real'} "
                         f"{'B' if model_b else '0'} constr={constrain}", fails,
                         f"{type(exc).__name__}: {str(exc).splitlines()[0][:70]}")
    return fails


def check_compile():
    """Q: torch.compile against eager, forces taken outside the compiled region."""
    fails = []
    for periodic in (False, True):
        for model_b in (False, True):
            r, cells, batch, _, _, _ = _system(periodic, n_per=(3, 4))
            les = _les(periodic, model_b, frontier_constrain_charge=True)
            q_ref = torch.tensor([0.5, -1.0], dtype=DT)
            m = torch.ones(2, dtype=DT)

            class Wrap(torch.nn.Module):
                def __init__(self, inner):
                    super().__init__()
                    self.les = inner

                def forward(self, pos):
                    z, head = _geometry_coupled(pos, batch, 2)
                    kw = dict(positions=pos, cell=cells, latent_charges=z,
                              batch=batch, carrier_count=m,
                              reference_total_charge=q_ref)
                    kw['latent_local_addition' if model_b else 'latent_frontier'] = head
                    return self.les(**kw)['E_lr'].sum()

            w = Wrap(les)
            rg = r.clone().requires_grad_(True)
            e0 = w(rg)
            f0 = -torch.autograd.grad(e0, rg)[0]
            try:
                cw = torch.compile(w, fullgraph=True, dynamic=False)
                rg2 = r.clone().requires_grad_(True)
                e1 = cw(rg2)
                f1 = -torch.autograd.grad(e1, rg2)[0]
                ok = (abs(e1.item() - e0.item()) < 1e-10
                      and (f1 - f0).abs().max().item() < 1e-9)
                _chk("Q compile", ok,
                     f"{'per' if periodic else 'real'} {'B' if model_b else '0'}",
                     fails, f"dE={abs(e1.item() - e0.item()):.1e} "
                            f"dF={(f1 - f0).abs().max().item():.1e}")
            except Exception as exc:
                _chk("Q compile", False,
                     f"{'per' if periodic else 'real'} {'B' if model_b else '0'}",
                     fails, f"{type(exc).__name__}: {str(exc).splitlines()[0][:70]}")
    return fails


def main():
    fails = []
    for fn in (check_normalization, check_charge_bookkeeping, check_sign,
               check_stability, check_gauge_model0, check_gauge_model_b,
               check_candidate_energy, check_reference_correction,
               check_no_double_counting, check_forces, check_training_gradients,
               check_remote_electrostatics, check_spectators,
               lambda: check_mini_training(False), lambda: check_mini_training(True),
               check_existing_api,
               check_torchscript, check_compile):
        name = getattr(fn, '__name__', 'check_mini_training')
        doc = (fn.__doc__ or check_mini_training.__doc__).strip().splitlines()[0]
        print(f"\n================ {name} ================")
        print(f"  {doc}")
        fails += fn()

    print("\n==================== SUMMARY ====================")
    if fails:
        print("FAILURES:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("All frontier (Model 0 / Model B) checks passed.")


if __name__ == "__main__":
    main()
