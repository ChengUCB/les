"""
Model 0-a: the two-head frontier allocation.

  A  per-structure normalization of both heads, uneven batch, no mixing
  B  charge conservation at positive, negative and zero deficit
  C  carrier modes: electron and hole, m = 0 and 1
  D  epsilon_e = 1 and 4 keep the physical bookkeeping
  E  zero deficit leaves the reference charges untouched atom by atom
  F  gate direction: negative deficit uses addition, positive uses removal
  G  equal scores in both columns reproduce Model 0 with its constraint on
  H  a per-structure offset on either head independently changes nothing
  I  gradients reach both heads
  J  forces against central differences, with geometry-dependent raw charges AND
     scores, and a deficit that crosses zero inside the difference window
  K  the same at a narrower gate, where the curvature is larger
  L  periodic and non-periodic
  M  the worked example of the specification, and input validation

Run:
    PYTHONPATH=src python test/test_model0_a.py
"""
import io
import math
import sys

import torch

from les import Les
from les.module import Model0A

DT = torch.float64


def _chk(gate, ok, label, fails, note=""):
    print(f"[{gate:10s}] {'ok  ' if ok else 'FAIL'} {label:36s} {note}")
    if not ok:
        fails.append(f"{label}:{gate.strip()}")


def _system(periodic, n_per=(5, 6, 7), seed=42):
    g = torch.Generator().manual_seed(seed)
    n = sum(n_per)
    batch = torch.cat([torch.full((k,), i) for i, k in enumerate(n_per)]).long()
    r = torch.rand(n, 3, generator=g, dtype=DT) * 7.0
    if periodic:
        cells = torch.stack([
            torch.eye(3, dtype=DT) * 12.0,
            torch.tensor([[11., .4, 0.], [.3, 12., .2], [0., .1, 13.]], dtype=DT),
            torch.diag(torch.tensor([11., 12., 14.], dtype=DT)),
        ])[:len(n_per)]
    else:
        cells = torch.zeros(len(n_per), 3, 3, dtype=DT)
    z = torch.randn(n, generator=g, dtype=DT) * 0.3
    frontier = torch.randn(n, 2, generator=g, dtype=DT)
    return r, cells, batch, z, frontier


def _onehot(batch, n_batch):
    from les.module import segment_onehot
    return segment_onehot(batch, n_batch, DT)


def _les(periodic, **cfg):
    args = {'is_periodic': periodic, 'use_model0_a': True}
    args.update(cfg)
    return Les(args).to(DT)


def _run(les, r, cells, batch, z, frontier, m, q_ref):
    return les(positions=r, cell=cells, batch=batch, latent_charges=z,
               latent_frontier=frontier, reference_total_charge=q_ref,
               carrier_count=m)


# ----------------------------------------------------------------------------
def check_normalization():
    """A: both heads sum to one per structure, with no leakage between them."""
    fails = []
    for n_per in ((5, 6, 7), (3, 9, 4, 8), (11,), (2, 2, 2)):
        r, cells, batch, z, frontier = _system(False, n_per=n_per)
        B = len(n_per)
        les = _les(False)
        out = _run(les, r, cells, batch, z, frontier, torch.ones(B, dtype=DT),
                   torch.zeros(B, dtype=DT))
        oh = _onehot(batch, B)
        worst = max((oh @ out[k] - 1.0).abs().max().item()
                    for k in ('occupation_add', 'occupation_remove'))
        _chk("A norm", worst < 1e-12, f"n_per={n_per}", fails,
             f"max|sum(p)-1|={worst:.1e}")

    # an unsorted batch gives the same answer atom for atom
    r, cells, batch, z, frontier = _system(False, n_per=(4, 5, 6))
    les = _les(False)
    m = torch.tensor([0.0, 1.0, 1.0], dtype=DT)
    q = torch.tensor([0.5, -1.0, 2.0], dtype=DT)
    ref = _run(les, r, cells, batch, z, frontier, m, q)
    perm = torch.randperm(len(batch), generator=torch.Generator().manual_seed(1))
    got = _run(les, r[perm], cells, batch[perm], z[perm], frontier[perm], m, q)
    ok = all(torch.allclose(got[k], ref[k][perm], atol=1e-12)
             for k in ('occupation_add', 'occupation_remove', 'latent_charges',
                       'reference_charges'))
    ok = ok and torch.allclose(got['E_lr'], ref['E_lr'], atol=1e-10)
    _chk("A unsort", ok, "shuffled atoms", fails)

    # moving one structure's scores must not touch another's charges
    bumped = frontier.clone()
    bumped[batch == 0, 0] += 3.0
    got = _run(les, r, cells, batch, z, bumped, m, q)
    far = batch == 1
    _chk("A isolate", torch.allclose(got['occupation_add'][far],
                                     ref['occupation_add'][far], atol=1e-12),
         "no cross-structure leak", fails)
    return fails


def check_conservation():
    """B: the reference total is Q_ref, and the final one is Q_ref mp m."""
    fails = []
    for periodic in (False, True):
        r, cells, batch, z, frontier = _system(periodic, n_per=(5, 6))
        B = 2
        oh = _onehot(batch, B)
        raw_sum = oh @ z
        for tag, q_ref in (("D<0", raw_sum - 0.4), ("D>0", raw_sum + 0.4),
                           ("D=0", raw_sum.clone()),
                           ("mixed", torch.stack([raw_sum[0] - 0.7,
                                                  raw_sum[1] + 0.9]))):
            for sign in (-1, +1):
                les = _les(periodic, frontier_carrier_sign=sign)
                worst_ref = worst_fin = 0.0
                for mv in (0.0, 1.0):
                    m = torch.full((B,), mv, dtype=DT)
                    out = _run(les, r, cells, batch, z, frontier, m, q_ref)
                    worst_ref = max(worst_ref,
                                    (oh @ out['reference_charges'] - q_ref
                                     ).abs().max().item())
                    worst_fin = max(worst_fin,
                                    (oh @ out['latent_charges']
                                     - (q_ref + sign * mv)).abs().max().item())
                _chk("B conserve", max(worst_ref, worst_fin) < 1e-11,
                     f"{'per' if periodic else 'real'} {tag} sign={sign:+d}", fails,
                     f"|ref-Q|={worst_ref:.1e} |fin-(Q+cm)|={worst_fin:.1e}")
    return fails


def check_scaling():
    """C/D: epsilon_e rescales the LES charges but not the physical bookkeeping."""
    fails = []
    for eps_e in (1.0, 4.0):
        for sign in (-1, +1):
            r, cells, batch, z, frontier = _system(False, n_per=(5, 6))
            B, s = 2, math.sqrt(eps_e)
            oh = _onehot(batch, B)
            q_ref = torch.tensor([1.3, -2.0], dtype=DT)
            les = _les(False, frontier_epsilon_e=eps_e, frontier_carrier_sign=sign)
            worst = 0.0
            for mv in (0.0, 1.0):
                m = torch.full((B,), mv, dtype=DT)
                out = _run(les, r, cells, batch, z, frontier, m, q_ref)
                phys = s * (oh @ out['latent_charges'])
                worst = max(worst, (phys - (q_ref + sign * mv)).abs().max().item())
            _chk("C/D scale", worst < 1e-11, f"eps_e={eps_e} sign={sign:+d}", fails,
                 f"max|s*sum z - (Q+cm)|={worst:.1e}")
    return fails


def check_zero_deficit():
    """E: at D = 0 the reference correction is identically zero, atom by atom.

    Not merely zero in the sum: the correction is D/s times a normalized mixture,
    so a vanishing deficit must leave every atom alone. The carrier at m = 1 is a
    separate step and does move the charges.
    """
    fails = []
    for periodic in (False, True):
        r, cells, batch, z, frontier = _system(periodic, n_per=(5, 6))
        B = 2
        oh = _onehot(batch, B)
        q_ref = oh @ z                  # exactly zero deficit in this dtype
        les = _les(periodic)
        out = _run(les, r, cells, batch, z, frontier, torch.zeros(B, dtype=DT), q_ref)
        d_atom = (out['reference_charges'] - z).abs().max().item()
        d_def = out['reference_charge_deficit'].abs().max().item()
        gate = out['removal_gate']
        _chk("E zero", d_atom == 0.0 and d_def == 0.0,
             f"{'per' if periodic else 'real'} correction is exactly 0", fails,
             f"max|z_ref-z_raw|={d_atom:.1e}, D={d_def:.1e}, g={gate[0].item():.3f}")
        # and the carrier still moves them at m = 1
        out1 = _run(les, r, cells, batch, z, frontier, torch.ones(B, dtype=DT), q_ref)
        moved = (out1['latent_charges'] - z).abs().max().item()
        _chk("E carrier", moved > 1e-6,
             f"{'per' if periodic else 'real'} m=1 still moves charge", fails,
             f"max|dq|={moved:.4f}")
    return fails


def check_gate_direction():
    """F: a negative deficit draws on the addition head, a positive one on removal."""
    fails = []
    r, cells, batch, z, frontier = _system(False, n_per=(6,))
    oh = _onehot(batch, 1)
    raw = oh @ z
    les = _les(False)
    for tag, q_ref, expect in (("D<<0", raw - 0.5, 'add'), ("D>>0", raw + 0.5, 'rem')):
        out = _run(les, r, cells, batch, z, frontier, torch.zeros(1, dtype=DT), q_ref)
        # recover the mixture actually used from the correction
        corr = out['reference_charges'] - z
        share = corr / (out['reference_charge_deficit'][0] / les.model0_a.scale)
        d_add = (share - out['occupation_add']).abs().max().item()
        d_rem = (share - out['occupation_remove']).abs().max().item()
        gate = out['removal_gate'][0].item()
        picked = 'add' if d_add < d_rem else 'rem'
        ok = picked == expect and min(d_add, d_rem) < 1e-9
        _chk("F gate", ok, f"{tag} uses the {expect} head", fails,
             f"g={gate:.6f} |share-p_{expect}|={min(d_add, d_rem):.1e}")

    # the documented gate values
    tab = [(-0.03, 0.00247), (-0.02, 0.01799), (-0.01, 0.11920),
           (0.0, 0.5), (0.01, 0.88080), (0.02, 0.98201), (0.03, 0.99753)]
    worst = 0.0
    for d, want in tab:
        got = 0.5 * (1.0 + math.tanh(d / 0.01))
        worst = max(worst, abs(got - want))
    _chk("F table", worst < 1e-5, "gate matches the documented values", fails,
         f"max|dg|={worst:.1e}")
    return fails


def check_matches_model0():
    """G: equal columns reproduce Model 0 with its constraint on, both carrier modes."""
    fails = []
    for periodic in (False, True):
        for sign in (-1, +1):
            r, cells, batch, z, frontier = _system(periodic, n_per=(5, 6))
            B = 2
            eps = frontier[:, 0].contiguous()
            two = torch.stack([eps, eps], dim=1)
            q_ref = torch.tensor([1.0, -0.5], dtype=DT)
            m = torch.tensor([0.0, 1.0], dtype=DT)
            ref = Les({'is_periodic': periodic, 'use_model0': True,
                       'frontier_constrain_charge': True,
                       'frontier_carrier_sign': sign}).to(DT)
            new = _les(periodic, frontier_carrier_sign=sign)
            rg1 = r.clone().requires_grad_(True)
            o1 = ref(positions=rg1, cell=cells, batch=batch, latent_charges=z,
                     latent_frontier=eps, reference_total_charge=q_ref, carrier_count=m)
            f1 = -torch.autograd.grad(o1['E_lr'].sum(), rg1)[0]
            rg2 = r.clone().requires_grad_(True)
            o2 = _run(new, rg2, cells, batch, z, two, m, q_ref)
            f2 = -torch.autograd.grad(o2['E_lr'].sum(), rg2)[0]
            worst = max(
                (o1['latent_charges'] - o2['latent_charges']).abs().max().item(),
                (o1['reference_charges'] - o2['reference_charges']).abs().max().item(),
                (o1['E_lr'] - o2['E_lr']).abs().max().item(),
                (f1 - f2).abs().max().item())
            _chk("G vs M0", worst < 1e-11,
                 f"{'per' if periodic else 'real'} sign={sign:+d}", fails,
                 f"max|d(q,q_ref,E,F)|={worst:.1e}")
    return fails


def check_offset_invariance():
    """H: each head carries its own gauge freedom, independently."""
    fails = []
    for periodic in (False, True):
        r, cells, batch, z, frontier = _system(periodic, n_per=(5, 6))
        B = 2
        oh_t = _onehot(batch, B).transpose(0, 1)
        q_ref = torch.tensor([0.8, -1.2], dtype=DT)
        m = torch.ones(B, dtype=DT)
        les = _les(periodic)
        rg = r.clone().requires_grad_(True)
        base = _run(les, rg, cells, batch, z, frontier, m, q_ref)
        f0 = -torch.autograd.grad(base['E_lr'].sum(), rg)[0]
        for name, sa, sr in (("add only", torch.tensor([3.0, -2.0], dtype=DT),
                              torch.zeros(B, dtype=DT)),
                             ("remove only", torch.zeros(B, dtype=DT),
                              torch.tensor([-1.5, 4.0], dtype=DT)),
                             ("both", torch.tensor([2.0, 5.0], dtype=DT),
                              torch.tensor([-3.0, 1.0], dtype=DT))):
            shifted = frontier + torch.stack([oh_t @ sa, oh_t @ sr], dim=1)
            rg2 = r.clone().requires_grad_(True)
            got = _run(les, rg2, cells, batch, z, shifted, m, q_ref)
            f1 = -torch.autograd.grad(got['E_lr'].sum(), rg2)[0]
            worst = max(
                (base['latent_charges'] - got['latent_charges']).abs().max().item(),
                (base['E_lr'] - got['E_lr']).abs().max().item(),
                (f0 - f1).abs().max().item())
            _chk("H gauge", worst < 1e-11,
                 f"{'per' if periodic else 'real'} {name}", fails,
                 f"max|d(q,E,F)|={worst:.1e}")
    return fails


def check_gradients():
    """I: gradient reaches both heads exactly when both are engaged.

    Which heads carry signal follows from the design, and asserting more than that
    would be asserting something false. The correction draws on the removal head only
    for a positive deficit, and an electron carrier draws only on the addition head.
    So a negative deficit with an electron leaves the removal head unused, and its
    gradient is not small but exactly zero -- the training-signal limitation the
    specification calls out, pinned here rather than hidden.
    """
    fails = []
    r, cells, batch, z, frontier = _system(False, n_per=(5, 6))
    B = 2
    oh = _onehot(batch, B)
    raw = oh @ z

    def grads(sign, q_ref, mv):
        les = _les(False, frontier_carrier_sign=sign)
        fg = frontier.clone().requires_grad_(True)
        zg = z.clone().requires_grad_(True)
        rg = r.clone().requires_grad_(True)
        out = _run(les, rg, cells, batch, zg, fg, torch.full((B,), mv, dtype=DT), q_ref)
        f = -torch.autograd.grad(out['E_lr'].sum(), rg, create_graph=True)[0]
        (f ** 2).sum().backward()
        return (fg.grad[:, 0].abs().max().item(), fg.grad[:, 1].abs().max().item(),
                zg.grad.abs().max().item(), out['removal_gate'])

    # both heads engaged: the correction uses one, the carrier the other
    for tag, sign, q_ref, mv in (
            ("electron, D>0", -1, raw + 0.3, 1.0),
            ("hole, D<0", +1, raw - 0.3, 1.0),
            ("inside the gate", -1, raw + 0.002, 1.0)):
        ga, gr, gz, gate = grads(sign, q_ref, mv)
        ok = ga > 0 and gr > 0 and gz > 0
        _chk("I both", ok, tag, fails,
             f"|g_add|={ga:.2e} |g_rem|={gr:.2e} g={gate[0].item():.4f}")

    # and the configurations where one head is genuinely unused
    for tag, sign, q_ref, mv, unused in (
            ("electron, D<0: removal unused", -1, raw - 0.3, 1.0, 'rem'),
            ("electron, D>0, m=0: addition unused", -1, raw + 0.3, 0.0, 'add'),
            ("hole, D>0, m=0: addition unused", +1, raw + 0.3, 0.0, 'add')):
        ga, gr, gz, gate = grads(sign, q_ref, mv)
        dead, live = (gr, ga) if unused == 'rem' else (ga, gr)
        _chk("I one-sided", dead == 0.0 and live > 0, tag, fails,
             f"unused={dead:.1e} used={live:.2e}")
    return fails


def _coupled(pos):
    """Raw charges and both scores as real functions of the geometry.

    Differentiating a case whose charges and scores were constants would leave the
    deficit and the gate out of the derivative entirely, which is the path this
    model adds.
    """
    d = torch.linalg.norm(pos, dim=-1)
    z = 0.23 * torch.sin(1.1 * d) + 0.07 * pos[:, 0]
    add = 0.8 * torch.cos(0.9 * d) + 0.13 * pos[:, 1]
    rem = -0.6 * torch.sin(0.7 * d) + 0.09 * pos[:, 2]
    return z, torch.stack([add, rem], dim=1)


def _force_check(gate, les, r, cells, batch, m, q_ref, label, fails, h=1e-6):
    def energy(pos):
        z, frontier = _coupled(pos)
        return _run(les, pos, cells, batch, z, frontier, m, q_ref)['E_lr'].sum()

    rg = r.clone().requires_grad_(True)
    f_auto = -torch.autograd.grad(energy(rg), rg)[0]
    f_fd = torch.zeros_like(r)
    for i in range(r.shape[0]):
        for c in range(3):
            rp = r.clone(); rp[i, c] += h
            rm = r.clone(); rm[i, c] -= h
            f_fd[i, c] = -(energy(rp) - energy(rm)) / (2 * h)
    err = (f_auto - f_fd).abs().max().item()
    scale = max(f_fd.abs().max().item(), 1e-12)
    _chk(gate, err / scale < 1e-5, label, fails, f"rel={err / scale:.2e}")
    return err / scale


def check_forces():
    """J: autograd forces against central differences, with the deficit crossing zero.

    Q_ref is set to the physical raw total at the reference geometry, so D is exactly
    zero there and the difference window straddles the gate's steepest point.
    """
    fails = []
    for periodic in (False, True):
        r, cells, batch, _, _ = _system(periodic, n_per=(3, 4))
        B = 2
        oh = _onehot(batch, B)
        les = _les(periodic)
        z0, _ = _coupled(r)
        q_ref = les.model0_a.scale * (oh @ z0)      # D = 0 at this geometry
        out = _run(les, r, cells, batch, *_coupled(r), torch.ones(B, dtype=DT), q_ref)
        d0 = out['reference_charge_deficit'].abs().max().item()
        _chk("J at D=0", d0 < 1e-12, f"{'per' if periodic else 'real'} deficit is zero",
             fails, f"|D|={d0:.1e}")
        for sign in (-1, +1):
            l = _les(periodic, frontier_carrier_sign=sign)
            for mv in (0.0, 1.0):
                _force_check("J force", l, r, cells, batch,
                             torch.full((B,), mv, dtype=DT), q_ref,
                             f"{'per' if periodic else 'real'} sign={sign:+d} m={mv:.0f}",
                             fails)

        # and away from the crossing, where the gate is saturated
        for off in (-0.5, +0.5):
            _force_check("J off-gate", les, r, cells, batch, torch.ones(B, dtype=DT),
                         q_ref + off,
                         f"{'per' if periodic else 'real'} D={off:+.1f}", fails)
    return fails


def check_gate_width():
    """K: the same check at a narrower gate, where the curvature is larger."""
    fails = []
    r, cells, batch, _, _ = _system(False, n_per=(3, 4))
    B = 2
    oh = _onehot(batch, B)
    z0, _ = _coupled(r)
    rels = {}
    for width in (0.01, 0.005):
        les = _les(False, frontier_charge_switch_width=width)
        q_ref = les.model0_a.scale * (oh @ z0)
        rels[width] = _force_check("K width", les, r, cells, batch,
                                   torch.ones(B, dtype=DT), q_ref,
                                   f"delta={width} at D=0", fails, h=1e-7)
    print(f"             narrower gate changes the finite-difference agreement from "
          f"{rels[0.01]:.2e} to {rels[0.005]:.2e}")
    return fails


def check_api_and_compat():
    """L/M: the worked example, validation, and the existing behaviour untouched."""
    fails = []

    # the specification's example
    p_add = torch.tensor([0.95, 0.05], dtype=DT)
    p_rem = torch.tensor([0.20, 0.80], dtype=DT)
    frontier = torch.stack([-torch.log(p_add), -torch.log(p_rem)], dim=1)
    z_raw = torch.tensor([1.60, 1.10], dtype=DT)
    mod = Model0A().to(DT)
    zf, zr, ps, pa, pr, D, g = mod(z_raw, frontier, torch.zeros(2, dtype=torch.long),
                                   torch.ones(1, dtype=DT), 1,
                                   torch.tensor([3.0], dtype=DT))
    ok = (abs(D.item() - 0.3) < 1e-12
          and (zr - torch.tensor([1.66, 1.34], dtype=DT)).abs().max() < 1e-9
          and (zf - torch.tensor([0.71, 1.29], dtype=DT)).abs().max() < 1e-9)
    _chk("M example", ok, "specification worked example", fails,
         f"q_ref={zr.tolist()} q_final={[round(v, 4) for v in zf.tolist()]}")

    base_keys = {'E_lr', 'latent_charges', 'latent_dipoles', 'latent_quads',
                 'latent_alphas', 'BEC'}
    extra = {'occupation', 'reference_charges', 'occupation_add', 'occupation_remove',
             'reference_charge_deficit', 'removal_gate'}
    r, cells, batch, z, fr = _system(True, n_per=(4, 5))
    les = _les(True)
    out = _run(les, r, cells, batch, z, fr, torch.ones(2, dtype=DT),
               torch.tensor([1.0, 0.0], dtype=DT))
    _chk("M keys", set(out.keys()) == base_keys | extra, "output keys", fails,
         f"+{sorted(extra)}")
    _chk("M params", list(les.state_dict().keys()) == [], "adds no parameters", fails)

    plain = Les({'is_periodic': True}).to(DT)
    po = plain(positions=r, cell=cells, latent_charges=z, batch=batch)
    _chk("M plain", set(po.keys()) == base_keys, "existing key set unchanged", fails)

    # an old pickle predates the flag
    buf = io.BytesIO()
    torch.save(plain, buf); buf.seek(0)
    obj = torch.load(buf, weights_only=False)
    obj.__dict__.pop('use_model0_a', None)
    buf2 = io.BytesIO()
    torch.save(obj, buf2); buf2.seek(0)
    migrated = torch.load(buf2, weights_only=False)
    mo = migrated(positions=r, cell=cells, latent_charges=z, batch=batch)
    _chk("M pickle", (hasattr(migrated, 'use_model0_a')
                      and migrated.use_model0_a is False
                      and set(mo.keys()) == base_keys
                      and torch.allclose(mo['E_lr'], po['E_lr'])),
         "old checkpoint restores", fails)

    n = len(z)
    refusals = [
        ("two charge models", lambda: Les({'is_periodic': True, 'use_model0': True,
                                           'use_model0_a': True})),
        ("legacy boundary", lambda: Les({'use_model0_a': True})),
        ("beta <= 0", lambda: Les({'is_periodic': True, 'use_model0_a': True,
                                   'frontier_beta': 0.0})),
        ("epsilon_e <= 0", lambda: Les({'is_periodic': True, 'use_model0_a': True,
                                        'frontier_epsilon_e': -1.0})),
        ("width <= 0", lambda: Les({'is_periodic': True, 'use_model0_a': True,
                                    'frontier_charge_switch_width': 0.0})),
        ("carrier sign 0", lambda: Les({'is_periodic': True, 'use_model0_a': True,
                                        'frontier_carrier_sign': 0})),
    ]
    for name, fn in refusals:
        try:
            fn()
            _chk("M refuse", False, name, fails, "accepted")
        except ValueError:
            _chk("M refuse", True, name, fails, "ValueError")

    runtime = [
        ("1-D frontier", dict(latent_frontier=fr[:, 0].contiguous(),
                              carrier_count=torch.ones(2, dtype=DT),
                              reference_total_charge=torch.zeros(2, dtype=DT))),
        ("3 columns", dict(latent_frontier=torch.zeros(n, 3, dtype=DT),
                           carrier_count=torch.ones(2, dtype=DT),
                           reference_total_charge=torch.zeros(2, dtype=DT))),
        ("no frontier", dict(carrier_count=torch.ones(2, dtype=DT),
                             reference_total_charge=torch.zeros(2, dtype=DT))),
        ("no reference charge", dict(latent_frontier=fr,
                                     carrier_count=torch.ones(2, dtype=DT))),
        ("no carrier", dict(latent_frontier=fr,
                            reference_total_charge=torch.zeros(2, dtype=DT))),
        ("dipoles", dict(latent_frontier=fr, carrier_count=torch.ones(2, dtype=DT),
                         reference_total_charge=torch.zeros(2, dtype=DT),
                         latent_dipoles=torch.zeros(n, 3, dtype=DT))),
        ("bec", dict(latent_frontier=fr, carrier_count=torch.ones(2, dtype=DT),
                     reference_total_charge=torch.zeros(2, dtype=DT),
                     compute_bec=True)),
    ]
    for name, kw in runtime:
        try:
            les(positions=r, cell=cells, batch=batch, latent_charges=z, **kw)
            _chk("M refuse", False, name, fails, "accepted")
        except ValueError:
            _chk("M refuse", True, name, fails, "ValueError")
    return fails


def main():
    fails = []
    for fn in (check_normalization, check_conservation, check_scaling,
               check_zero_deficit, check_gate_direction, check_matches_model0,
               check_offset_invariance, check_gradients, check_forces,
               check_gate_width, check_api_and_compat):
        print(f"\n================ {fn.__name__} ================")
        print(f"  {fn.__doc__.strip().splitlines()[0]}")
        fails += fn()

    print("\n==================== SUMMARY ====================")
    if fails:
        print("FAILURES:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("All Model 0-a checks passed.")


if __name__ == "__main__":
    main()
