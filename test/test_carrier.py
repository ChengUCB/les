#!/usr/bin/env python
"""Tests of les/module/carrier.py and of the carrier options of Les (Model B): float64, CPU, synthetic structures.

    python test/test_carrier.py        (exit code 0 = every check passed)

  helpers     smooth_cut_weights, carrier_state, weighted_centre, virtual_carrier_edges, check_min_image
  Ewald       VirtualCarrierLR 'fixed' == Ewald_vectorized on the same charges; m = 0 == the host alone;
              'learn' at sigma_c = sigma == 'fixed'; 'learnfix' vs 'learn': same gradients, energy shifted by a constant
              per cell; 'learnfix' with zero host charges: dE/dsigma_c = 0; forces and dE/dt == finite differences
  Les         Model B carrier_width_t at sigma_c = sigma == no width; carrier_self_fixed: dE/dt = 0 at zero charges;
              raw_charge_species (scale 0) gives per-element charges, (zeros, scale 1) == off
  training    FrozenPredictor (frozen, no key leak, p keeps d/dR), DualForces == autograd
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from les import Les                                                                  # noqa: E402
from les.module.carrier import (DualForces, FrozenPredictor, VirtualCarrierLR, carrier_state,  # noqa: E402
                                check_min_image, minimum_image, smooth_cut_weights, virtual_carrier_edges,
                                weighted_centre)
from les.module.ewald_vectorized import Ewald_vectorized                             # noqa: E402

torch.set_default_dtype(torch.float64)
g = torch.Generator().manual_seed(7)
OK = True


def check(name, cond, msg=''):
    global OK
    print('  %-66s %s  %s' % (name, 'PASS' if cond else 'FAIL', msg))
    OK &= bool(cond)


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp(min=1e-300))


# two structures: 40 atoms in an orthorhombic 12.0 x 12.5 x 13.0 A cell, 30 atoms in a skewed cell
CELL = torch.stack([torch.diag(torch.tensor([12.0, 12.5, 13.0])),
                    torch.tensor([[11.0, 0.0, 0.0], [1.5, 11.5, 0.0], [0.5, 1.0, 12.0]])])
NAT = [40, 30]
BATCH = torch.cat([torch.full((n,), b, dtype=torch.long) for b, n in enumerate(NAT)])
N, B = BATCH.numel(), len(NAT)
FRAC = torch.rand(N, 3, generator=g)
POS = torch.einsum('ni,nij->nj', FRAC, CELL[BATCH])
Z = torch.where(torch.rand(N, generator=g) < 0.33, 8, 1)
LOGIT = 3.0 * torch.randn(N, generator=g)
P = torch.cat([torch.softmax(LOGIT[BATCH == b], 0) for b in range(B)])
Q = 0.2 * torch.randn(N, generator=g)
M = torch.tensor([1.0, 1.0])
CUT = 4.0


def edges(pos):
    """all ordered pairs within CUT (minimum image), edge_index = [sender, receiver], shifts: r_s - r_r + shift."""
    snd, rcv, sh = [], [], []
    for b in range(B):
        idx = torch.nonzero(BATCH == b).reshape(-1)
        d = pos[idx].unsqueeze(0) - pos[idx].unsqueeze(1)                 # [i, j] = r_j - r_i
        dm = minimum_image(d.reshape(-1, 3), CELL[b].expand(idx.numel() ** 2, 3, 3)).reshape(d.shape)
        dist = dm.norm(dim=-1)
        for i in range(idx.numel()):
            for j in range(idx.numel()):
                if i != j and dist[i, j] < CUT:
                    snd.append(idx[j]); rcv.append(idx[i])
                    sh.append(dm[i, j] - (pos[idx[j]] - pos[idx[i]]))   # r_rcv - r_snd + shift = -(r_j - r_i)_mic
    snd, rcv = torch.stack(snd), torch.stack(rcv)
    sh = torch.stack(sh)
    return torch.stack([snd, rcv]), -sh


EI, SH = edges(POS)

print('\n== helpers')
w = smooth_cut_weights(P, BATCH, B)
sums = torch.zeros(B).index_add(0, BATCH, w)
check('smooth_cut_weights: sum 1 per structure', torch.allclose(sums, torch.ones(B), atol=1e-12))
check('smooth_cut_weights: 0 where p <= 1e-4', bool((w[P <= 1e-4] == 0).all()))
for mode in ('p', 'field', 'centroid'):
    s, c = carrier_state(mode, P, POS, CELL, BATCH, M, edge_index=EI, shifts=SH)
    check("carrier_state '%s': sum s = m" % mode,
          torch.allclose(torch.zeros(B).index_add(0, BATCH, s), M, atol=1e-12))
    s0, _ = carrier_state(mode, P, POS, CELL, BATCH, torch.zeros(B), edge_index=EI, shifts=SH)
    check("carrier_state '%s': m = 0 gives s = 0" % mode, bool((s0 == 0).all()))
check("carrier_state 'p' == m p", torch.equal(carrier_state('p', P, POS, CELL, BATCH, M)[0], P))
c0 = weighted_centre(P, POS, CELL, BATCH, B)
t = torch.tensor([3.3, -2.1, 5.7])
c1 = weighted_centre(P, POS + t, CELL, BATCH, B)
dc = minimum_image(c1 - c0 - t, CELL)
check('weighted_centre: moves with a rigid translation (mod lattice)', float(dc.abs().max()) < 1e-10,
      '%.1e' % float(dc.abs().max()))
d, recv, wt, _ = virtual_carrier_edges('sites', w, M, POS, CELL, BATCH, CUT, edge_index=EI, shifts=SH)
n_expect = int((w > 0).sum()) + int((w[EI[0]] > 0).sum())
check("virtual_carrier_edges 'sites': one edge per kept site + per neighbour", d.shape[0] == n_expect,
      '%d edges' % d.shape[0])
d, recv, wt, cen = virtual_carrier_edges('centroid', w, M, POS, CELL, BATCH, CUT)
dd = minimum_image(POS - cen[BATCH], CELL[BATCH]).norm(dim=-1)
check("virtual_carrier_edges 'centroid': exactly the atoms within the cutoff",
      recv.numel() == int((dd < CUT).sum()) and torch.allclose(d.norm(dim=-1), dd[recv]))
try:
    check_min_image(CELL, 6.0)
    check('check_min_image raises for cutoff above half a cell width', False)
except ValueError:
    check('check_min_image raises for cutoff above half a cell width', True)

print('\n== VirtualCarrierLR')
KW = dict(sigma=1.0, dl=2.0, N_max=10, carrier_sign=-1.0, epsilon_e=1.78)
k = -1.0 / math.sqrt(1.78)
ew = Ewald_vectorized(sigma=1.0, dl=2.0, is_periodic=True, N_max=10, remove_self_interaction=True)
E_host = ew(q=Q, r=POS, cell=CELL, batch=BATCH)[0].reshape(B, -1).sum(1)
lr = VirtualCarrierLR('field', **KW)
e, _ = lr(Q, POS, CELL, BATCH, M, w=w)
e_ref = ew(q=Q + k * M[BATCH] * w, r=POS, cell=CELL, batch=BATCH)[0].reshape(B, -1).sum(1)
check("'field' fixed == Ewald_vectorized(q + k m w)", rel(e, e_ref) < 1e-12, '%.1e' % rel(e, e_ref))
check("'field' fixed, m = 0 == host alone", rel(lr(Q, POS, CELL, BATCH, torch.zeros(B), w=w)[0], E_host) < 1e-12)
lrc = VirtualCarrierLR('centroid', **KW)
e, _ = lrc(Q, POS, CELL, BATCH, M, centre=cen)
b_all = torch.cat([BATCH, torch.arange(B)])
e_ref = ew(q=torch.cat([Q, k * M]), r=torch.cat([POS, cen]), cell=CELL, batch=b_all)[0].reshape(B, -1).sum(1)
check("'centroid' fixed == Ewald_vectorized with the carrier appended", rel(e, e_ref) < 1e-12, '%.1e' % rel(e, e_ref))
check("'centroid' fixed, m = 0 == host alone",
      rel(lrc(Q, POS, CELL, BATCH, torch.zeros(B), centre=cen)[0], E_host) < 1e-12)


def run(mod, q=Q, pos=POS, centre=None, ww=None):
    pos = pos.clone().requires_grad_(True)
    args = {}
    if mod.mode == 'centroid':
        c = (cen if centre is None else centre).clone().requires_grad_(True)
        args['centre'] = c
    if mod.mode == 'field' or mod.width == 'pspread':
        args['w'] = w if ww is None else ww
    e, s = mod(q, pos, CELL, BATCH, M, **args)
    params = [x for x in mod.parameters()]
    grads = torch.autograd.grad(e.sum(), [pos] + ([args['centre']] if 'centre' in args else []) + params,
                                allow_unused=True)
    return e.detach(), s.detach(), grads


T15 = math.log(math.expm1(0.5))
for mode in ('field', 'centroid'):
    fixed = VirtualCarrierLR(mode, **KW)
    e0, _, g0 = run(fixed)
    for width in (('learn', 'learnfix') if mode == 'centroid' else ('learn',)):
        mod = VirtualCarrierLR(mode, width=width, **KW)
        with torch.no_grad():
            mod.width_t.fill_(-60.0)
        e1, s1, g1 = run(mod)
        r = max(rel(e1, e0), rel(g1[0], g0[0]))
        check("'%s' %s at sigma_c = sigma == fixed (E, dE/dR)" % (mode, width), r < 1e-10, '%.1e' % r)
lrn, lfx = VirtualCarrierLR('centroid', width='learn', **KW), VirtualCarrierLR('centroid', width='learnfix', **KW)
for mod in (lrn, lfx):
    with torch.no_grad():
        mod.width_t.fill_(T15)
eL, sL, gL = run(lrn)
eF, sF, gF = run(lfx)
check('sigma_c = 1.5 output', torch.allclose(sF, torch.full((B,), 1.5)))
check("'learnfix' vs 'learn': same dE/dR and dE/dc", rel(gF[0], gL[0]) < 1e-10 and rel(gF[1], gL[1]) < 1e-10,
      '%.1e %.1e' % (rel(gF[0], gL[0]), rel(gF[1], gL[1])))
_, _, gL0 = run(lrn, q=torch.zeros(N))
_, _, gF0 = run(lfx, q=torch.zeros(N))
check("zero host charges: 'learn' dE/dt != 0, 'learnfix' dE/dt == 0",
      abs(float(gL0[2])) > 1e-6 and abs(float(gF0[2])) <= 1e-10 * abs(float(gL0[2])),
      'learn %.3e learnfix %.1e' % (float(gL0[2]), float(gF0[2])))
for mod, name in ((VirtualCarrierLR('field', width='learn', **KW), "'field' learn"), (lfx, "'centroid' learnfix")):
    with torch.no_grad():
        mod.width_t.fill_(T15)
    _, _, gg = run(mod)
    worst = 0.0
    for i, ax in ((0, 0), (17, 1), (45, 2), (N - 1, 0)):
        e_pm = []
        for sgn in (1, -1):
            pos = POS.clone()
            pos[i, ax] += sgn * 1e-5
            e_pm.append(float(run(mod, pos=pos)[0].sum()))
        fd = (e_pm[0] - e_pm[1]) / 2e-5
        worst = max(worst, abs(float(gg[0][i, ax]) - fd) / max(abs(fd), 1e-3))
    e_pm = []
    for sgn in (1, -1):
        with torch.no_grad():
            mod.width_t.fill_(T15 + sgn * 1e-6)
        e_pm.append(float(run(mod)[0].sum()))
    with torch.no_grad():
        mod.width_t.fill_(T15)
    fd_t = (e_pm[0] - e_pm[1]) / 2e-6
    gt = float(gg[-1])
    check('%s: dE/dR and dE/dt == finite differences' % name,
          worst < 1e-6 and abs(gt - fd_t) <= 1e-6 * max(abs(fd_t), 1e-3), 'dE/dR %.1e, dE/dt %.6e vs %.6e'
          % (worst, gt, fd_t))

print('\n== Les Model B options')
ARGS = {'is_periodic': True, 'sigma': 1.0, 'dl': 2.0, 'N_max': 10, 'remove_self_interaction': True,
        'frontier_beta': 2.0, 'frontier_epsilon_e': 1.78, 'frontier_carrier_sign': -1.0, 'use_model_b': True}
A = 0.5 * torch.randn(N, generator=g)


def les_run(lm, z=Q):
    pos = POS.clone().requires_grad_(True)
    out = lm(positions=pos, cell=CELL, batch=BATCH, latent_charges=z, latent_local_addition=A, carrier_count=M,
             atomic_numbers=Z, compute_energy=True)
    E = out['E_lr']
    params = [x for x in lm.parameters()]
    grads = torch.autograd.grad(E.sum(), [pos] + params, allow_unused=True)
    return E.detach(), out['occupation'].detach(), grads, out


base = Les(dict(ARGS)).double()
E0, p0, g0, _ = les_run(base)
lm = Les(dict(ARGS)).double()
lm.carrier_width_t = torch.nn.Parameter(torch.tensor(-60.0))
E1, p1, g1, out1 = les_run(lm)
r = max(rel(E1, E0), rel(p1, p0), rel(g1[0], g0[0]))
check('carrier_width_t at sigma_c = sigma == no width (E, p, dE/dR)', r < 1e-10, '%.1e' % r)
check("output 'carrier_sigma'", 'carrier_sigma' in out1 and out1['carrier_sigma'].shape == (B,))
for fixself in (False, True):
    lm = Les(dict(ARGS)).double()
    lm.carrier_width_t = torch.nn.Parameter(torch.tensor(T15))
    lm.carrier_self_fixed = fixself
    _, _, gz, _ = les_run(lm, z=torch.zeros(N))
    gt = float(gz[-1])
    if fixself:
        check('carrier_self_fixed, zero charges: dE/dt == 0', abs(gt) < 1e-12, '%.1e' % gt)
    else:
        check('carrier_width_t, zero charges: dE/dt != 0 (carrier self-image term)', abs(gt) > 1e-6, '%.3e' % gt)
lm = Les(dict(ARGS)).double()
lm.raw_charge_species = torch.nn.Parameter(torch.zeros(119))
lm.raw_charge_local_scale = 1.0
E2, p2, _, _ = les_run(lm)
check('raw_charge_species zeros, scale 1 == off', rel(E2, E0) < 1e-12 and rel(p2, p0) < 1e-12)
with torch.no_grad():
    lm.raw_charge_species[1] = 0.3
    lm.raw_charge_species[8] = -0.6
lm.raw_charge_local_scale = 0.0
_, _, _, out3 = les_run(lm)
zr = out3['reference_charges'].detach()
check('raw_charge_species, scale 0: charges = per-element values',
      torch.allclose(zr, torch.where(Z == 8, -0.6, 0.3)), '')

print('\n== training modes')


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 1)
        self.model_outputs = ['toy_p']

    def forward(self, data, training=False):
        data['toy_score'] = self.lin(data['positions']).reshape(-1)
        data['toy_p'] = torch.softmax(data['toy_score'], 0)
        return data


toy = Toy()
fp = FrozenPredictor([toy], source_key='toy_p')
pos = POS.clone().requires_grad_(True)
data = {'positions': pos}
p_out = fp(data)
check('FrozenPredictor: parameters frozen, marked', all(not x.requires_grad for x in fp.parameters())
      and all(getattr(mm, 'frozen_predictor', False) for mm in fp.modules()))
check("FrozenPredictor: caller's dict untouched, model_outputs emptied", set(data) == {'positions'}
      and toy.model_outputs == [])
gp = torch.autograd.grad(p_out[0], pos)[0]
check('FrozenPredictor: p keeps its derivative with respect to the positions', float(gp.abs().sum()) > 0)
pos = POS.clone().requires_grad_(True)
data = {'positions': pos, 'E1': (pos ** 2).sum().reshape(1), 'E2': (pos ** 3).sum().reshape(1)}
data = DualForces(('E1', 'E2'), ('F1', 'F2'))(data)
check('DualForces == -dE/dR for both energies', torch.allclose(data['F1'], -2 * POS)
      and torch.allclose(data['F2'], -3 * POS ** 2))

print('\nALL PASS' if OK else '\nSOME CHECK FAILED')
sys.exit(0 if OK else 1)
