"""Carrier-conditioned models: backbone-independent building blocks.

An excess carrier (electron or hole) whose position is not labelled is located by a predictor P (Model B,
les/module/model_b.py): an occupation p_i >= 0 per atom, sum_i p_i = 1 per structure. A second model F takes that
allocation as an input. This file holds the parts of F that do not depend on the backbone (CACE, MACE, ...); a backbone
supplies the representation and two small hooks (below).

Three F models
    p re-input  s_i = m p_i is added to the atom embedding of F's backbone (s_i times a learned vector that starts at 0).
                E_F = E_SR(h^s) + Ewald(q(h^s)).                                     -> carrier_state(mode='p')
    va_site     a virtual atom on every candidate atom j, weight w_j (p after a smooth cut, smooth_cut_weights); it only
                sends messages, into the first message / A-basis sum of F's backbone; the carrier charge k m w_j sits on
                the atoms in the Ewald energy.                       -> virtual_carrier_edges('sites'), VirtualCarrierLR('field')
    va_centre   one virtual atom at the w-weighted minimum-image centre c (weighted_centre), sending to every atom within
                the cutoff; the carrier charge k m is a Gaussian at c.  -> virtual_carrier_edges('centroid'),
                                                                           VirtualCarrierLR('centroid')
    k = carrier_sign / sqrt(epsilon_e) is the carrier charge in LES units, m the carrier count of the structure.

Virtual-atom message (what a backbone adds for each virtual edge, receiver i, displacement d = r_i - r_v):
    radial(r) x angular(u) x [(m w_v e) (x) receiver_embedding(Z_i)]
    u = d / sqrt(|d|^2 + a^2) (softened_direction, a = 0.4 A): equals d/|d| for r >> a, only l = 0 survives at r = 0;
    radial = the backbone's Bessel basis in its sinc form, prefactor w sinc(w r / pi) = prefactor sin(w r) / r, finite
             at r = 0 (sinc_bessel), times the backbone's cutoff function;
    e = a learned sender vector, initialised to 0 (F starts as the carrier-blind host model).
Energy and forces stay continuous when a virtual atom passes through an atom.

Two training modes
    joint      P and F in one network; loss = L(E_F, F_F) + w_aux L(E_P, F_P); both forces are -dE/dR on the same positions
               (DualForces); F's forces include the path through p(R).
    two-stage  P = a trained network, frozen (FrozenPredictor): it runs on a shallow copy of the inputs, so it shares the
               positions (p(R) keeps its coordinate derivative) and none of its outputs overwrite F's.
Never detach p in a production model: the forces would stop being the gradient of the energy.

Carrier Gaussian width (VirtualCarrierLR width=...): 'fixed' = the LES sigma for every charge (the default); 'learn' =
sigma_c = sigma + softplus(t), one trained number; 'learnfix' (centroid only) = as 'learn', but the carrier with its own
periodic images keeps the LES width, so sigma_c enters the carrier-host term only; 'pspread' (centroid only) =
sigma_c^2 = sigma^2 + softplus(t0) + softplus(ta) <|r - c|^2>_w per structure. The width of P's own carrier (Model B)
is set on the Les module (carrier_width_t, carrier_self_fixed; les/les.py).

Origin: polaron_mlip (11/2026), training/pcond_modules.py, vnode_modules.py, width_modules.py, polaron_modules.py; the
functions here reproduce those to floating-point accuracy. Everything works for several structures per batch, any
elements and m in {0, 1, ...}; m = 0 gives exactly the carrier-free model.
"""
import math
import os
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from les.module.ewald_vectorized import Ewald_vectorized

__all__ = ['segment_sum', 'carrier_per_structure', 'minimum_image', 'weighted_centre', 'smooth_cut_weights',
           'check_min_image', 'carrier_state', 'virtual_carrier_edges', 'softened_direction', 'sinc_bessel',
           'WidthEwald', 'VirtualCarrierLR', 'FrozenPredictor', 'DualForces']

TWOPI = 2.0 * math.pi


# ---------------------------------------------------------------------------------------------------- per-structure helpers
def segment_sum(x: torch.Tensor, batch: torch.Tensor, n_batch: int) -> torch.Tensor:
    """Sum of per-atom values over each structure, [n_batch, ...]."""
    out = torch.zeros((n_batch,) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)
    return out.index_add_(0, batch, x)


def carrier_per_structure(carrier: Optional[torch.Tensor], batch: torch.Tensor, n_batch: int,
                          dtype: torch.dtype) -> torch.Tensor:
    """m per structure [n_batch] from a per-atom column in which every atom of a structure holds the same value;
    1 per structure when there is no such column."""
    if carrier is None:
        return torch.ones(n_batch, dtype=dtype, device=batch.device)
    c = carrier.reshape(-1).to(dtype)
    tot = segment_sum(c, batch, n_batch)
    cnt = segment_sum(torch.ones_like(c), batch, n_batch).clamp(min=1.0)
    return tot / cnt


def minimum_image(d: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Displacements d [n, 3] reduced to the minimum image of the cells [n, 3, 3] (rows = lattice vectors). Exact for
    orthorhombic cells, and for skewed ones while the displacement is shorter than half the smallest width."""
    frac = torch.linalg.solve(cells.transpose(1, 2), d.unsqueeze(-1)).squeeze(-1)    # d = frac @ cell
    frac = frac - torch.round(frac.detach())
    return torch.bmm(frac.unsqueeze(1), cells).squeeze(1)


def weighted_centre(p: torch.Tensor, pos: torch.Tensor, cell: torch.Tensor, batch: torch.Tensor,
                    n_batch: int) -> torch.Tensor:
    """p-weighted centre of each structure [n_batch, 3] under periodic boundaries: displacements from the atom with the
    largest p (the anchor, chosen without gradient) are taken to their minimum image and averaged with weight p.
    Continuous in p and the positions while the weight sits within half a cell of the anchor."""
    centres = []
    for b in range(n_batch):
        idx = torch.nonzero(batch == b, as_tuple=False).reshape(-1)
        pb = p[idx]
        anchor = pos[idx[torch.argmax(pb.detach())]]
        d = minimum_image(pos[idx] - anchor, cell[b].expand(idx.numel(), 3, 3))
        w = pb / pb.sum().clamp(min=1e-12)
        centres.append(anchor + (w.unsqueeze(-1) * d).sum(dim=0))
    return torch.stack(centres)


def smooth_cut_weights(p: torch.Tensor, batch: torch.Tensor, n_batch: int, lo: float = 1e-4,
                       hi: float = 1e-3) -> torch.Tensor:
    """w = p s(p), normalised per structure (sum 1 wherever p has weight); s = 0 below lo, 1 above hi, C1 smoothstep
    between. Values below hi carry < 0.1 % of the carrier each; the cut only saves virtual edges and never jumps."""
    if hi <= lo:
        raise ValueError('smooth cut: hi must be above lo')
    t = ((p - lo) / (hi - lo)).clamp(0.0, 1.0)
    w = p * t * t * (3.0 - 2.0 * t)
    tot = segment_sum(w, batch, n_batch)
    return w / tot.clamp(min=1e-30)[batch]


def check_min_image(cell: torch.Tensor, cutoff: float) -> None:
    """The centroid's neighbours are found by minimum image: needs the cutoff below half of every cell width."""
    vol = torch.det(cell).abs()
    for i, j in ((1, 2), (2, 0), (0, 1)):
        width = vol / torch.linalg.norm(torch.cross(cell[:, i], cell[:, j], dim=-1), dim=-1)
        if bool((width <= 2 * cutoff).any()):
            raise ValueError('centroid virtual atom: a cell width (%.2f A) is not above twice the cutoff (%.2f A)'
                             % (float(width.min()), cutoff))


# ---------------------------------------------------------------------------------------------------- carrier inputs of F
def carrier_state(mode: str, p: torch.Tensor, positions: torch.Tensor, cell: torch.Tensor, batch: torch.Tensor,
                  m: torch.Tensor, sigma: float = 1.0, edge_index: Optional[torch.Tensor] = None,
                  shifts: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Per-atom carrier state s [n_atoms] fed to F's atom embedding (p re-input), and the centre for 'centroid'.
        'p'         s_i = m p_i
        'field'     s_i = m f_i / sum_k f_k,  f_i = p_i + sum_j p_j g(r_ij) over the backbone's edges j -> i
        'centroid'  s_i = m g_i / sum_k g_k,  g_i = g(|r_i - c|), c = weighted_centre(p)
    g(r) = exp(-r^2 / 2 sigma^2); sum_i s_i = m in every mode. The recipe uses 'p'."""
    if mode not in ('p', 'field', 'centroid'):
        raise ValueError('carrier_state mode: p, field or centroid, not %r' % mode)
    p = p.reshape(-1)
    n_batch = cell.shape[0]
    m_atom = m.to(p.dtype)[batch]
    centre = None
    if mode == 'p':
        return m_atom * p, None
    g = lambda r2: torch.exp(-0.5 * r2 / (sigma * sigma))
    if mode == 'field':
        if edge_index is None or shifts is None:
            raise ValueError("carrier_state 'field' needs edge_index and shifts")
        snd, rcv = edge_index[0], edge_index[1]
        vec = positions[rcv] - positions[snd] + shifts
        w = p + segment_sum(p[snd] * g((vec * vec).sum(dim=-1)), rcv, p.shape[0])
    else:
        centre = weighted_centre(p, positions, cell, batch, n_batch)
        d = minimum_image(positions - centre[batch], cell[batch])
        w = g((d * d).sum(dim=-1))
    tot = segment_sum(w, batch, n_batch).clamp(min=1e-12)
    return m_atom * w / tot[batch], centre


def virtual_carrier_edges(mode: str, w: torch.Tensor, m: torch.Tensor, positions: torch.Tensor, cell: torch.Tensor,
                          batch: torch.Tensor, cutoff: float, edge_index: Optional[torch.Tensor] = None,
                          shifts: Optional[torch.Tensor] = None, centroid_gamma: float = 1.0,
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Virtual-atom edges for va_site ('sites') / va_centre ('centroid'): (d [E, 3] = r_receiver - r_virtual,
    receiver [E], weight m w_v [E], centre [n_batch, 3] or None). w = smooth_cut_weights(p).
        'sites'     a virtual atom on every atom j with w_j > 0: an edge to j itself (d = 0) and one to every receiver i
                    of the backbone's edges j -> i (edge_index = [sender, receiver], d = r_i - r_j + shift)
        'centroid'  one virtual atom per structure at c = weighted_centre(w ** centroid_gamma), an edge to every atom
                    within the cutoff (minimum image; needs the cutoff below half of every cell width)"""
    if mode == 'sites':
        if edge_index is None or shifts is None:
            raise ValueError("virtual_carrier_edges 'sites' needs edge_index and shifts")
        keep = w > 0
        on = torch.nonzero(keep, as_tuple=False).reshape(-1)                 # virtual atom on its own atom: d = 0
        nb = keep[edge_index[0]]                                               # ... and on every neighbour
        snd, rcv = edge_index[0][nb], edge_index[1][nb]
        d = torch.cat([torch.zeros(on.numel(), 3, dtype=positions.dtype, device=positions.device),
                       positions[rcv] - positions[snd] + shifts[nb]], dim=0)
        recv = torch.cat([on, rcv])
        wt = torch.cat([w[on], w[snd]]) * m.to(positions.dtype)[batch[recv]]
        return d, recv, wt, None
    if mode != 'centroid':
        raise ValueError('virtual_carrier_edges mode: sites or centroid, not %r' % mode)
    n_batch = cell.shape[0]
    check_min_image(cell, cutoff)
    centre = weighted_centre(w if centroid_gamma == 1.0 else w ** centroid_gamma, positions, cell, batch, n_batch)
    d_all = minimum_image(positions - centre[batch], cell[batch])
    inside = (d_all * d_all).sum(dim=-1) < cutoff ** 2
    recv = torch.nonzero(inside, as_tuple=False).reshape(-1)
    return d_all[recv], recv, m.to(positions.dtype)[batch[recv]], centre


def softened_direction(d: torch.Tensor, soft_a: float = 0.4) -> Tuple[torch.Tensor, torch.Tensor]:
    """(u [E, 3], r [E, 1]): u = d / sqrt(|d|^2 + a^2), r = |d| (1e-24 added inside the root keeps the gradient finite
    at d = 0). Feed u to the backbone's (unnormalised Cartesian / spherical-harmonic) angular basis."""
    r = torch.sqrt((d * d).sum(dim=-1, keepdim=True) + 1e-24)
    return d / torch.sqrt(r * r + soft_a * soft_a), r


def sinc_bessel(r: torch.Tensor, freqs: torch.Tensor, prefactor) -> torch.Tensor:
    """A Bessel radial basis prefactor sin(w r) / r written as prefactor w sinc(w r / pi), finite at r = 0; r [E, 1],
    freqs [n_rbf] -> [E, n_rbf]. Multiply by the backbone's cutoff function."""
    return prefactor * freqs * torch.sinc(freqs * r / math.pi)


# ---------------------------------------------------------------------------------------------------- long-range energy
def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class WidthEwald(nn.Module):
    """LES's periodic k-space Ewald (same k grid, cutoff and norm factor, self interaction removed) with its own Gaussian
    width s for every charge:
        E = norm [ (1/V) sum_k f_k / k^2 |S(k)|^2 - sum_i sum_{a,b on atom i} q_a q_b / (s_ab (2 pi)^1.5) ],
        S(k) = sum_charges q exp(-s^2 k^2 / 4) exp(i k.r),   s_ab^2 = (s_a^2 + s_b^2) / 2.
    With every width equal to the LES sigma this is LES's energy. One charge of width s has an RMS radius sqrt(3/2) s."""

    def __init__(self, dl: float = 2.0, N_max: int = 10, norm_factor: float = 90.4756):
        super().__init__()
        self.dl, self.N_max, self.norm_factor = float(dl), int(N_max), float(norm_factor)
        self.k_sq_max = (TWOPI / self.dl) ** 2
        r = torch.arange(-self.N_max, self.N_max + 1)
        nvec = torch.stack(torch.meshgrid(r, r, r, indexing='ij'), dim=-1).reshape(-1, 3)
        non_zero = nvec != 0
        has_nz = non_zero.any(dim=1)
        first = torch.argmax(non_zero.to(torch.int), dim=1)
        sign = torch.gather(nvec, 1, first.unsqueeze(1)).squeeze(1)
        hemi = (sign > 0) | (~has_nz)
        self.register_buffer('nvec', nvec[hemi & has_nz], persistent=False)   # half space without k = 0 (factor 2)

    def forward(self, r: torch.Tensor, cell: torch.Tensor, batch: torch.Tensor, n_batch: int,
                charges: Sequence[torch.Tensor], widths: Sequence[torch.Tensor]) -> torch.Tensor:
        """r [N, 3], cell [B, 3, 3], batch [N]; charges / widths: lists of [N] tensors (every list entry sits on the
        points r). Returns the energy [B]."""
        dtype, device = r.dtype, r.device
        nvec = self.nvec.to(device=device, dtype=dtype)
        G = TWOPI * torch.linalg.inv(cell).transpose(-2, -1)
        kvec = torch.einsum('kj,bji->bki', nvec, G)
        k_sq = (kvec * kvec).sum(dim=-1)
        valid = ((k_sq > 0) & (k_sq <= self.k_sq_max)).to(dtype)
        weight = 2.0 * valid / (k_sq + 1e-12)
        kr = (kvec[batch] * r.unsqueeze(1)).sum(dim=-1)
        ksq_a = k_sq[batch]
        amp = 0.0
        for q, s in zip(charges, widths):
            amp = amp + q.unsqueeze(1) * torch.exp(-0.25 * (s * s).unsqueeze(1) * ksq_a)
        s_re = segment_sum(amp * torch.cos(kr), batch, n_batch)
        s_im = segment_sum(amp * torch.sin(kr), batch, n_batch)
        e = (weight * (s_re * s_re + s_im * s_im)).sum(dim=-1) / torch.det(cell)
        on_site = 0.0
        for a in range(len(charges)):
            for b in range(len(charges)):
                s_ab = torch.sqrt(0.5 * (widths[a] * widths[a] + widths[b] * widths[b]))
                on_site = on_site + charges[a] * charges[b] / s_ab
        e = e - segment_sum(on_site / TWOPI ** 1.5, batch, n_batch)
        return self.norm_factor * e


class VirtualCarrierLR(nn.Module):
    """Long-range energy of F's host charges plus the carrier charge k m, k = carrier_sign / sqrt(epsilon_e):
        'field'     the carrier on the atoms, k m w_i on atom i (va_site; w = smooth_cut_weights(p))
        'centroid'  one Gaussian of charge k m at the centre c (va_centre)
    width: 'fixed' (LES sigma for every charge, LES Ewald_vectorized), 'learn', 'learnfix' (centroid only), 'pspread'
    (centroid only); see the module docstring. forward(...) -> (energy [n_batch], sigma_c [n_batch])."""

    def __init__(self, mode: str, width: str = 'fixed', sigma: float = 1.0, dl: float = 2.0, N_max: int = 10,
                 carrier_sign: float = -1.0, epsilon_e: float = 1.0, is_periodic: bool = True,
                 remove_self_interaction: bool = True, norm_factor: float = 90.4756):
        super().__init__()
        if mode not in ('field', 'centroid'):
            raise ValueError('VirtualCarrierLR mode: field or centroid, not %r' % mode)
        if width not in ('fixed', 'learn', 'learnfix', 'pspread'):
            raise ValueError('VirtualCarrierLR width: fixed, learn, learnfix or pspread, not %r' % width)
        if width in ('learnfix', 'pspread') and mode != 'centroid':
            raise ValueError('the %s width is for the centroid carrier only' % width)
        if width != 'fixed' and not (is_periodic and remove_self_interaction):
            raise ValueError('a carrier width of its own needs the periodic Ewald with the self interaction removed')
        if epsilon_e <= 0.0:
            raise ValueError('epsilon_e must be positive')
        self.mode, self.width, self.sigma = mode, width, float(sigma)
        self.k = float(carrier_sign) / float(epsilon_e) ** 0.5
        if width == 'fixed':
            self.ewald = Ewald_vectorized(sigma=sigma, dl=dl, is_periodic=is_periodic, N_max=N_max,
                                          remove_self_interaction=remove_self_interaction, norm_factor=norm_factor)
        else:
            self.ewald = WidthEwald(dl=dl, N_max=N_max, norm_factor=norm_factor)
        if width in ('learn', 'learnfix'):
            self.width_t = nn.Parameter(torch.tensor(_inv_softplus(0.02)))          # sigma_c starts at sigma + 0.02
        elif width == 'pspread':
            self.width_t0 = nn.Parameter(torch.tensor(_inv_softplus(0.02)))
            self.width_ta = nn.Parameter(torch.tensor(_inv_softplus(2.0 / 3.0)))

    def carrier_sigma(self, n_batch: int, dtype: torch.dtype, w: Optional[torch.Tensor] = None,
                      positions: Optional[torch.Tensor] = None, cell: Optional[torch.Tensor] = None,
                      batch: Optional[torch.Tensor] = None, centre: Optional[torch.Tensor] = None) -> torch.Tensor:
        sp = nn.functional.softplus
        if self.width == 'fixed':
            return torch.full((n_batch,), self.sigma, dtype=dtype)
        if self.width in ('learn', 'learnfix'):
            return (self.sigma + sp(self.width_t)).to(dtype).expand(n_batch)
        d = minimum_image(positions - centre[batch], cell[batch])
        spread = segment_sum(w * (d * d).sum(dim=-1), batch, n_batch)                  # <|r - c|^2>_w
        return torch.sqrt(self.sigma ** 2 + sp(self.width_t0) + sp(self.width_ta) * spread)

    def forward(self, q: torch.Tensor, positions: torch.Tensor, cell: torch.Tensor, batch: torch.Tensor,
                m: torch.Tensor, w: Optional[torch.Tensor] = None,
                centre: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """q [N] host charges (LES units), positions [N, 3], cell [B, 3, 3], batch [N], m [B]; 'field' needs w [N]
        (sum 1 per structure), 'centroid' needs centre [B, 3] ('pspread' also w)."""
        n_batch = cell.shape[0]
        q = q.reshape(-1)
        m = m.to(q.dtype)
        sig_c = self.carrier_sigma(n_batch, q.dtype, w, positions, cell, batch, centre).to(q.device)
        if self.mode == 'field':
            if w is None:
                raise ValueError("VirtualCarrierLR 'field' needs w")
            c = self.k * m[batch] * w.reshape(-1)
            if self.width == 'fixed':
                e = self.ewald(q=q + c, r=positions, cell=cell, batch=batch)[0].reshape(n_batch, -1).sum(dim=1)
            else:
                e = self.ewald(positions, cell, batch, n_batch, [q, c],
                               [torch.full_like(q, self.sigma), sig_c[batch]])
            return e, sig_c
        if centre is None:
            raise ValueError("VirtualCarrierLR 'centroid' needs centre")
        n = q.shape[0]
        r_all = torch.cat([positions, centre], dim=0)
        b_all = torch.cat([batch, torch.arange(n_batch, device=batch.device, dtype=batch.dtype)])
        if self.width == 'fixed':
            e = self.ewald(q=torch.cat([q, self.k * m], dim=0), r=r_all, cell=cell,
                           batch=b_all)[0].reshape(n_batch, -1).sum(dim=1)
            return e, sig_c
        q_host = torch.cat([q, q.new_zeros(n_batch)])
        q_car = torch.cat([q.new_zeros(n), self.k * m])
        s_h = torch.full_like(q_host, self.sigma)
        s_c = torch.cat([torch.full_like(q, self.sigma), sig_c])
        e = self.ewald(r_all, cell, b_all, n_batch, [q_host, q_car], [s_h, s_c])
        if self.width == 'learnfix':
            # the carrier alone (its own periodic images) at the LES width: frame-independent for a fixed cell, so it
            # must not depend on sigma_c
            b_c, q_c = b_all[n:], self.k * m
            e = e + (self.ewald(centre, cell, b_c, n_batch, [q_c], [torch.full_like(q_c, self.sigma)])
                     - self.ewald(centre, cell, b_c, n_batch, [q_c], [sig_c]))
        return e, sig_c


# ---------------------------------------------------------------------------------------------------- training modes
class FrozenPredictor(nn.Module):
    """Two-stage P: a trained predictor network, frozen. forward(data) runs it on a shallow copy of the input dict
    (shared tensors, so p keeps its derivative with respect to the positions; none of P's outputs overwrite the
    caller's keys) and returns data[source_key] of the copy (the occupation p, [n_atoms]).
    Every submodule is marked frozen_predictor = True (a trainer's stage settings must skip it) and its model_outputs
    lists are emptied (an outer model that collects them must not return P's keys as its own)."""

    def __init__(self, stages: Sequence[nn.Module], source_key: str = 'LES_occupation', source: str = ''):
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.source_key, self.source = source_key, source
        for q in self.parameters():
            q.requires_grad_(False)
        for mod in self.modules():
            mod.frozen_predictor = True
            if mod is not self and hasattr(mod, 'model_outputs'):
                mod.model_outputs = []

    @classmethod
    def from_checkpoint(cls, path: str, source_key: str = 'LES_occupation', map_location='cpu',
                        stop_at: str = 'LesWrapper') -> 'FrozenPredictor':
        """From a pickled network with .representation and .output_modules (e.g. a CACE NeuralNetworkPotential, a
        run's checkpoint_best.pth): its representation and output modules up to the first one whose class name starts
        with stop_at (the LES wrapper that writes the occupation), the rest dropped (SR head, energy sum, forces).
        A directory is read as <dir>/checkpoint_best.pth."""
        if os.path.isdir(path):
            path = os.path.join(path, 'checkpoint_best.pth')
        net = torch.load(path, map_location=map_location, weights_only=False)
        mods = []
        for mod in net.output_modules:
            mods.append(mod)
            if type(mod).__name__.startswith(stop_at):
                break
        else:
            raise ValueError('%s: no %s among the output modules' % (path, stop_at))
        return cls([_Call(net.representation)] + [_Call(m, training_kw=True) for m in mods], source_key=source_key,
                   source=os.path.abspath(path))

    def forward(self, data, training: bool = False):
        d = dict(data)
        for stage in self.stages:
            d = stage(d, training=training)
        return d[self.source_key].reshape(-1)


class _Call(nn.Module):
    """A module called as mod(data) or mod(data, training=...)."""

    def __init__(self, mod: nn.Module, training_kw: bool = False):
        super().__init__()
        self.mod, self.training_kw = mod, training_kw

    def forward(self, data, training: bool = False):
        return self.mod(data, training=training) if self.training_kw else self.mod(data)


class DualForces(nn.Module):
    """Joint mode: forces of several energies on the same positions, data[forces_keys[i]] = -d data[energy_keys[i]] / dR
    (main model F and predictor P), the graph kept until the last one."""

    def __init__(self, energy_keys: Sequence[str] = ('CACE_energy', 'CACE_energy_aux'),
                 forces_keys: Sequence[str] = ('CACE_forces', 'CACE_forces_aux')):
        super().__init__()
        self.energy_keys, self.forces_keys = list(energy_keys), list(forces_keys)
        self.model_outputs = list(forces_keys)
        self.required_derivatives = ['positions']

    def forward(self, data, training: bool = False, output_index=None):
        pos = data['positions']
        for i, (ek, fk) in enumerate(zip(self.energy_keys, self.forces_keys)):
            last = i == len(self.energy_keys) - 1
            g = torch.autograd.grad(data[ek].sum(), pos, retain_graph=training or not last, create_graph=training,
                                    allow_unused=True)[0]
            data[fk] = -g if g is not None else torch.zeros_like(pos)
        return data
