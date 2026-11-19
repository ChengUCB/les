"""
Charge offsets in LES: charge_offset_mode = True.

Every multipole and induced term of atom i sits at r_i + delta_i, delta_i built
from latent_offsets (soft cap in Les).

Everything runs in float64 on the CPU, for both boundary conditions:
  1  offsets off == origin/main, bit for bit (energy, forces, BEC, outputs)
  2  offsets     == the model without offsets evaluated at the shifted sites
  3  forces == central finite differences, with position-dependent delta;
     finite gradient with respect to the raw offsets at delta = 0
  4  BEC == finite differences of P (complex P when periodic)
  5  lattice translation of an atom, and of a charge site alone
  6  legacy Ewald (is_periodic unset) == vectorised Ewald with offsets
  +  offset cap, settings, old pickles

Run:
    PYTHONPATH=src python -m pytest test/test_charge_offsets.py -q
"""
import io
import itertools
import math
import os
import subprocess
import sys

import pytest
import torch

from les import Les

DT = torch.float64
PERIODIC = (True, False)
CELLS = (((9.0, 0.8, 0.4), (0.0, 8.5, 0.9), (0.0, 0.0, 9.5)),
         ((8.0, 0.0, 0.0), (0.5, 9.0, 0.0), (0.3, 0.2, 8.8)))


# ---------------------------------------------------------------------------
# inputs: positions, and multipoles/offsets as smooth functions of them
# ---------------------------------------------------------------------------
def geometry(periodic, n_per=(6, 7), seed=0):
    """Jittered grid in each cell (no close contacts); zero cells when open."""
    g = torch.Generator().manual_seed(seed)
    grid = torch.tensor(list(itertools.product((0.25, 0.75), repeat=3)), dtype=DT)
    rs, batch = [], []
    for b, n in enumerate(n_per):
        cell = torch.tensor(CELLS[b], dtype=DT)
        frac = grid[torch.randperm(8, generator=g)[:n]]
        frac = frac + 0.08 * (2 * torch.rand(n, 3, generator=g, dtype=DT) - 1)
        rs.append(frac @ cell)
        batch.append(torch.full((n,), b, dtype=torch.long))
    if periodic:
        cell = torch.tensor(CELLS[:len(n_per)], dtype=DT)
    else:
        cell = torch.zeros(len(n_per), 3, 3, dtype=DT)
    return torch.cat(rs), cell, torch.cat(batch)


def features(r, batch, terms, seed=1, offset_scale=1.0):
    """A stand-in for a host model: rotation-equivariant, translation-invariant
    multipoles and raw offsets that depend smoothly on all positions. Q carries
    a trace part, which LES does not forbid. Returns (Les kwargs, raw offsets)."""
    g = torch.Generator().manual_seed(seed)
    n = r.shape[0]

    def rand(*shape):
        return torch.rand(*shape, generator=g, dtype=DT)

    q0, cu, cq, ctr, cd, k0, a0 = (rand(n, 1), rand(n, 1), rand(n, 1), rand(n, 1),
                                   rand(n, 1), rand(n, 1), rand(n, 1))
    rij = r.unsqueeze(0) - r.unsqueeze(1)                              # r_j - r_i
    pair = (batch.unsqueeze(0) == batch.unsqueeze(1)) & ~torch.eye(n, dtype=torch.bool)
    w = torch.exp(-(rij ** 2).sum(-1) / 16.0) * pair
    s = w.sum(1)                                                       # [n]
    v = (w.unsqueeze(-1) * rij).sum(1)                                 # [n, 3]
    T = (w[..., None, None] * rij.unsqueeze(-1) * rij.unsqueeze(-2)).sum(1)
    eye = torch.eye(3, dtype=DT)
    tr = T.diagonal(dim1=-2, dim2=-1).sum(-1)

    out = {"latent_charges": (q0 - 0.5) * (1.0 + 0.3 * s.unsqueeze(1))}
    if "u" in terms:
        out["latent_dipoles"] = (0.3 * (cu - 0.5)).unsqueeze(-1) * v.unsqueeze(1)
    if "Q" in terms:
        Q = (cq - 0.5)[..., None, None] * (T - tr[:, None, None] * eye / 3).unsqueeze(1)
        Q = Q + 0.1 * (ctr - 0.5)[..., None, None] * (tr[:, None, None] * eye).unsqueeze(1)
        out["latent_quads"] = 0.03 * Q
    if "k" in terms:
        out["latent_kappas"] = 0.1 + 0.2 * k0 * torch.sigmoid(s).unsqueeze(1)
    if "a" in terms:
        out["latent_alphas"] = 0.1 + 0.2 * a0 * torch.sigmoid(s).unsqueeze(1)
    offsets = offset_scale * 0.4 * (cd - 0.5) * v
    return out, offsets


def model(periodic, offsets=False, cap=0.0):
    args = {"is_periodic": periodic}
    if offsets:
        args.update(charge_offset_mode=True, charge_offset_max=cap)
    return Les(args)


def evaluate(m, r, cell, batch, make, bec=True):
    """Energy, forces and every output; make(r) -> (Les kwargs, offsets or None)."""
    r = r.detach().clone().requires_grad_(True)
    feats, offsets = make(r)
    if offsets is not None:
        feats = dict(feats, latent_offsets=offsets)
    out = m(positions=r, cell=cell, batch=batch, compute_bec=bec, **feats)
    E = out["E_lr"]
    F = -torch.autograd.grad(E.sum(), r)[0]
    res = {"E": E.detach(), "F": F.detach()}
    for k, v in out.items():
        if k != "E_lr" and v is not None:
            res[k] = v.detach()
    return res


def energy(m, r, cell, batch, make):
    with torch.no_grad():
        feats, offsets = make(r)
        if offsets is not None:
            feats = dict(feats, latent_offsets=offsets)
        return m(positions=r, cell=cell, batch=batch, **feats)["E_lr"].sum()


def assert_close(a, b, atol=1e-12, rtol=1e-12, keys=None, exact=False, label=""):
    keys = keys if keys is not None else sorted(set(a) & set(b))
    for k in keys:
        assert k in a and k in b, f"{label}: output {k} missing"
        if exact:
            assert torch.equal(a[k], b[k]), (
                f"{label}: {k} differs, max {float((a[k] - b[k]).abs().max()):.3e}")
        else:
            torch.testing.assert_close(a[k], b[k], atol=atol, rtol=rtol,
                                       msg=lambda m: f"{label}: {k}: {m}")


ALL_TERMS = ["q" + "u" * u + "Q" * Q + "k" * k + "a" * a
             for u, Q, k, a in itertools.product((0, 1), repeat=4)]


# ---------------------------------------------------------------------------
# 1  offsets off == origin/main, bit for bit
# ---------------------------------------------------------------------------
OFF_CASES = [(ewald, p, terms, rsi) for ewald in ("vectorised", "legacy") for p in PERIODIC
             for terms in ("q", "qu", "quQ", "qka", "quQka") for rsi in (True, False)]


def _off_results():
    """Run in this process and, by test_offsets_off_bitwise_equals_main, under
    origin/main: so it may only use the API both versions have."""
    res = {}
    for ewald, p, terms, rsi in OFF_CASES:
        r, cell, batch = geometry(p)
        args = {"remove_self_interaction": rsi}
        if ewald == "vectorised":
            args["is_periodic"] = p       # unset: the legacy Ewald (periodic or open by the cell)
        m = Les(args)
        make = lambda rr: (features(rr, batch, terms)[0], None)   # noqa: E731
        res[f"{ewald}-{p}-{terms}-{rsi}"] = evaluate(m, r, cell, batch, make, bec=True)
    return res


def test_offsets_off_bitwise_equals_main(tmp_path):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main = tmp_path / "main"
    main.mkdir()
    tar = subprocess.run(["git", "archive", "origin/main", "src"], cwd=repo,
                         capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(main)], input=tar, check=True)
    out = tmp_path / "main.pt"
    code = (f"import sys; sys.path[:0] = [{str(main / 'src')!r}, {os.path.dirname(__file__)!r}]\n"
            f"import les, torch\n"
            f"assert les.__file__.startswith({str(main)!r}), les.__file__\n"
            f"import test_charge_offsets as t\n"
            f"torch.save(t._off_results(), {str(out)!r})\n")
    subprocess.run([sys.executable, "-c", code], check=True)
    ref = torch.load(out)
    got = _off_results()
    assert ref.keys() == got.keys()
    for case in ref:
        assert ref[case].keys() == got[case].keys(), f"{case}: output keys differ"
        assert_close(got[case], ref[case], exact=True, label=f"off {case}")


# ---------------------------------------------------------------------------
# 2  offsets == the model without offsets at the shifted sites
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("periodic", PERIODIC)
def test_offsets_equal_parent_at_shifted_sites(periodic):
    r, cell, batch = geometry(periodic)
    parent = model(periodic)
    m = model(periodic, offsets=True)                      # no cap: delta = raw offsets
    for terms in ALL_TERMS:
        feats, offsets = features(r, batch, terms)         # held fixed: not functions of r
        feats = {k: v.detach() for k, v in feats.items()}
        offsets = offsets.detach()
        # with fixed dipoles the BEC is only defined if they are induced (alpha)
        bec = "u" not in terms or "a" in terms
        got = evaluate(m, r, cell, batch, lambda rr: (feats, offsets), bec=bec)
        ref = evaluate(parent, r + offsets, cell, batch, lambda rr: (feats, None), bec=bec)
        assert_close(got, ref, keys=sorted(ref), atol=1e-12, rtol=1e-11,
                     label=f"offsets vs shifted {periodic} [{terms}]")


# ---------------------------------------------------------------------------
# 3  forces vs central finite differences; gradient at delta = 0
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("periodic", PERIODIC)
def test_forces_match_finite_differences(periodic):
    r, cell, batch = geometry(periodic, n_per=(5,))
    m = model(periodic, offsets=True, cap=0.5)
    make = lambda rr: features(rr, batch, "quQka", offset_scale=2.0)   # noqa: E731
    F = evaluate(m, r, cell, batch, make, bec=False)["F"]
    h = 1e-5
    F_fd = torch.zeros_like(r)
    for i in range(r.shape[0]):
        for d in range(3):
            rp, rm = r.clone(), r.clone()
            rp[i, d] += h
            rm[i, d] -= h
            F_fd[i, d] = -(energy(m, rp, cell, batch, make) - energy(m, rm, cell, batch, make)) / (2 * h)
    torch.testing.assert_close(F, F_fd, atol=2e-8, rtol=1e-6)


@pytest.mark.parametrize("periodic", PERIODIC)
def test_offset_gradient_at_zero(periodic):
    """Offset heads start at delta = 0: the gradient there must be finite and right."""
    r, cell, batch = geometry(periodic, n_per=(5,))
    feats, _ = features(r, batch, "quQka")
    feats = {k: v.detach() for k, v in feats.items()}
    m = model(periodic, offsets=True, cap=1.0)
    v = torch.zeros_like(r, requires_grad=True)
    E = m(positions=r, cell=cell, batch=batch, latent_offsets=v, **feats)["E_lr"].sum()
    g = torch.autograd.grad(E, v)[0]
    assert torch.isfinite(g).all()
    h = 1e-6
    g_fd = torch.zeros_like(r)
    with torch.no_grad():
        for i in range(r.shape[0]):
            for d in range(3):
                vp, vm = torch.zeros_like(r), torch.zeros_like(r)
                vp[i, d], vm[i, d] = h, -h
                ep = m(positions=r, cell=cell, batch=batch, latent_offsets=vp, **feats)["E_lr"].sum()
                em = m(positions=r, cell=cell, batch=batch, latent_offsets=vm, **feats)["E_lr"].sum()
                g_fd[i, d] = (ep - em) / (2 * h)
    torch.testing.assert_close(g, g_fd, atol=1e-7, rtol=1e-6)


# ---------------------------------------------------------------------------
# 4  BEC vs finite differences of the polarization
# ---------------------------------------------------------------------------
def _polarization(m, r, cell, make, periodic):
    """P from the model's own outputs: charge part (complex per reciprocal vector
    when periodic, as in BEC.compute_pol_pbc) and dipole part."""
    feats, offsets = make(r)
    batch = torch.zeros(r.shape[0], dtype=torch.long)
    out = m(positions=r, cell=cell, batch=batch, latent_offsets=offsets, **feats)
    q = out["latent_charges"]
    q = q - q.mean(dim=0, keepdim=True)                     # remove_mean=True
    sites = r + out["latent_offsets"]
    if periodic:
        phase = torch.exp(2j * math.pi * (sites @ torch.linalg.inv(cell[0])))
        Pq = (q * phase).sum(0) / (2j * math.pi)
    else:
        Pq = (q * sites).sum(0)
    Pu = out["latent_dipoles"].sum((0, 1)) if out["latent_dipoles"] is not None else None
    return Pq, Pu, sites


@pytest.mark.parametrize("periodic", PERIODIC)
def test_bec_matches_finite_differences(periodic):
    r, cell, batch = geometry(periodic, n_per=(6,))
    m = model(periodic, offsets=True, cap=0.5)
    make = lambda rr: features(rr, batch, "quka", offset_scale=2.0)   # noqa: E731
    Z = evaluate(m, r, cell, batch, make, bec=True)["BEC"]           # [N, 2, 3, 3]
    with torch.no_grad():
        _, _, sites = _polarization(m, r, cell, make, periodic)
        h = 1e-5
        n = r.shape[0]
        Zq = torch.zeros(n, 3, 3, dtype=torch.complex128)
        Zu = torch.zeros(n, 3, 3, dtype=DT)
        for i in range(n):
            for d in range(3):
                rp, rm = r.clone(), r.clone()
                rp[i, d] += h
                rm[i, d] -= h
                Pp, Pup, _ = _polarization(m, rp, cell, make, periodic)
                Pm, Pum, _ = _polarization(m, rm, cell, make, periodic)
                Zq[i, :, d] = (Pp - Pm) / (2 * h)
                Zu[i, :, d] = (Pup - Pum) / (2 * h)
        if periodic:
            # dephase at the charge site, project back to Cartesian (BEC.forward)
            phase = torch.exp(2j * math.pi * (sites @ torch.linalg.inv(cell[0])))
            Zq = cell[0].T @ (Zq * phase.conj().unsqueeze(2)).real
        else:
            Zq = Zq.real
    torch.testing.assert_close(Z[:, 0], Zq, atol=1e-8, rtol=1e-6)
    torch.testing.assert_close(Z[:, 1], Zu, atol=1e-8, rtol=1e-6)


# ---------------------------------------------------------------------------
# 5  lattice translation
# ---------------------------------------------------------------------------
def test_lattice_translation():
    """Unwrapped sites: an atom, or a charge site alone, moved by a lattice vector
    is the same structure."""
    r, cell, batch = geometry(True)
    feats, offsets = features(r, batch, "quQka")
    feats = {k: v.detach() for k, v in feats.items()}
    offsets = offsets.detach()
    m = model(True, offsets=True)                          # no cap, so delta + L is reachable
    L = cell[0, 0] - 2.0 * cell[0, 2] + cell[0, 1]
    ref = evaluate(m, r, cell, batch, lambda rr: (feats, offsets))
    r_moved = r.clone()
    r_moved[2] += L
    got = evaluate(m, r_moved, cell, batch, lambda rr: (feats, offsets))
    assert_close(got, ref, atol=1e-10, rtol=1e-10, label="atom + L")
    moved = offsets.clone()
    moved[2] += L
    got = evaluate(m, r, cell, batch, lambda rr: (feats, moved))
    assert_close(got, ref, keys=["E", "F", "latent_charges", "latent_dipoles", "BEC"],
                 atol=1e-10, rtol=1e-10, label="site + L")


# ---------------------------------------------------------------------------
# 6  legacy Ewald == vectorised Ewald with offsets
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("periodic", PERIODIC)
def test_legacy_equals_vectorised(periodic):
    """Every term, with the soft cap active: energy, forces, BEC, induced charges
    and dipoles, capped offsets. The two implementations sum in different orders,
    so they agree to round-off. Each k-grid holds every k inside the cutoff for
    these cells."""
    r, cell, batch = geometry(periodic)
    make = lambda rr: features(rr, batch, "quQka", offset_scale=3.0)   # noqa: E731
    got = evaluate(Les({"charge_offset_mode": True, "charge_offset_max": 0.8}), r, cell, batch, make)
    ref = evaluate(model(periodic, offsets=True, cap=0.8), r, cell, batch, make)
    assert got.keys() == ref.keys()
    assert float(got["latent_offsets"].norm(dim=-1).max()) > 0.3      # offsets matter here
    assert_close(got, ref, atol=1e-10, rtol=1e-10, label=f"legacy {periodic}")


# ---------------------------------------------------------------------------
# +  cap, settings, old pickles
# ---------------------------------------------------------------------------
def test_offset_cap():
    v = torch.randn(200, 3, generator=torch.Generator().manual_seed(2), dtype=DT) * 3.0
    m = model(True, offsets=True, cap=0.7)
    d = m._cap_offsets(v)
    assert (d.norm(dim=-1) < 0.7).all()
    torch.testing.assert_close(d / d.norm(dim=-1, keepdim=True), v / v.norm(dim=-1, keepdim=True))
    small = 1e-4 * v
    # delta = v (1 - |v|^2 / (2 d^2) + ...): identity at small v, to second order
    rel = (m._cap_offsets(small) - small).norm(dim=-1) / small.norm(dim=-1)
    torch.testing.assert_close(rel, (small ** 2).sum(-1) / (2 * 0.7 ** 2), rtol=1e-3, atol=0.0)
    for cap in (0.0, None):
        assert torch.equal(model(True, offsets=True, cap=cap)._cap_offsets(v), v)


def test_settings():
    assert Les({"is_periodic": True}).charge_offset_mode is False
    r, cell, batch = geometry(True, n_per=(4,))
    q = torch.rand(4, 1, dtype=DT)
    with pytest.raises(ValueError, match="charge_offset_mode is False"):
        Les({"is_periodic": True})(positions=r, cell=cell, latent_charges=q,
                                   latent_offsets=torch.zeros_like(r))
    with pytest.raises(ValueError, match="latent_offsets must be provided"):
        model(True, offsets=True)(positions=r, cell=cell, latent_charges=q)


def _reloaded(m):
    buf = io.BytesIO()
    torch.save(m, buf)
    buf.seek(0)
    return torch.load(buf, weights_only=False)


def test_old_pickle_gets_offsets_off():
    m = Les({"is_periodic": True})
    obj = _reloaded(m)
    for flag in ("charge_offset_mode", "charge_offset_max"):
        obj.__dict__.pop(flag)
    old = _reloaded(obj)
    assert old.charge_offset_mode is False and old.charge_offset_max == 1.0
    r, cell, batch = geometry(True, n_per=(4,))
    make = lambda rr: (features(rr, batch, "qu")[0], None)   # noqa: E731
    assert_close(evaluate(old, r, cell, batch, make), evaluate(m, r, cell, batch, make),
                 exact=True, label="old pickle")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
