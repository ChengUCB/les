"""
Charge equilibration in LES: use_qeq = True.

The charges of the selected atoms minimise
    U = E_ewald (with their Gaussian self energy) + sum (chi q + 1/2 J q^2) - sum v q
under a total-charge constraint; the other atoms keep their latent charges.

Everything runs in float64, for both Ewald modules and both boundary conditions, on the CPU or, with
LES_TEST_DEVICE=cuda, on the GPU (there the bit-for-bit comparisons allow 1e-12):
  1  use_qeq off == the model before (covered by test_charge_offsets.py against origin/main)
  2  coupling matrix: 1/2 q^T A q == Ewald energy; legacy == vectorised
  3  stationarity (equal chemical potential on the selected atoms) and the charge constraint
  4  the solution is the minimum of U
  5  forces == central finite differences, every option
  6  MetalWall flag == Qeq with chi = 0, J = 0; its two energy definitions coincide
  7  electrode physics: image charge, equipotential, capacitor, applied potential
  8  charge offsets: solve at the sites or at the nuclei
  9  batch == single structures; cache; old pickles; TorchScript
 10  float32 on the GPU, the precision and the device of real use == float64 (skipped without a GPU)

Run:
    PYTHONPATH=src python -m pytest test/test_qeq.py -q
    LES_TEST_DEVICE=cuda PYTHONPATH=src python -m pytest test/test_qeq.py -q
"""
import copy
import io
import itertools
import math
import os
import sys

import pytest
import torch

from les import Les

DT = torch.float64
torch.set_default_dtype(DT)
DEV = os.environ.get("LES_TEST_DEVICE", "cpu")
EXACT = 0.0 if DEV == "cpu" else 1e-12      # sums on the GPU are not reproducible to the last bit
CASES = list(itertools.product((True, False), (True, False)))          # (periodic, vectorised)
IDS = [f"{'periodic' if p else 'open'}-{'vectorised' if v else 'legacy'}" for p, v in CASES]
EPS0 = 0.00552635      # e / (V A)


@pytest.fixture(autouse=True, scope="module")
def _device():
    torch.set_default_device(DEV)
    yield
    torch.set_default_device("cpu")


def randn(*shape, generator, device=None):
    """The same numbers on every device."""
    return torch.randn(*shape, generator=generator, device="cpu", dtype=DT).to(device or DEV)


def model(periodic, vectorised, **args):
    if vectorised:
        args["is_periodic"] = periodic
    return Les({"use_qeq": True, **args})


def geometry(periodic, n_structures=2, n=18, seed=0):
    """Jittered grid, 6 'metal' atoms (Z = 78) then O, H, H in each structure; triclinic when periodic."""
    g = torch.Generator().manual_seed(seed)
    grid = torch.stack(torch.meshgrid(*[torch.arange(3.0)] * 3, indexing="ij"), -1).reshape(-1, 3)[:n] * 2.6
    r = torch.cat([grid + 0.4 * torch.rand(n, 3, generator=g, device="cpu", dtype=DT).to(DEV) for _ in range(n_structures)])
    batch = torch.arange(n_structures).repeat_interleave(n)
    cell = torch.eye(3).repeat(n_structures, 1, 1) * (8.3 if periodic else 0.0)
    if periodic:
        cell[:, 0, 1] = 0.9
    Z = torch.tensor(([78] * 6 + [8, 1, 1] * 4)[:n] * n_structures)
    return r, cell, batch, Z


def latents(r, seed=1):
    """Latent inputs that depend on the positions, as the outputs of a model do."""
    n = r.shape[0]
    g = torch.Generator().manual_seed(seed)
    f = torch.sin(1.3 * r).sum(1)
    rnd = lambda *s: randn(*s, generator=g, device=r.device).to(r.dtype)   # noqa: E731
    return {"latent_charges": 0.5 * rnd(n) + 0.1 * f, "latent_chi": rnd(n) + 0.2 * f,
            "latent_hardness": rnd(n) + 0.2 * f, "latent_dipoles": 0.3 * rnd(n, 3) * (1 + 0.1 * f[:, None]),
            "applied_potential": 0.5 * rnd(n) + 0.3 * r[:, 2], "latent_offsets": 0.3 * rnd(n, 3) * (1 + 0.1 * f[:, None])}


def call(m, r, cell, batch, Z, keys, **extra):
    lat = latents(r)
    kw = {k: lat[k] for k in keys}
    kw.update(extra)
    return m(positions=r, cell=cell, batch=batch, atomic_numbers=Z, **kw)


def sums(x, batch, n):
    return torch.zeros(n, dtype=DT).index_add(0, batch, x)


# ---------------------------------------------------------------------------
# 2  coupling matrix
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("periodic,vectorised", CASES, ids=IDS)
def test_coupling_matrix_gives_the_ewald_energy(periodic, vectorised):
    r, cell, batch, _ = geometry(periodic)
    args = {"remove_self_interaction": False}
    if vectorised:
        args["is_periodic"] = periodic
    m = Les(args)
    q = randn(r.shape[0], generator=torch.Generator().manual_seed(3))
    A = m.ewald.compute_A_matrix(r, cell, batch)
    assert torch.equal(A, A * (batch[:, None] == batch[None, :])), "A must be block diagonal"
    torch.testing.assert_close(A, A.T, rtol=0, atol=1e-12)
    E = m(positions=r, cell=cell, batch=batch, latent_charges=q)["E_lr"]
    torch.testing.assert_close(sums(0.5 * q * (A @ q), batch, 2), E, rtol=1e-12, atol=1e-12)
    # a subset of the atoms gives the matching block
    sub = torch.arange(r.shape[0]) % 3 == 0
    torch.testing.assert_close(m.ewald.compute_A_matrix(r[sub], cell, batch[sub]), A[sub][:, sub], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("periodic", (True, False))
def test_coupling_matrix_legacy_equals_vectorised(periodic):
    r, cell, batch, _ = geometry(periodic)
    A_legacy = Les({}).ewald.compute_A_matrix(r, cell, batch)
    A_vec = Les({"is_periodic": periodic}).ewald.compute_A_matrix(r, cell, batch)
    torch.testing.assert_close(A_legacy, A_vec, rtol=1e-11, atol=1e-11)


# ---------------------------------------------------------------------------
# 3, 4  stationarity, constraint, minimum
# ---------------------------------------------------------------------------
OPTIONS = {
    "all atoms": ({}, ("latent_chi",)),
    "all atoms, constant J": ({"qeq_hardness_trainable": False, "qeq_hardness": 0.7}, ("latent_chi",)),
    "all atoms, atom-wise J": ({"qeq_hardness_trainable": False, "qeq_hardness": 0.0}, ("latent_chi", "latent_hardness")),
    "one element": ({"qeq_atomic_numbers": [78]}, ("latent_charges", "latent_chi", "latent_hardness")),
    "one element, with dipoles and potential": ({"qeq_atomic_numbers": [78]},
                                                ("latent_charges", "latent_chi", "latent_dipoles", "applied_potential")),
    "selected atoms carry the total charge": ({"qeq_atomic_numbers": [78], "qeq_charge_scope": "selected"},
                                              ("latent_charges", "latent_chi")),
    "self interaction kept": ({"qeq_atomic_numbers": [78], "remove_self_interaction": False}, ("latent_charges", "latent_chi")),
    "element chi": ({"qeq_chi": {78: 0.3, 8: -0.2, 1: 0.1}}, ()),
    "trainable element chi, with latent chi": ({"qeq_chi": {78: 0.3, 8: -0.2}, "qeq_chi_trainable": True, "qeq_atomic_numbers": [78, 8]},
                                               ("latent_charges", "latent_chi", "latent_hardness")),
    "metalwall": ({"qeq_siepmann_sprik": True, "qeq_atomic_numbers": [78]}, ("latent_charges", "latent_dipoles")),
    "metalwall, potential": ({"qeq_siepmann_sprik": True, "qeq_atomic_numbers": [78]}, ("latent_charges", "applied_potential")),
    "metalwall, neutral system": ({"qeq_siepmann_sprik": True, "qeq_atomic_numbers": [78], "qeq_charge_scope": "system"},
                                  ("latent_charges", "applied_potential")),
}


def total_energy(m, r, cell, batch, Z, keys, q_trial=None):
    """U at the model's own charges, or at trial charges for the selected atoms (qeq switched off)."""
    out = call(m, r, cell, batch, Z, keys)
    if q_trial is None:
        return out["E_lr"], out
    lat = latents(r)
    sel = selected(m, Z)
    plain = Les({k: v for k, v in (("is_periodic", m.is_periodic), ("remove_self_interaction", m.remove_self_interaction))})
    kw = {k: lat[k] for k in keys if k in ("latent_dipoles",)}
    E = plain(positions=r, cell=cell, batch=batch, latent_charges=q_trial, **kw)["E_lr"]
    q_s = q_trial[sel]
    if m.remove_self_interaction:
        E = E + sums(q_s ** 2, batch[sel], 2) * m.ewald.norm_factor / (m.sigma * (2 * math.pi) ** 1.5)
    chi = torch.zeros_like(q_s) if m.qeq_siepmann_sprik else m.qeq.electronegativity(
        Z[sel], lat["latent_chi"][sel] if "latent_chi" in keys else None)
    J = torch.zeros_like(q_s) if m.qeq_siepmann_sprik else m.qeq.hardness(
        Z[sel], lat["latent_hardness"][sel] if "latent_hardness" in keys else None)
    if m.qeq_onsite_energy:
        E = E + sums(chi * q_s + 0.5 * J * q_s ** 2, batch[sel], 2)
    if "applied_potential" in keys:
        E = E - sums(lat["applied_potential"] * q_trial, batch, 2)
    return E, out


def selected(m, Z):
    return torch.isin(Z, m.qeq_z) if m.qeq_z.numel() else torch.ones_like(Z, dtype=torch.bool)


@pytest.mark.parametrize("periodic,vectorised", CASES, ids=IDS)
@pytest.mark.parametrize("name", OPTIONS)
def test_stationary_constrained_minimum(periodic, vectorised, name):
    args, keys = OPTIONS[name]
    m = model(periodic, vectorised, **args)
    r, cell, batch, Z = geometry(periodic)
    E, out = total_energy(m, r, cell, batch, Z, keys)
    q = out["latent_charges"].reshape(-1)
    sel = selected(m, Z)
    lat = latents(r)

    # the atoms that are not equilibrated keep their charges
    if "latent_charges" in keys:
        assert torch.equal(q[~sel], lat["latent_charges"][~sel])
    # constraint
    target = torch.zeros(2)
    if m.qeq_charge_scope == "system":
        torch.testing.assert_close(sums(q, batch, 2), target, rtol=0, atol=1e-11)
    else:
        torch.testing.assert_close(sums(q[sel], batch[sel], 2), target, rtol=0, atol=1e-11)

    # the same energy from the trial-charge route, and dU/dq = -lambda on every selected atom
    q_var = q.detach().clone().requires_grad_(True)
    U, _ = total_energy(m, r, cell, batch, Z, keys, q_trial=q_var)
    torch.testing.assert_close(U, E, rtol=1e-11, atol=1e-11)
    dU = torch.autograd.grad(U.sum(), q_var)[0]
    torch.testing.assert_close(dU[sel], -out["qeq_lambda"][batch[sel]], rtol=1e-9, atol=1e-9)

    # minimum: any neutral change of the selected charges raises U
    g = torch.Generator().manual_seed(5)
    for _ in range(5):
        dq = torch.zeros_like(q)
        step = 0.05 * randn(int(sel.sum()), generator=g)
        step = step - (sums(step, batch[sel], 2) / sums(torch.ones_like(step), batch[sel], 2))[batch[sel]]
        dq[sel] = step
        U2, _ = total_energy(m, r, cell, batch, Z, keys, q_trial=q.detach() + dq)
        assert bool((U2 > U - 1e-12).all()), f"{name}: the solution is not the minimum"


def test_element_chi_equals_the_same_numbers_as_latent_chi():
    """qeq_chi per element is latent_chi with those numbers; one number for all atoms only shifts lambda."""
    r, cell, batch, Z = geometry(True)
    table = {78: 0.3, 8: -0.2, 1: 0.1}
    chi = torch.tensor([table[int(z)] for z in Z], dtype=DT)
    a = call(model(True, True, qeq_chi=table), r, cell, batch, Z, ())
    b = call(model(True, True), r, cell, batch, Z, (), latent_chi=chi)
    c = call(model(True, True, qeq_chi=0.7), r, cell, batch, Z, ())
    d = call(model(True, True), r, cell, batch, Z, ())
    for x, y in ((a, b), (c, d)):
        torch.testing.assert_close(x["latent_charges"], y["latent_charges"], rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(x["E_lr"], y["E_lr"], rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(a["qeq_lambda"], b["qeq_lambda"], rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(c["qeq_lambda"], d["qeq_lambda"] - 0.7, rtol=1e-10, atol=1e-10)
    assert float((a["latent_charges"] - d["latent_charges"]).abs().max()) > 1e-3


def test_element_chi_is_trainable_only_when_asked():
    """dE/dchi of an element is the charge on its atoms (the energy is at its minimum in the charges)."""
    fixed = model(True, True, qeq_chi={78: 0.3})
    assert "qeq.chi_elem" not in dict(fixed.named_parameters())
    m = model(True, True, qeq_chi={78: 0.3}, qeq_chi_trainable=True)
    assert float(m.qeq.chi_elem[78]) == pytest.approx(0.3) and float(m.qeq.chi_elem[8]) == 0.0
    r, cell, batch, Z = geometry(True)
    out = call(m, r, cell, batch, Z, ("latent_chi",))
    grad = torch.autograd.grad(out["E_lr"].sum(), m.qeq.chi_elem)[0]
    q = out["latent_charges"].reshape(-1)
    expected = torch.zeros(119, dtype=q.dtype).index_add(0, Z.to(q.device), q.detach())
    torch.testing.assert_close(grad.to(q.dtype), expected, rtol=1e-9, atol=1e-9)
    assert float(expected.abs().max()) > 1e-3


def test_onsite_energy_switch():
    """qeq_onsite_energy False: the same charges, E_lr without sum(chi q + 1/2 J q^2)."""
    r, cell, batch, Z = geometry(True)
    keys = ("latent_charges", "latent_chi", "latent_hardness")
    a = model(True, True, qeq_atomic_numbers=[78])
    b = model(True, True, qeq_atomic_numbers=[78], qeq_onsite_energy=False)
    b.load_state_dict(a.state_dict())
    oa, ob = call(a, r, cell, batch, Z, keys), call(b, r, cell, batch, Z, keys)
    # same charges; (a) and (b) solve the same system in two ways (see Qeq.forward)
    torch.testing.assert_close(oa["latent_charges"], ob["latent_charges"], rtol=1e-12, atol=1e-12)
    sel = Z == 78
    lat = latents(r)
    q = oa["latent_charges"][sel]
    J = a.qeq.hardness(Z[sel], lat["latent_hardness"][sel])
    onsite = sums(lat["latent_chi"][sel] * q + 0.5 * J * q ** 2, batch[sel], 2)
    torch.testing.assert_close(oa["E_lr"] - ob["E_lr"], onsite, rtol=1e-11, atol=1e-11)


# ---------------------------------------------------------------------------
# 5  forces
# ---------------------------------------------------------------------------
FORCE_OPTIONS = dict(OPTIONS)
FORCE_OPTIONS.update({
    "onsite energy off": ({"qeq_atomic_numbers": [78], "qeq_onsite_energy": False},
                          ("latent_charges", "latent_chi", "latent_hardness", "applied_potential")),
    "offsets, at sites": ({"qeq_siepmann_sprik": True, "qeq_atomic_numbers": [78], "charge_offset_mode": True},
                          ("latent_charges", "latent_offsets", "latent_dipoles")),
    "offsets, at nuclei": ({"qeq_siepmann_sprik": True, "qeq_atomic_numbers": [78], "charge_offset_mode": True, "qeq_at_sites": False},
                           ("latent_charges", "latent_offsets", "latent_dipoles")),
})


@pytest.mark.parametrize("periodic,vectorised", CASES, ids=IDS)
@pytest.mark.parametrize("name", FORCE_OPTIONS)
def test_forces_match_finite_differences(periodic, vectorised, name):
    args, keys = FORCE_OPTIONS[name]
    m = model(periodic, vectorised, **args)
    r, cell, batch, Z = geometry(periodic)
    energy = lambda rr: call(m, rr, cell, batch, Z, keys)["E_lr"].sum()   # noqa: E731
    rr = r.clone().requires_grad_(True)
    F = -torch.autograd.grad(energy(rr), rr)[0]
    h = 1e-5
    for i, c in itertools.product((0, 4, 7, 20, 31), range(3)):
        rp, rm = r.clone(), r.clone()
        rp[i, c] += h
        rm[i, c] -= h
        fd = -(energy(rp) - energy(rm)) / (2 * h)
        assert abs(float(fd - F[i, c])) < 2e-7 * max(1.0, float(F.abs().max())), f"{name}: atom {i} component {c}"


# ---------------------------------------------------------------------------
# 6  MetalWall flag
# ---------------------------------------------------------------------------
def test_charge_scope_default_is_system_and_selected_for_metalwall():
    assert model(True, True).qeq_charge_scope == "system"
    assert model(True, True, qeq_atomic_numbers=[78]).qeq_charge_scope == "system"
    assert model(True, True, qeq_siepmann_sprik=True, qeq_atomic_numbers=[78]).qeq_charge_scope == "selected"
    assert model(True, True, qeq_siepmann_sprik=True, qeq_atomic_numbers=[78], qeq_charge_scope="system").qeq_charge_scope == "system"


@pytest.mark.parametrize("periodic,vectorised", CASES, ids=IDS)
def test_metalwall_is_qeq_without_chi_and_hardness(periodic, vectorised):
    r, cell, batch, Z = geometry(periodic)
    keys = ("latent_charges", "latent_dipoles", "applied_potential")
    mw = model(periodic, vectorised, qeq_siepmann_sprik=True, qeq_atomic_numbers=[78])
    ref = call(mw, r, cell, batch, Z, keys)
    # chi and hardness inputs are not used
    out = call(mw, r, cell, batch, Z, keys + ("latent_chi", "latent_hardness"))
    torch.testing.assert_close(out["E_lr"], ref["E_lr"], rtol=0, atol=EXACT)
    for onsite in (True, False):
        qeq = model(periodic, vectorised, qeq_atomic_numbers=[78], qeq_hardness=0.0, qeq_hardness_trainable=False,
                    qeq_onsite_energy=onsite, qeq_charge_scope="selected")
        out = call(qeq, r, cell, batch, Z, keys)
        torch.testing.assert_close(out["latent_charges"], ref["latent_charges"], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(out["E_lr"], ref["E_lr"], rtol=1e-12, atol=1e-12)
    # and the two energy definitions coincide for the flag itself
    out = call(model(periodic, vectorised, qeq_siepmann_sprik=True, qeq_atomic_numbers=[78], qeq_onsite_energy=False),
               r, cell, batch, Z, keys)
    torch.testing.assert_close(out["E_lr"], ref["E_lr"], rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# 7  electrode physics
# ---------------------------------------------------------------------------
def slab(n_xy=6, n_layers=3, a=2.5, lz=40.0):
    """Square-lattice slab centred on z = 0 (wrapped), in an orthorhombic cell."""
    xy = torch.stack(torch.meshgrid(torch.arange(n_xy) * a, torch.arange(n_xy) * a, indexing="ij"), -1).reshape(-1, 2)
    z = (torch.arange(n_layers) - (n_layers - 1) / 2) * a
    r = torch.cat([torch.cat([xy, torch.full((len(xy), 1), float(zz))], 1) for zz in z])
    cell = torch.diag(torch.tensor([n_xy * a, n_xy * a, lz])).unsqueeze(0)
    return r, cell, float(z.max())


def electrode(r_metal, cell, r_other, q_other, vectorised=True, **kw):
    m = model(True, vectorised, qeq_siepmann_sprik=True, qeq_atomic_numbers=[78], N_max=20, **kw.pop("args", {}))
    r = torch.cat([r_metal, r_other])
    Z = torch.tensor([78] * len(r_metal) + [8] * len(r_other))
    q = torch.cat([torch.zeros(len(r_metal)), q_other])
    return m, m(positions=r, cell=cell, batch=torch.zeros(len(r), dtype=torch.long), atomic_numbers=Z, latent_charges=q, **kw)


@pytest.mark.parametrize("vectorised", (True, False))
def test_metal_is_an_equipotential_and_screens_a_charge(vectorised):
    r_m, cell, z_top = slab()
    r_o = torch.tensor([[3.1, 2.2, z_top + 4.0]])
    m, out = electrode(r_m, cell, r_o, torch.tensor([1.0]), vectorised,
                       args={"remove_self_interaction": False, "qeq_charge_scope": "system"})
    q = out["latent_charges"].reshape(-1)
    # potential on the metal atoms: d E / d q, uniform and equal to -lambda
    qv = q.detach().clone().requires_grad_(True)
    plain = Les({"remove_self_interaction": False, **({"is_periodic": True, "N_max": 20} if vectorised else {})})
    r = torch.cat([r_m, r_o])
    E = plain(positions=r, cell=cell, batch=torch.zeros(len(r), dtype=torch.long), latent_charges=qv)["E_lr"].sum()
    phi = torch.autograd.grad(E, qv)[0][:len(r_m)]
    torch.testing.assert_close(phi, -out["qeq_lambda"].expand(len(r_m)), rtol=1e-9, atol=1e-9)
    # the system is neutral, so the metal carries -1; most of it on the face that sees the charge
    assert abs(float(q[:len(r_m)].sum()) + 1.0) < 1e-10
    near = r_m[:, 2] > z_top - 0.1
    far = r_m[:, 2] < -z_top + 0.1
    assert float(q[:len(r_m)][near].sum()) < -0.7 and abs(float(q[:len(r_m)][far].sum())) < 0.2


def plain_energy(r, cell, q, v):
    """Ewald energy of fixed charges plus -sum v q: the Hamiltonian the electrode charges minimise."""
    E = Les({"is_periodic": True, "N_max": 20, "remove_self_interaction": False})(
        positions=r, cell=cell, batch=torch.zeros(len(r), dtype=torch.long), latent_charges=q)["E_lr"].sum()
    return E - (v * q).sum()


def test_applied_potential_charges_the_capacitor_and_its_charge_acts_on_the_gap():
    """Empty capacitor under dPhi = E L. The electrode charges are odd in z and neutral, the metal is an
    equipotential of Coulomb plus applied potential, and a test charge in the gap feels the applied field
    plus the field of the electrode charge: the force at fixed charges, larger than q E."""
    r_m, cell, z_top = slab()
    lz = float(cell[0, 2, 2])
    field = 2.0 / lz
    n_m = len(r_m)

    def run(z_t, q_t, with_field):
        r_o = torch.tensor([[3.1, 2.2, z_t]], requires_grad=True)
        r = torch.cat([r_m, r_o])
        v = field * r[:, 2] if with_field else torch.zeros(len(r))      # the slab is centred on z = 0
        m, out = electrode(r_m, cell, r_o, torch.tensor([q_t]), applied_potential=v,
                           args={"remove_self_interaction": False, "qeq_charge_scope": "selected"})
        f = -torch.autograd.grad(out["E_lr"].sum(), r_o, retain_graph=True)[0][0, 2]
        q = out["latent_charges"].reshape(-1).detach()
        f_fixed = -torch.autograd.grad(plain_energy(r, cell, q, v), r_o)[0][0, 2]
        return q, float(f), float(f_fixed), out, r, v

    q, _, _, out, r, v = run(lz / 2, 0.0, True)
    upper, lower = r_m[:, 2] > 0.1, r_m[:, 2] < -0.1
    assert float(q[:n_m][upper].sum()) > 0.01 and abs(float(q[:n_m].sum())) < 1e-10
    torch.testing.assert_close(q[:n_m][upper].sum(), -q[:n_m][lower].sum(), rtol=1e-9, atol=1e-12)
    qv = q.clone().requires_grad_(True)
    mu = torch.autograd.grad(plain_energy(r.detach(), cell, qv, v.detach()), qv)[0][:n_m]
    torch.testing.assert_close(mu, -out["qeq_lambda"].detach().expand(n_m), rtol=1e-9, atol=1e-9)

    for z_t in (lz / 2 - 5.0, lz / 2, lz / 2 + 5.0):
        _, f_on, f_fixed, _, _, _ = run(z_t, 1.0, True)
        _, f_off, _, _, _, _ = run(z_t, 1.0, False)
        assert abs(f_on - f_fixed) < 1e-9, (f_on, f_fixed)
        assert (f_on - f_off) > 1.1 * field, ((f_on - f_off) / field)


def test_offsets_on_the_metal_solved_at_the_sites_or_at_the_nuclei():
    """Metal charges displaced outward on both faces, under an applied potential. Solved at the sites, the
    metal is an equipotential there and the capacitor charge changes. Solved at the nuclei, the charges are
    those of the model without offsets."""
    r_m, cell, z_top = slab()
    lz = float(cell[0, 2, 2])
    n_m = len(r_m)
    r_o = torch.tensor([[1.0, 1.0, lz / 2]])
    v = torch.cat([2.0 / lz * r_m[:, 2], torch.zeros(1)])
    delta = torch.zeros(n_m + 1, 3)
    delta[:n_m, 2] = 0.8 * torch.sign(r_m[:, 2]) * (r_m[:, 2].abs() > z_top - 0.1)    # surface layers, outward
    base = {"remove_self_interaction": False, "qeq_charge_scope": "selected"}

    def run(**args):
        offsets = {"latent_offsets": delta} if args.get("charge_offset_mode") else {}
        _, out = electrode(r_m, cell, r_o, torch.tensor([0.0]), applied_potential=v, args={**base, **args}, **offsets)
        return out["latent_charges"].reshape(-1).detach(), out

    q0, _ = run()
    q_sites, out = run(charge_offset_mode=True, charge_offset_max=0.0)
    q_nuclei, _ = run(charge_offset_mode=True, charge_offset_max=0.0, qeq_at_sites=False)
    upper = torch.cat([r_m[:, 2] > 0.1, torch.zeros(1, dtype=torch.bool)])
    torch.testing.assert_close(q_nuclei, q0, rtol=1e-11, atol=1e-12)
    assert abs(float(q_sites[upper].sum()) / float(q0[upper].sum()) - 1.0) > 0.02
    qv = q_sites.clone().requires_grad_(True)
    mu = torch.autograd.grad(plain_energy(torch.cat([r_m, r_o]) + delta, cell, qv, v), qv)[0][:n_m]
    torch.testing.assert_close(mu, -out["qeq_lambda"].detach().expand(n_m), rtol=1e-9, atol=1e-9)


def test_applied_potential_is_taken_at_the_sites_with_e_ext():
    """With offsets and e_ext, v at a site is v(nucleus) + e_ext . delta: same as passing that potential,
    and a charge displaced along the field gains -q E delta."""
    r_m, cell, z_top = slab()
    lz = float(cell[0, 2, 2])
    n_m = len(r_m)
    e_ext = torch.tensor([0.0, 0.0, 2.0 / lz])
    r_o = torch.tensor([[1.0, 1.0, lz / 2 - 6.0], [4.0, 7.0, lz / 2 + 5.0]])
    q_o = torch.tensor([0.7, -0.4])
    v = torch.cat([e_ext[2] * r_m[:, 2], e_ext[2] * r_o[:, 2]])
    g = torch.Generator().manual_seed(3)
    delta = 0.4 * randn(n_m + 2, 3, generator=g)
    args = {"remove_self_interaction": False, "charge_offset_mode": True, "charge_offset_max": 0.0}

    def run(potential, **kw):
        _, out = electrode(r_m, cell, r_o, q_o, applied_potential=potential, latent_offsets=delta, args=dict(args), **kw)
        return out

    with_field = run(v, e_ext=e_ext)
    by_hand = run(v + delta @ e_ext)
    at_nuclei = run(v)
    torch.testing.assert_close(with_field["E_lr"], by_hand["E_lr"], rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(with_field["latent_charges"], by_hand["latent_charges"], rtol=1e-13, atol=1e-13)
    assert float((with_field["latent_charges"] - at_nuclei["latent_charges"]).abs().max()) > 1e-4
    # without offsets e_ext changes nothing
    m, a = electrode(r_m, cell, r_o, q_o, applied_potential=v, e_ext=e_ext, args={"remove_self_interaction": False})
    _, b = electrode(r_m, cell, r_o, q_o, applied_potential=v, args={"remove_self_interaction": False})
    torch.testing.assert_close(a["E_lr"], b["E_lr"], rtol=0, atol=EXACT)


# ---------------------------------------------------------------------------
# 8  offsets projected on an axis
# ---------------------------------------------------------------------------
def test_offsets_projected_on_an_axis():
    """charge_offset_axis keeps the component of the offsets along the axis, delta = (v . n) n."""
    r, cell, batch, Z = geometry(True)
    lat = latents(r)
    v = lat["latent_offsets"]
    along = torch.zeros_like(v)
    along[:, 2] = v[:, 2]
    kw = dict(positions=r, cell=cell, batch=batch, latent_charges=lat["latent_charges"])
    a = Les({"is_periodic": True, "charge_offset_mode": True, "charge_offset_axis": [0, 0, 2]})(latent_offsets=v, **kw)
    b = Les({"is_periodic": True, "charge_offset_mode": True})(latent_offsets=along, **kw)
    assert float(a["latent_offsets"][:, :2].abs().max()) == 0.0
    torch.testing.assert_close(a["E_lr"], b["E_lr"], rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(a["latent_offsets"], b["latent_offsets"], rtol=1e-13, atol=1e-13)
    with pytest.raises(AssertionError, match="latent_offsets"):
        Les({"is_periodic": True, "charge_offset_mode": True, "charge_offset_axis": [0, 0, 1]})(latent_offsets=v[:, 2].contiguous(), **kw)


@pytest.mark.parametrize("periodic,vectorised", CASES, ids=IDS)
def test_projected_offsets_turn_with_the_system(periodic, vectorised):
    """Turned upside down (180 degrees about x), the energy is the same and the offsets point the other way along z."""
    args = {"qeq_siepmann_sprik": True, "qeq_atomic_numbers": [78], "charge_offset_mode": True, "charge_offset_axis": [0, 0, 1]}
    m = model(periodic, vectorised, **args)
    r, cell, batch, Z = geometry(periodic)
    lat = latents(r)
    flip = torch.tensor([1.0, -1.0, -1.0], dtype=r.dtype)
    kw = dict(batch=batch, atomic_numbers=Z, latent_charges=lat["latent_charges"])
    a = m(positions=r, cell=cell, latent_offsets=lat["latent_offsets"], **kw)
    b = m(positions=r * flip, cell=cell * flip, latent_offsets=lat["latent_offsets"] * flip, **kw)
    torch.testing.assert_close(a["E_lr"], b["E_lr"], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(a["latent_charges"], b["latent_charges"], rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(a["latent_offsets"] * flip, b["latent_offsets"], rtol=1e-13, atol=1e-13)
    assert float(a["latent_offsets"][:, 2].abs().max()) > 0.0


def test_offsets_on_chosen_elements():
    """charge_offset_atomic_numbers: the other atoms keep their charges at the nuclei."""
    r, cell, batch, Z = geometry(True)
    lat = latents(r)
    v = lat["latent_offsets"]
    metal = (Z == 78).to(r.device)
    only = v.clone()
    only[~metal] = 0.0
    kw = dict(positions=r, cell=cell, batch=batch, latent_charges=lat["latent_charges"])
    m = Les({"is_periodic": True, "charge_offset_mode": True, "charge_offset_atomic_numbers": [78]})
    a = m(latent_offsets=v, atomic_numbers=Z, **kw)
    b = Les({"is_periodic": True, "charge_offset_mode": True})(latent_offsets=only, **kw)
    assert float(a["latent_offsets"][~metal].abs().max()) == 0.0
    assert float(a["latent_offsets"][metal].abs().min()) > 0.0
    torch.testing.assert_close(a["E_lr"], b["E_lr"], rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(a["latent_offsets"], b["latent_offsets"], rtol=1e-13, atol=1e-13)
    with pytest.raises(ValueError, match="atomic_numbers"):
        m(latent_offsets=v, **kw)


def test_only_the_equilibrated_atoms_are_projected():
    """With Qeq the axis applies to the equilibrated atoms (the slab); the others keep their 3D offsets."""
    r, cell, batch, Z = geometry(True)
    lat = latents(r)
    v = lat["latent_offsets"]
    metal = (Z == 78).to(r.device)
    along = v.clone()
    along[metal, :2] = 0.0
    args = {"qeq_siepmann_sprik": True, "qeq_atomic_numbers": [78], "charge_offset_mode": True}
    kw = dict(positions=r, cell=cell, batch=batch, atomic_numbers=Z, latent_charges=lat["latent_charges"])
    a = model(True, True, charge_offset_axis=[0, 0, 1], **args)(latent_offsets=v, **kw)
    b = model(True, True, **args)(latent_offsets=along, **kw)
    assert float(a["latent_offsets"][metal][:, :2].abs().max()) == 0.0
    assert float(a["latent_offsets"][~metal][:, :2].abs().min()) > 0.0
    torch.testing.assert_close(a["E_lr"], b["E_lr"], rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(a["latent_offsets"], b["latent_offsets"], rtol=1e-13, atol=1e-13)


# ---------------------------------------------------------------------------
# 8b  Born effective charges with Qeq: water
# ---------------------------------------------------------------------------
def water(n=4, box=7.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    o = torch.stack(torch.meshgrid(*[torch.arange(2.0)] * 3, indexing="ij"), -1).reshape(-1, 3)[:n] * box / 2 + 1.0
    r = []
    for x in o:
        a, b = randn(3, generator=g), randn(3, generator=g)
        a = a / a.norm()
        b = b - (b @ a) * a
        b = b / b.norm()
        r += [x, x + 0.96 * a, x + 0.96 * (-0.25 * a + 0.968 * b)]      # H-O-H 104.5 degrees
    return torch.stack(r), torch.tensor([8, 1, 1] * n)


@pytest.mark.parametrize("periodic,vectorised", CASES, ids=IDS)
def test_bec_with_qeq_is_the_derivative_of_the_polarisation(periodic, vectorised):
    """The equilibrated charges move with the atoms, and the BEC contains that charge transfer:
    BEC == finite differences of P (complex P when periodic, as in BEC.compute_pol_pbc)."""
    r, Z = water()
    n = len(r)
    cell = torch.eye(3).unsqueeze(0) * (7.0 if periodic else 0.0)
    batch = torch.zeros(n, dtype=torch.long)
    m = model(periodic, vectorised, qeq_hardness=0.8)
    chi_0 = torch.where(Z == 8, 1.5, -0.5)

    def charges(rr, bec=False):
        out = m(positions=rr, cell=cell, batch=batch, atomic_numbers=Z,
                latent_chi=chi_0 + 0.2 * torch.sin(1.3 * (rr - rr.mean(0))).sum(1), compute_bec=bec)   # translation invariant
        return out["latent_charges"].reshape(-1), out["BEC"]

    def polarisation(rr):
        q, _ = charges(rr)
        q = q - q.mean()
        if periodic:
            return (q[:, None] * torch.exp(2j * math.pi * rr / 7.0)).sum(0) / (2j * math.pi)
        return (q[:, None] * rr).sum(0).to(torch.complex128)

    rg = r.clone().requires_grad_(True)
    q, bec = charges(rg, bec=True)
    assert float(q[Z == 8].mean()) < -0.1 and abs(float(q.sum())) < 1e-10
    h = 1e-5
    fd = torch.zeros(n, 3, 3, dtype=torch.complex128)
    for i in range(n):          # not under no_grad: the solve takes the potential from autograd
        for d in range(3):
            rp, rm = r.clone(), r.clone()
            rp[i, d] += h
            rm[i, d] -= h
            fd[i, :, d] = ((polarisation(rp) - polarisation(rm)) / (2 * h)).detach()
    if periodic:
        fd = 7.0 * (fd * torch.exp(-2j * math.pi * r / 7.0).unsqueeze(2))
    torch.testing.assert_close(bec.detach(), fd.real, atol=1e-7, rtol=1e-6)
    # charge transfer is in it: the BEC is not the charge times the unit tensor
    assert float((bec.detach() - q.detach()[:, None, None] * torch.eye(3)).abs().max()) > 0.05
    if not periodic:
        # acoustic sum rule of a neutral cluster
        assert float(bec.detach().sum(0).abs().max()) < 1e-9


# ---------------------------------------------------------------------------
# 9  batches, pickles, TorchScript
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("periodic,vectorised", CASES, ids=IDS)
def test_batch_equals_single_structures(periodic, vectorised):
    m = model(periodic, vectorised, qeq_atomic_numbers=[78])
    r, cell, batch, Z = geometry(periodic)
    keys = ("latent_charges", "latent_chi", "latent_hardness", "latent_dipoles")
    lat = latents(r)
    both = m(positions=r, cell=cell, batch=batch, atomic_numbers=Z, **{k: lat[k] for k in keys})
    for s in range(2):
        i = batch == s
        one = m(positions=r[i], cell=cell[s:s + 1], batch=torch.zeros(int(i.sum()), dtype=torch.long),
                atomic_numbers=Z[i], **{k: lat[k][i] for k in keys})
        torch.testing.assert_close(one["E_lr"], both["E_lr"][s:s + 1], rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(one["latent_charges"], both["latent_charges"][i], rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(one["qeq_lambda"], both["qeq_lambda"][s:s + 1], rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("periodic,vectorised", CASES, ids=IDS)
def test_cache_reuses_the_inverse_while_the_matrix_is_unchanged(periodic, vectorised):
    """qeq_cache: the inverse is kept while the equilibrated sites are the same, and the results do not change."""
    args = {"qeq_siepmann_sprik": True, "qeq_atomic_numbers": [78]}
    keys = ("latent_charges", "applied_potential")
    plain = model(periodic, vectorised, **args).eval()
    cached = model(periodic, vectorised, qeq_cache=True, **args).eval()
    r0, cell, batch, Z = geometry(periodic)

    def same(r):
        r = r.clone().requires_grad_(True)
        a, b = call(plain, r, cell, batch, Z, keys), call(cached, r, cell, batch, Z, keys)
        for k in ("E_lr", "latent_charges", "qeq_lambda"):
            torch.testing.assert_close(b[k], a[k], rtol=1e-10, atol=1e-10)
        fa, fb = (torch.autograd.grad(o["E_lr"].sum(), r)[0] for o in (a, b))
        torch.testing.assert_close(fb, fa, rtol=1e-9, atol=1e-9)

    same(r0)
    first = cached.qeq_cache_inverse
    assert first.numel() > 0 and plain.qeq_cache_inverse.numel() == 0
    r1 = r0.clone()
    r1[10] += torch.tensor([0.3, -0.2, 0.25])          # an atom that is not equilibrated: the matrix is the same
    same(r1)
    assert cached.qeq_cache_inverse is first
    r2 = r1.clone()
    r2[2] += torch.tensor([0.1, 0.15, -0.2])           # a metal atom: new matrix
    same(r2)
    assert cached.qeq_cache_inverse is not first
    same(r1)                                           # and back

    # not in training mode
    m = model(periodic, vectorised, qeq_cache=True, **args).train()
    call(m, r0, cell, batch, Z, ("latent_charges",))
    assert m.qeq_cache_inverse.numel() == 0


@pytest.mark.parametrize("periodic,vectorised", CASES, ids=IDS)
@pytest.mark.parametrize("args,keys", [
    ({"qeq_siepmann_sprik": True, "charge_offset_mode": True, "qeq_at_sites": False}, ("latent_charges", "latent_offsets")),
    ({"qeq_onsite_energy": False}, ("latent_charges",))], ids=["solved at the nuclei", "no onsite energy"])
def test_cache_when_the_energy_is_not_the_one_the_charges_minimise(periodic, vectorised, args, keys):
    """Then the first call is exact; a reused call still gives the energy, the charges and the forces on the
    atoms that are not equilibrated, but the forces on the equilibrated atoms lack the term through the matrix."""
    args = {"qeq_atomic_numbers": [78], **args}
    plain = model(periodic, vectorised, **args).eval()
    cached = model(periodic, vectorised, qeq_cache=True, **args).eval()
    assert not plain.qeq_variational
    r0, cell, batch, Z = geometry(periodic)
    metal = (Z == 78).to(r0.device)

    def forces(m):
        r = r0.clone().requires_grad_(True)
        out = call(m, r, cell, batch, Z, keys)
        return out, torch.autograd.grad(out["E_lr"].sum(), r)[0]

    a, fa = forces(plain)
    b, fb = forces(cached)                             # first call: builds the inverse
    inverse = cached.qeq_cache_inverse
    assert inverse.numel() > 0
    torch.testing.assert_close(fb, fa, rtol=1e-9, atol=1e-9)
    c, fc = forces(cached)                             # reused
    assert cached.qeq_cache_inverse is inverse
    for out in (b, c):
        for k in ("E_lr", "latent_charges", "qeq_lambda"):
            torch.testing.assert_close(out[k], a[k], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(fc[~metal], fa[~metal], rtol=1e-9, atol=1e-9)
    assert float((fc[metal] - fa[metal]).abs().max()) > 1e-6


def test_structure_without_selected_atoms():
    m = model(True, True, qeq_siepmann_sprik=True, qeq_atomic_numbers=[78])
    r, cell, batch, Z = geometry(True)
    Z = torch.where(batch == 1, torch.full_like(Z, 8), Z)          # no metal in the second structure
    lat = latents(r)
    out = m(positions=r, cell=cell, batch=batch, atomic_numbers=Z, latent_charges=lat["latent_charges"])
    plain = Les({"is_periodic": True})(positions=r, cell=cell, batch=batch, latent_charges=lat["latent_charges"])
    assert torch.equal(out["latent_charges"][batch == 1], lat["latent_charges"][batch == 1])
    torch.testing.assert_close(out["E_lr"][1], plain["E_lr"][1], rtol=1e-12, atol=1e-12)
    assert float(out["qeq_lambda"][1]) == 0.0


def test_settings():
    r, cell, batch, Z = geometry(True)
    q = latents(r)["latent_charges"]
    assert Les({}).use_qeq is False
    with pytest.raises(ValueError, match="atomic_numbers"):
        model(True, True)(positions=r, cell=cell, batch=batch, latent_charges=q)
    with pytest.raises(ValueError, match="latent_kappas"):
        model(True, True)(positions=r, cell=cell, batch=batch, atomic_numbers=Z, latent_charges=q, latent_kappas=q)
    with pytest.raises(ValueError, match="positive"):
        model(True, True, qeq_hardness=0.0)
    out = Les({"is_periodic": True})(positions=r, cell=cell, batch=batch, latent_charges=q)
    assert "qeq_lambda" not in out


def test_old_pickle_gets_qeq_off():
    m = Les({"is_periodic": True})
    buf = io.BytesIO()
    torch.save(m, buf)
    buf.seek(0)
    obj = torch.load(buf, weights_only=False)
    for flag in ("use_qeq", "charge_offset_axis", "charge_offset_z"):
        obj.__dict__.pop(flag)
    buf = io.BytesIO()
    torch.save(obj, buf)
    buf.seek(0)
    old = torch.load(buf, weights_only=False)
    assert old.use_qeq is False
    r, cell, batch, _ = geometry(True)
    q = latents(r)["latent_charges"]
    torch.testing.assert_close(old(positions=r, cell=cell, batch=batch, latent_charges=q)["E_lr"],
                               m(positions=r, cell=cell, batch=batch, latent_charges=q)["E_lr"], rtol=0, atol=EXACT)


@pytest.mark.parametrize("periodic,vectorised", CASES, ids=IDS)
def test_torchscript_equals_eager(periodic, vectorised):
    m = model(periodic, vectorised, qeq_atomic_numbers=[78], charge_offset_mode=True, qeq_chi={78: 0.2},
              charge_offset_atomic_numbers=[78, 1])
    scripted = torch.jit.script(m)
    r, cell, batch, Z = geometry(periodic)
    keys = ("latent_charges", "latent_chi", "latent_hardness", "latent_dipoles", "applied_potential", "latent_offsets")
    lat = latents(r)
    kw = dict(positions=r, cell=cell, batch=batch, atomic_numbers=Z, **{k: lat[k] for k in keys})
    a, b = m(**kw), scripted(**kw)
    for k in ("E_lr", "latent_charges", "qeq_lambda"):
        torch.testing.assert_close(b[k], a[k], rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# 10  float32 on the GPU
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("periodic,vectorised", CASES, ids=IDS)
@pytest.mark.parametrize("name", FORCE_OPTIONS)
def test_float32_on_the_gpu_agrees_with_float64(periodic, vectorised, name):
    """Energy, forces, charges and lambda in float32 on the GPU against float64, every option."""
    args, keys = FORCE_OPTIONS[name]
    m = model(periodic, vectorised, **args)
    r, cell, batch, Z = geometry(periodic)

    def run(mm, dtype, device):
        rr = r.to(device=device, dtype=dtype).requires_grad_(True)
        out = call(mm, rr, cell.to(device=device, dtype=dtype), batch.to(device), Z.to(device), keys)
        F = -torch.autograd.grad(out["E_lr"].sum(), rr)[0]
        return {"E_lr": out["E_lr"], "forces": F, "latent_charges": out["latent_charges"], "qeq_lambda": out["qeq_lambda"]}

    ref = run(m, DT, DEV)
    torch.set_default_dtype(torch.float32)
    try:
        low = run(copy.deepcopy(m).to(device="cuda", dtype=torch.float32), torch.float32, "cuda")
    finally:
        torch.set_default_dtype(DT)
    for k, v in low.items():
        assert v.dtype == torch.float32, f"{name}: {k} is {v.dtype}"
        scale = max(1.0, float(ref[k].abs().max()))
        err = float((v.detach().cpu().double() - ref[k].detach().cpu()).abs().max())
        assert err < 1e-5 * scale, f"{name}: {k} differs by {err:.1e} (scale {scale:.1e})"      # measured: below 1e-6


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
