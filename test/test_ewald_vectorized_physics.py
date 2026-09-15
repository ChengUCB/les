"""
Physics of the vectorized Ewald: eager only, so it runs in seconds.

  A  eager vectorized == loop-based reference from origin/main
  A2 eager vectorized == the in-repo LEGACY module through Les (the fallback
     users get when 'is_periodic' is omitted), also at non-unit sigma/dl
  H  convenience input shapes (1D q, [N,3] u, [N,3,3] quad, batch=None)
  I  tiny / near-degenerate systems
  F  float64 cell with float32 positions
  M  the OUTPUT dict is shape-consistent between 1D and 2D charge input, and the
     per-atom baselines added to the charges do not broadcast into [N, N]
  R  N_max can be resized on a built model: above max|a_i|/dl the answer does not
     move at all, below it the reciprocal sum is truncated

Energies AND forces are compared everywhere. Compilation is covered separately
by test_ewald_vectorized_compile.py.

Run:
    PYTHONPATH=src python test/test_ewald_vectorized_physics.py
"""
import sys

import torch

from _vec_harness import *          # noqa: F401,F403
from _vec_harness import (Les, WRAPPERS, NORM_FACTOR, REMOVE_SELF_INTERACTION,
                          RefEwald, SIGMA, DL, build, make_batched, make_single,
                          pack, _report)


def check_physics(is_periodic, terms, label, n_q=1):
    """A: vectorized == loop-based reference from main, in float64 on CPU."""
    fails = []
    if RefEwald is None:
        return fails
    r, q, u, quad, kappa, alpha, cells, batch = make_batched("cpu", torch.float64, is_periodic, n_q=n_q)
    vec = build(is_periodic, terms, "cpu")
    Ev, Fv = vec(*pack(r, q, u, quad, kappa, alpha, cells, batch, terms))

    ref = RefEwald(sigma=SIGMA, dl=DL, remove_self_interaction=True,
                   norm_factor=NORM_FACTOR)
    r_ref = r.detach().requires_grad_(True)
    outs = []
    for i in torch.unique(batch):
        m = batch == i
        uu = u[m] if "u" in terms else None
        QQ = quad[m] if "Q" in terms else None
        kk = kappa[m] if "k" in terms else None
        aa = alpha[m] if "a" in terms else None
        if is_periodic:
            res = ref.compute_potential_triclinic(r_ref[m], q[m], cells[i], u=uu,
                                                  quad=QQ, kappa=kk, alpha=aa)
        else:
            res = ref.compute_potential_realspace(r_ref[m], q[m], u=uu,
                                                  quad=QQ, kappa=kk, alpha=aa)
        outs.append(res["pot"])
    Er = torch.stack(outs, 0).sum(dim=1).sum()
    Fr = -torch.autograd.grad(Er, r_ref)[0]
    _report("A eager", (Ev, Fv), (Er.detach(), Fr), 1e-9, 1e-9, label, fails,
            note=f"vs main ref (n_q={n_q})")
    return fails


def check_vs_legacy(is_periodic, terms, label, sigma=SIGMA, dl=DL, n_q=1, rsi=True):
    """A2: the vectorized model must equal the in-repo LEGACY model through the
    same Les interface -- that is the fallback users actually get when
    'is_periodic' is omitted, and it exercises the full Les plumbing.

    Also the place to vary sigma/dl and remove_self_interaction: with sigma=1
    every sigma**k factor is 1, so a wrong exponent in the self terms is
    invisible at the default settings, and rsi=False is what turns those self
    terms on in the first place.
    """
    fails = []
    r, q, u, quad, kappa, alpha, cells, batch = make_batched("cpu", torch.float64, is_periodic, n_q=n_q)
    kw = {"sigma": sigma, "dl": dl, "remove_self_interaction": rsi}

    def run(les_args):
        m = WRAPPERS[terms](None)               # placeholder, replaced below
        m.les = Les(les_args)
        m.les.ewald.norm_factor = NORM_FACTOR
        return m(*pack(r, q, u, quad, kappa, alpha, cells, batch, terms))

    # legacy needs a non-degenerate cell to pick reciprocal space, and a zero
    # cell for real space -- make_batched already provides exactly that
    got = run({"is_periodic": is_periodic, **kw})
    ref = run(kw)                                # no is_periodic -> legacy Ewald
    _report("A2 legacy", got, ref, 1e-9, 1e-9, label, fails,
            note=f"vs legacy sigma={sigma} dl={dl}")
    return fails


def check_input_shapes(is_periodic, terms, label):
    """H: the convenience input shapes must give exactly the same answer as the
    canonical ones -- q as [N] instead of [N, 1], u as [N, 3], quad as [N, 3, 3],
    and batch=None instead of an explicit all-zeros batch. These normalization
    branches are what callers with a single latent channel actually hit."""
    fails = []
    r, q, u, quad, kappa, alpha, cell, batch = make_single("cpu", torch.float64, 12, is_periodic)
    vec = build(is_periodic, terms, "cpu")
    ref = vec(*pack(r, q, u, quad, kappa, alpha, cell, batch, terms))
    try:
        got = vec(*pack(r, q.squeeze(-1), u.squeeze(1), quad.squeeze(1),
                        kappa.squeeze(-1), alpha.squeeze(-1), cell, None, terms))
        _report("H shapes", got, ref, 1e-12, 1e-12, label, fails,
                note="1D q / [N,3] u / batch=None")
    except Exception as e:
        print(f"[H shapes ] FAILED -> {type(e).__name__}: {str(e)[:150]}")
        fails.append(f"{label}:input-shapes-error")
    return fails


def check_n_max_resize(label):
    """R: set_N_max changes cost, not physics.

    The physical cutoff is dl; N_max only sizes the integer grid that has to reach it,
    and its buffers are non-persistent, so a trained model can be moved to a larger cell
    by resizing the grid. Checked both ways round: no change above the requirement, a
    real change below it, and a replicated cell whose energy must scale exactly.
    """
    from les import Les, required_N_max, set_N_max
    fails = []
    n, L = 8, 10.0
    g = torch.Generator().manual_seed(5)
    r = torch.rand(n, 3, generator=g, dtype=torch.float64) * L
    q = torch.randn(n, generator=g, dtype=torch.float64)
    q = q - q.mean()
    cell = (torch.eye(3, dtype=torch.float64) * L).unsqueeze(0)
    batch = torch.zeros(n, dtype=torch.long)
    need = required_N_max(cell, 2.0)

    # above the requirement nothing moves, energy or forces
    ref = None
    stable = True
    for nm in (need, need + 2, need + 5, need + 15):
        les = Les({"is_periodic": True, "N_max": nm}).to(torch.float64)
        rg = r.clone().requires_grad_(True)
        E = les(positions=rg, cell=cell, latent_charges=q, batch=batch)["E_lr"].sum()
        F = -torch.autograd.grad(E, rg)[0]
        if ref is None:
            ref = (E.item(), F)
        stable = (stable and abs(E.item() - ref[0]) < 1e-12
                  and (F - ref[1]).abs().max().item() < 1e-11)
    print(f"[R stable ] {'ok  ' if stable else 'FAIL'} {label} "
          f"N_max {need}..{need + 15} give one answer")
    if not stable:
        fails.append(f"{label}:n-max-stability")

    # below it the sum really is truncated -- otherwise the gate above proves nothing
    les = Les({"is_periodic": True, "N_max": max(1, need - 3)}).to(torch.float64)
    E_low = les(positions=r, cell=cell, latent_charges=q, batch=batch)["E_lr"].sum()
    moved = abs(E_low.item() - ref[0]) > 1e-6
    print(f"[R trunc  ] {'ok  ' if moved else 'FAIL'} {label} "
          f"N_max {max(1, need - 3)} differs by {abs(E_low.item() - ref[0]):.3e}")
    if not moved:
        fails.append(f"{label}:n-max-truncation")

    # resizing a built model recovers the exact replicated energy
    nx = 3
    shift = torch.stack(torch.meshgrid(
        *[torch.arange(nx, dtype=torch.float64)] * 3, indexing="ij"),
        -1).reshape(-1, 3) * L
    rr = (r.unsqueeze(0) + shift.unsqueeze(1)).reshape(-1, 3)
    qq = q.repeat(nx ** 3)
    big = (torch.eye(3, dtype=torch.float64) * (L * nx)).unsqueeze(0)
    bb = torch.zeros(len(qq), dtype=torch.long)
    trained = Les({"is_periodic": True, "N_max": need}).to(torch.float64)
    sd_before = {k: v.clone() for k, v in trained.state_dict().items()}
    got = set_N_max(trained, cell=big)
    E_big = trained(positions=rr, cell=big, latent_charges=qq, batch=bb)["E_lr"].sum()
    rel = abs(E_big.item() - ref[0] * nx ** 3) / abs(ref[0] * nx ** 3)
    ok = rel < 1e-10 and got == required_N_max(big, 2.0)
    print(f"[R resize ] {'ok  ' if ok else 'FAIL'} {label} "
          f"{nx}x replicated cell, N_max->{got}, rel={rel:.1e}")
    if not ok:
        fails.append(f"{label}:n-max-resize")

    same = (set(sd_before) == set(trained.state_dict())
            and all(torch.equal(sd_before[k], trained.state_dict()[k])
                    for k in sd_before))
    print(f"[R weights] {'ok  ' if same else 'FAIL'} {label} state dict untouched")
    if not same:
        fails.append(f"{label}:n-max-weights")
    return fails


def check_output_shapes(is_periodic, label):
    """M: a [N] and a [N, 1] charge input must agree in the RETURNED charges too.

    Gate H compares only the energy and forces, so it cannot see a merge that
    broadcasts. Two such merges existed: the induced charge (q_induced is always
    [N, n_q], so [N] + [N, 1] gave [N, N]) and the fixed-charge baseline (which is
    [N], so [N, 1] + [N] gave [N, N] -- that one reached the Ewald and corrupted
    the energy, not just the reported charges).
    """
    fails = []
    n = 6
    r, q, u, quad, kappa, alpha, cell, batch = make_single("cpu", torch.float64, n, is_periodic)
    z = torch.arange(n) % 8 + 1
    q1 = q.squeeze(-1)                      # [N]
    common = dict(positions=r, cell=cell, batch=batch)

    # induced charge: same values and a [N] result for a [N] input
    les = Les({"is_periodic": is_periodic, "sigma": SIGMA, "dl": DL})
    o1 = les(latent_charges=q1, latent_kappas=kappa.squeeze(-1), **common)
    o2 = les(latent_charges=q, latent_kappas=kappa, **common)
    c1, c2 = o1["latent_charges"], o2["latent_charges"]
    ok = (c1.shape == q1.shape
          and torch.allclose(c1, c2.reshape(-1), rtol=1e-12, atol=1e-12))
    print(f"[M kappa  ] {'ok  ' if ok else 'FAIL'} {label} "
          f"1D->{tuple(c1.shape)} 2D->{tuple(c2.shape)}")
    if not ok:
        fails.append(f"{label}:output-shape-kappa")

    # fixed-charge baseline: the energy must not depend on how q was shaped
    lesf = Les({"is_periodic": is_periodic, "sigma": SIGMA, "dl": DL,
                "use_fixed_atomic_charges": True})
    e1 = lesf(latent_charges=q1, atomic_numbers=z, **common)["E_lr"].sum()
    e2 = lesf(latent_charges=q, atomic_numbers=z, **common)["E_lr"].sum()
    ok = torch.allclose(e1, e2, rtol=1e-12, atol=1e-12)
    print(f"[M fixedq ] {'ok  ' if ok else 'FAIL'} {label} "
          f"E_lr 1D={e1.item():.10f} 2D={e2.item():.10f}")
    if not ok:
        fails.append(f"{label}:output-shape-fixedq")

    # and it must be the value you get by adding the baseline by hand
    from les.module import FixedCharges
    hand = q1 + FixedCharges(normalization_factor=0.5)(z).to(q1.dtype)
    e3 = Les({"is_periodic": is_periodic, "sigma": SIGMA, "dl": DL})(
        latent_charges=hand, **common)["E_lr"].sum()
    ok = torch.allclose(e1, e3, rtol=1e-12, atol=1e-12)
    print(f"[M baseline] {'ok  ' if ok else 'FAIL'} {label} vs hand-added baseline")
    if not ok:
        fails.append(f"{label}:fixedq-value")
    return fails


def check_edge_cases(is_periodic, terms, label):
    """I: tiny and near-degenerate systems -- a single atom (no pairs at all),
    two atoms, and a close contact. Compared to the legacy module so we check
    correctness, not just absence of a crash.

    On the close contact: the energies agree to ~1e-15 at any separation, but the
    quadrupole kernels are differences of 1/r^4 and 1/r^5 terms, so below roughly
    0.01 A they lose most of float64's digits to cancellation and the two
    implementations (same formulas, different op order) drift apart in the
    forces -- measured max|dF| ~ 1e-5 at 0.01 A and ~1 at 0.001 A. That is a
    conditioning limit of the formulas themselves, shared with the reference, not
    a property of the vectorization; 0.1 A is already a far closer approach than
    any physical configuration.
    """
    fails = []
    for n, sep, name in ((1, None, "N=1"), (2, None, "N=2"), (4, 0.1, "close contact")):
        r, q, u, quad, kappa, alpha, cell, batch = make_single("cpu", torch.float64, n, is_periodic)
        if sep is not None:
            r = r.clone()
            r[1] = r[0] + sep          # two atoms almost coincident
        kw = {"sigma": SIGMA, "dl": DL,
              "remove_self_interaction": REMOVE_SELF_INTERACTION}

        def run(les_args):
            m = WRAPPERS[terms](None)
            m.les = Les(les_args)
            m.les.ewald.norm_factor = NORM_FACTOR
            return m(*pack(r, q, u, quad, kappa, alpha, cell, batch, terms))

        try:
            got = run({"is_periodic": is_periodic, **kw})
            ref = run(kw)
            if not torch.isfinite(got[0]).all() or not torch.isfinite(got[1]).all():
                print(f"[I edge    ] {name}: non-finite output | FAIL")
                fails.append(f"{label}:edge-{name}-nonfinite")
                continue
            _report("I edge   ", got, ref, 1e-9, 1e-9, label, fails, note=f"{name} vs legacy")
        except Exception as e:
            print(f"[I edge   ] {name} FAILED -> {type(e).__name__}: {str(e)[:120]}")
            fails.append(f"{label}:edge-{name}-error")
    return fails


def check_mixed_dtype(is_periodic, terms, label):
    """F: LAMMPS/ASE hand over a float64 cell with float32 positions; the compute
    dtype must follow the positions."""
    fails = []
    r, q, u, quad, kappa, alpha, cell, batch = make_single("cpu", torch.float32, 12, is_periodic)
    vec = build(is_periodic, terms, "cpu")
    ref = vec(*pack(r, q, u, quad, kappa, alpha, cell, batch, terms))
    try:
        got = vec(*pack(r, q, u, quad, kappa, alpha, cell.double(), batch, terms))
        _report("F dtype", got, ref, 1e-5, 1e-5, label, fails,
                note="f64 cell + f32 pos")
    except Exception as e:
        print(f"[F dtype  ] FAILED -> {type(e).__name__}: {str(e)[:150]}")
        fails.append(f"{label}:mixed-dtype-error")
    return fails


def check_torchscript(is_periodic, terms, label):
    """J: torch.jit.script the vectorized module and run it.

    Nothing else in the suite scripts this class -- the TorchScript tests build
    Les({}), which is the legacy module -- and a refactor did silently break
    scriptability once (a name defined in only one branch, which TorchScript
    rejects while eager and dynamo accept it). float64, so this is about
    scripting rather than float32 cancellation in the quadrupole kernels.
    """
    fails = []
    r, q, u, quad, kappa, alpha, cells, batch = make_batched("cpu", torch.float64, is_periodic)
    # script the Les module itself: the (E, F) wrappers call torch.autograd.grad,
    # which is not scriptable, so the forces are taken outside the scripted call
    les = build(is_periodic, terms, "cpu").les
    kw = {"latent_charges": q, "cell": cells, "batch": batch, "compute_bec": False}
    if "u" in terms:
        kw["latent_dipoles"] = u
    if "Q" in terms:
        kw["latent_quads"] = quad
    if "k" in terms:
        kw["latent_kappas"] = kappa
    if "a" in terms:
        kw["latent_alphas"] = alpha

    def run(module):
        pos = r.detach().requires_grad_(True)
        E = module(positions=pos, **kw)["E_lr"].sum()
        return E.detach(), -torch.autograd.grad(E, pos)[0]

    ref = run(les)
    try:
        got = run(torch.jit.script(les))
        _report("J script", got, ref, 1e-9, 1e-9, label, fails, note="scripted vs eager")
    except Exception as e:
        print(f"[J script ] FAILED -> {type(e).__name__}: {str(e)[:150]}")
        fails.append(f"{label}:torchscript-error")
    return fails


def main():
    fails = []
    for is_periodic, pname in ((True, "periodic"), (False, "realspace")):
        for terms in ("q", "q+u", "q+u+Q", "q+u+Q+k+a"):
            label = f"{pname} [{terms}]"
            print(f"\n================ {label} ================")
            fails += check_physics(is_periodic, terms, label, n_q=1)
            if terms != "q":
                fails += check_physics(is_periodic, terms, label, n_q=2)
            fails += check_vs_legacy(is_periodic, terms, label)
            fails += check_vs_legacy(is_periodic, terms, label, sigma=1.3, dl=1.5)
            fails += check_vs_legacy(is_periodic, terms, label, sigma=0.7, dl=2.5, n_q=2)
            # remove_self_interaction=False turns on the r-independent self terms
            fails += check_vs_legacy(is_periodic, terms, label, rsi=False)
            fails += check_vs_legacy(is_periodic, terms, label, sigma=1.3, dl=1.5,
                                     n_q=2, rsi=False)
            fails += check_input_shapes(is_periodic, terms, label)
            fails += check_edge_cases(is_periodic, terms, label)
            fails += check_mixed_dtype(is_periodic, terms, label)
            fails += check_torchscript(is_periodic, terms, label)
        # once per periodicity: shapes of the returned dict, not just E and F
        fails += check_output_shapes(is_periodic, f"{pname} [out-shapes]")
    fails += check_n_max_resize("N_max resize")

    print("\n==================== SUMMARY ====================")
    if fails:
        print("FAILURES:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("All physics checks passed (energy + forces).")


if __name__ == "__main__":
    main()
