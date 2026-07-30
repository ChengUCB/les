"""
Physics of the vectorized Ewald: eager only, so it runs in seconds.

  A  eager vectorized == loop-based reference from origin/main
  A2 eager vectorized == the in-repo LEGACY module through Les (the fallback
     users get when 'is_periodic' is omitted), also at non-unit sigma/dl
  H  convenience input shapes (1D q, [N,3] u, [N,3,3] quad, batch=None)
  I  tiny / near-degenerate systems
  F  float64 cell with float32 positions

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

    print("\n==================== SUMMARY ====================")
    if fails:
        print("FAILURES:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("All physics checks passed (energy + forces).")


if __name__ == "__main__":
    main()
