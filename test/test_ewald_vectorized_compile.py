"""
Compilation of the vectorized Ewald: torch.compile and AOTInductor.

  B torch.compile            == eager
  C AOTInductor              == eager
  D torch.compile dynamic    == eager, traced at N1 and run at N2 != N1
  E AOTInductor dynamic      == eager, traced at N1 and run at N2 != N1
  G remove_self_interaction=False under compile (the r-independent self terms
    cancel out of the forces, so only the energy catches a bad reduction there)

FORCES are the strict gate: issue ChengUCB/NequIP-LES#15 showed AOTInductor can
match the energy while breaking the gradient.

Deliberately uses small systems -- codegen correctness depends on which ops are
emitted, not on the atom count, and the kernels for the full multipole set are
expensive to build. The physics is validated at larger sizes in float64 by
test_ewald_vectorized_physics.py.

Run:
    PYTHONPATH=src python test/test_ewald_vectorized_compile.py
"""
import sys

import torch

from _vec_harness import *          # noqa: F401,F403
from _vec_harness import (AOTI_DEVICE, ARG_NAMES, DEVICE, DEVICE_DTYPE, KNOWN_GAPS,
                          _devices, _report, build, make_batched, make_single, pack)
import _vec_harness

# small systems keep the codegen tractable (see the module docstring)
_vec_harness.N_PER = [2, 3]


def _gate_once(gate, tag, fails, body):
    """Run body(AOTI_DEVICE) once, reporting any failure."""
    try:
        body(AOTI_DEVICE)
    except Exception as e:
        note = "KNOWN-GAP" if tag in KNOWN_GAPS else "FAILED"
        print(f"[{gate}] {note} -> {type(e).__name__}: {str(e)[:150]}")
        if tag not in KNOWN_GAPS:
            fails.append(f"{tag}-error")


def _gate_with_retry(gate, tag, fails, body):
    """Run body(device); if the backend fails to build the kernel, retry on the
    CPU and say so. `body` reports its own comparison via _report."""
    for dev in _devices():
        try:
            body(dev)
            return
        except Exception as e:
            if dev != _devices()[-1]:
                print(f"[{gate}] {dev.type} codegen failed "
                      f"({type(e).__name__}), retrying on cpu")
                continue
            note = "KNOWN-GAP" if tag in KNOWN_GAPS else "FAILED"
            print(f"[{gate}] {note} -> {type(e).__name__}: {str(e)[:150]}")
            if tag not in KNOWN_GAPS:
                fails.append(f"{tag}-error")


def check_compile_export(is_periodic, terms, label, n_q=1):
    """B/C: torch.compile and AOTInductor vs eager, static shapes, B=3."""
    fails = []

    def _build(dev):
        r, q, u, quad, kappa, alpha, cells, batch = make_batched(dev, DEVICE_DTYPE, is_periodic, n_q=n_q)
        vec = build(is_periodic, terms, dev)
        args = pack(r, q, u, quad, kappa, alpha, cells, batch, terms)
        return vec, args, vec(*args)

    def body_compile(dev):
        vec, args, ref = _build(dev)
        cvec = torch.compile(vec, fullgraph=True, dynamic=False)
        _report("B compile", cvec(*args), ref, 1e-4, 1e-4, label, fails,
                note=f"compiled vs eager ({dev.type})")

    def body_aoti(dev):
        vec, args, ref = _build(dev)
        ep = torch.export.export(vec, args)
        aoti = torch._inductor.aoti_load_package(
            torch._inductor.aoti_compile_and_package(ep))
        _report("C aoti", aoti(*args), ref, 1e-4, 1e-4, label, fails,
                note=f"aoti vs eager ({dev.type})")

    _gate_with_retry("B compile", f"{label}:compile", fails, body_compile)
    _gate_once("C aoti   ", f"{label}:aoti", fails, body_aoti)
    return fails


def check_dynamic(is_periodic, terms, label):
    """D/E: dynamic shapes on a single structure (B=1), as LAMMPS calls it.
    Trace/compile at N1, then EXECUTE at N2 != N1 and compare to eager at N2."""
    fails = []
    N1, N2 = 5, 8

    def _build(dev):
        vec = build(is_periodic, terms, dev)
        a1 = pack(*make_single(dev, DEVICE_DTYPE, N1, is_periodic), terms)
        a2 = pack(*make_single(dev, DEVICE_DTYPE, N2, is_periodic, seed=11), terms)
        return vec, a1, a2, vec(*a2)

    def body_compile(dev):
        vec, a1, a2, ref = _build(dev)
        cvec = torch.compile(vec, fullgraph=True, dynamic=True)
        cvec(*a1)
        _report("D dyn-cmp", cvec(*a2), ref, 1e-4, 1e-4, label, fails,
                note=f"N{N1}->N{N2} vs eager ({dev.type})")

    def body_aoti(dev):
        vec, a1, a2, ref = _build(dev)
        nat = torch.export.Dim("natoms", min=2, max=1 << 20)
        dyn = {n: ({0: nat} if n != "cell" else None) for n in ARG_NAMES[terms]}
        ep = torch.export.export(vec, a1, dynamic_shapes=dyn)
        aoti = torch._inductor.aoti_load_package(
            torch._inductor.aoti_compile_and_package(ep))
        _report("E dyn-aoti", aoti(*a2), ref, 1e-4, 1e-4, label, fails,
                note=f"N{N1}->N{N2} vs eager ({dev.type})")

    _gate_with_retry("D dyn-cmp ", f"{label}:dynamic-compile", fails, body_compile)
    _gate_once("E dyn-aoti", f"{label}:dynamic-aoti", fails, body_aoti)
    return fails


def check_no_self_removal(is_periodic, terms, label):
    """G: with remove_self_interaction=False the r-independent self terms are
    added. They cancel out of the forces, so only an energy comparison catches a
    miscompiled reduction there (this gate exists because one such inductor bug
    silently over-counted the quadrupole self term by 9x)."""
    import _vec_harness as H
    fails = []
    H.REMOVE_SELF_INTERACTION = False

    def body(dev):
        r, q, u, quad, kappa, alpha, cells, batch = make_batched(dev, DEVICE_DTYPE, is_periodic)
        vec = build(is_periodic, terms, dev)
        args = pack(r, q, u, quad, kappa, alpha, cells, batch, terms)
        ref = vec(*args)
        cvec = torch.compile(vec, fullgraph=True, dynamic=False)
        _report("G no-rsi", cvec(*args), ref, 1e-4, 1e-4, label, fails,
                note=f"compiled vs eager ({dev.type})")

    try:
        _gate_with_retry("G no-rsi ", f"{label}:no-self-removal", fails, body)
    finally:
        H.REMOVE_SELF_INTERACTION = True
    return fails


def check_latent_grads(is_periodic, label):
    """L: differentiate the latent multipoles too, in float64 on the CPU.

    Gates B-E differentiate positions only, but in a trained model every
    multipole is a network output, so the deployed graph carries their backward
    as well -- and float64 is what nequip runs by default. That combination is
    not just a shape detail: the backward of a gather over the [N, N] pair grid
    is a scatter-add, and the inductor CPU backend fails to vectorize it, which
    made every non-periodic model unexportable while all the gates above passed.
    """
    import _vec_harness as H
    fails = []
    dev = torch.device("cpu")   # this is the backend the codegen gap lives in

    def _build():
        vec = H.build(is_periodic, "all-grads", dev)
        args = H.pack_all_grads(*H.make_single(dev, torch.float64, 6, is_periodic))
        return vec, args, vec(*args)

    def body_compile(_dev):
        vec, args, ref = _build()
        cvec = torch.compile(vec, fullgraph=True, dynamic=False)
        _report("L lat-cmp", cvec(*args), ref, 1e-9, 1e-9, label, fails,
                note="compiled vs eager (f64)")

    def body_aoti(_dev):
        vec, args, ref = _build()
        nat = torch.export.Dim("natoms", min=2, max=1 << 20)
        dyn = {n: ({0: nat} if n != "cell" else None) for n in H.ARG_NAMES["all-grads"]}
        ep = torch.export.export(vec, args, dynamic_shapes=dyn)
        aoti = torch._inductor.aoti_load_package(
            torch._inductor.aoti_compile_and_package(ep))
        _report("L lat-aoti", aoti(*args), ref, 1e-9, 1e-9, label, fails,
                note="dynamic aoti vs eager (f64)")

    _gate_once("L lat-cmp ", f"{label}:latent-grad-compile", fails, body_compile)
    _gate_once("L lat-aoti", f"{label}:latent-grad-aoti", fails, body_aoti)
    return fails


# TERMS still to add: induced charges (-iq), induced dipoles (-iu) and
# epsilon_r scaling -> extend WRAPPERS/ARG_NAMES/pack and the reference call.


def main():
    fails = []
    for is_periodic, pname in ((True, "periodic"), (False, "realspace")):
        for terms in ("q", "q+u", "q+u+Q", "q+u+Q+k+a"):
            label = f"{pname} [{terms}]"
            print(f"\n================ {label} ================")
            fails += check_compile_export(is_periodic, terms, label)
            if terms != "q":
                fails += check_compile_export(is_periodic, terms, label, n_q=2)
            fails += check_dynamic(is_periodic, terms, label)
            fails += check_no_self_removal(is_periodic, terms, label)
        # once per periodicity: the full term set with every multipole
        # differentiated, which is the shape of a real model's graph
        fails += check_latent_grads(is_periodic, f"{pname} [latent-grads]")

    print("\n==================== SUMMARY ====================")
    if fails:
        print("FAILURES:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("All compile / AOTInductor / dynamic checks passed (energy + forces).")


if __name__ == "__main__":
    main()
