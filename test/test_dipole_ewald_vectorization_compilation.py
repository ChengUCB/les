"""
Device-aware validation harness for the vectorized, torch.compile / AOTInductor
friendly Ewald.

Runs unchanged on CPU / MPS (locally) and CUDA (cluster): just `git pull` on a
GPU box and run. For every configuration it checks, for ENERGY *and* FORCES:

  A. eager vectorized (is_periodic=True/False) == legacy loop Ewald   [physics]
  B. torch.compile(vectorized)                 == eager vectorized     [compile]
  C. AOTInductor(vectorized)                   == eager vectorized     [export]

FORCES are the strict gate. Issue ChengUCB/NequIP-LES#15 showed AOTInductor can
match the energy while breaking the gradient (forces) -- especially on GPU with
scatter/gather backward -- so we always compare F = -dE/dr, not just E.

Currently exercises the MONOPOLE path (the only vectorized term implemented).
Dipole / quadrupole / induced-charge / induced-dipole cases are added here as
they land in Ewald_vectorized (see TERMS below).

Run:
    PYTHONPATH=src python test/test_dipole_ewald_vectorization_compilation.py
"""
import sys
import torch

import les
from les import Les
# Fail fast (and show which les is loaded) so we never silently test an OLD
# installed les that lacks the vectorized Ewald -- e.g. one that nequip-les
# pulled into site-packages. On the cluster run with the develop source:
#   PYTHONPATH=src python test/test_dipole_ewald_vectorization_compilation.py
# or reinstall the develop repo editable (last, after nequip-les).
from les.module import Ewald_vectorized  # noqa: F401  -> ImportError if wrong les
print(f"[info] les from: {les.__file__}")

# Compiling a forward that calls torch.autograd.grad (force = -dE/dr) requires
# dynamo to trace through the autograd op. AOTInductor export handles this
# natively; torch.compile needs this flag. Guarded for older torch.
try:
    torch._dynamo.config.trace_autograd_ops = True
except Exception:
    pass


# ----------------------------------------------------------------------------
# device / dtype helpers
# ----------------------------------------------------------------------------
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Known gaps: reported but do NOT hard-fail the suite, so "green" always means
# "everything currently supported works". Add a tag when a phase defers something,
# remove it once fixed. (The realspace AOTInductor gap was fixed by expressing
# sqrt/division as pow(0.5)/pow(-1) in Ewald_vectorized.)
KNOWN_GAPS = set()

DEVICE = pick_device()
# MPS is float32-only; use float32 on-device for compile/AOTI, float64 on CPU
# for the tight physics-correctness comparison.
DEVICE_DTYPE = torch.float32
print(f"[info] device={DEVICE}, device_dtype={DEVICE_DTYPE}")


# ----------------------------------------------------------------------------
# inputs: three configs (cubic / triclinic / orthorhombic), batched
# ----------------------------------------------------------------------------
def make_inputs(device, dtype, periodic: bool):
    torch.manual_seed(42)
    n_per = [5, 6, 7]
    if periodic:
        box0 = torch.tensor([[10., 0, 0], [0, 10., 0], [0, 0, 10.]])
        box_tric = torch.tensor([[10., 2., 1.], [0, 9., 1.5], [0, 0, 10.]])
        box2 = torch.tensor([[8., 0, 0], [0, 12., 0], [0, 0, 9.]])
        cells = torch.stack([box0, box_tric, box2], dim=0).to(device=device, dtype=dtype)
        rs = [(torch.rand(n_i, 3, dtype=dtype).to(device) @ cells[cfg])
              for cfg, n_i in enumerate(n_per)]
    else:
        # non-periodic: zero cells so the legacy Ewald also picks the realspace
        # (det<1e-6) branch -> apples-to-apples with vectorized is_periodic=False
        cells = torch.zeros(len(n_per), 3, 3, device=device, dtype=dtype)
        rs = [torch.rand(n_i, 3, dtype=dtype, device=device) * 8.0 for n_i in n_per]
    r = torch.cat(rs, 0)
    batch = torch.cat([torch.full((n_i,), cfg, device=device, dtype=torch.long)
                       for cfg, n_i in enumerate(n_per)], 0)
    q = (torch.rand(sum(n_per), dtype=dtype, device=device) * 2 - 1)
    return r, q, cells, batch


def make_single(device, dtype, n: int, periodic: bool, seed: int = 7):
    """One structure (B=1, batch all zeros) -- the shape a LAMMPS pair_style
    call actually feeds, and the configuration of NequIP-LES issue #15."""
    torch.manual_seed(seed)
    if periodic:
        cell = torch.tensor([[10., 2., 1.], [0, 9., 1.5], [0, 0, 10.]],
                            dtype=dtype, device=device).unsqueeze(0)
        r = torch.rand(n, 3, dtype=dtype, device=device) @ cell[0]
    else:
        cell = torch.zeros(1, 3, 3, dtype=dtype, device=device)
        r = torch.rand(n, 3, dtype=dtype, device=device) * 8.0
    q = (torch.rand(n, dtype=dtype, device=device) * 2 - 1)
    batch = torch.zeros(n, dtype=torch.long, device=device)
    return r, q, cell, batch


# ----------------------------------------------------------------------------
# wrapper: E (scalar) and F = -dE/dr, so we can validate forces everywhere
# ----------------------------------------------------------------------------
class ForceWrapper(torch.nn.Module):
    def __init__(self, les):
        super().__init__()
        self.les = les

    def forward(self, positions, latent_charges, cell, batch):
        out = self.les(positions=positions, latent_charges=latent_charges,
                       cell=cell, batch=batch, compute_bec=False)
        E = out["E_lr"].sum()
        F = -torch.autograd.grad(E, positions, create_graph=False)[0]
        # detach E for the return: torch.compile refuses to return a tensor
        # whose grad_fn was already consumed by the inner autograd.grad. We only
        # need E's value (F carries the gradient information we validate).
        return E.detach(), F


def _run(model, r, q, cell, batch):
    r = r.detach().requires_grad_(True)
    return model(r, q, cell, batch)


def _close(a, b, rtol, atol):
    return torch.allclose(a, b, rtol=rtol, atol=atol)


# ----------------------------------------------------------------------------
# the three checks for a given periodicity
# ----------------------------------------------------------------------------
def check_case(is_periodic: bool, label: str):
    print(f"\n================ {label} (is_periodic={is_periodic}) ================")
    fails = []

    # --- A. physics correctness: vectorized vs legacy, CPU float64, tight -----
    r, q, cells, batch = make_inputs("cpu", torch.float64, periodic=is_periodic)
    vec = ForceWrapper(Les({"is_periodic": is_periodic})).to("cpu")
    # legacy path: omit is_periodic entirely
    leg = ForceWrapper(Les({})).to("cpu")
    Ev, Fv = _run(vec, r, q, cells, batch)
    El, Fl = _run(leg, r, q, cells, batch)
    okE = _close(Ev, El, 1e-9, 1e-9)
    okF = _close(Fv, Fl, 1e-7, 1e-8)
    print(f"[A eager  ] vec vs legacy  | dE={float((Ev-El).abs().max()):.2e} "
          f"dF={float((Fv-Fl).abs().max()):.2e} | E {'OK' if okE else 'FAIL'} F {'OK' if okF else 'FAIL'}")
    if not (okE and okF):
        fails.append(f"{label}:correctness")

    # --- device inputs (float32) for compile / AOTI --------------------------
    r, q, cells, batch = make_inputs(DEVICE, DEVICE_DTYPE, periodic=is_periodic)
    vec = ForceWrapper(Les({"is_periodic": is_periodic})).to(DEVICE)
    Ee, Fe = _run(vec, r, q, cells, batch)

    # --- B. torch.compile ----------------------------------------------------
    try:
        cvec = torch.compile(vec, fullgraph=True, dynamic=False)
        Ec, Fc = _run(cvec, r, q, cells, batch)
        okE = _close(Ec, Ee, 1e-4, 1e-4)
        okF = _close(Fc, Fe, 1e-3, 1e-3)
        print(f"[B compile ] compiled vs eager | dE={float((Ec-Ee).abs().max()):.2e} "
              f"dF={float((Fc-Fe).abs().max()):.2e} | E {'OK' if okE else 'FAIL'} F {'OK' if okF else 'FAIL'}")
        if not (okE and okF):
            fails.append(f"{label}:compile")
    except Exception as e:
        print(f"[B compile ] FAILED -> {type(e).__name__}: {str(e)[:200]}")
        fails.append(f"{label}:compile-error")

    # --- C. AOTInductor export ----------------------------------------------
    try:
        r_ex = r.detach().requires_grad_(True)
        ep = torch.export.export(vec, (r_ex, q, cells, batch))
        pkg = torch._inductor.aoti_compile_and_package(ep)
        aoti = torch._inductor.aoti_load_package(pkg)
        Ea, Fa = aoti(r_ex, q, cells, batch)
        okE = _close(Ea, Ee, 1e-4, 1e-4)
        okF = _close(Fa, Fe, 1e-3, 1e-3)
        print(f"[C aoti    ] aoti vs eager    | dE={float((Ea-Ee).abs().max()):.2e} "
              f"dF={float((Fa-Fe).abs().max()):.2e} | E {'OK' if okE else 'FAIL'} F {'OK' if okF else 'FAIL'}")
        if not (okE and okF):
            tag = f"{label}:aoti"
            (print(f"           (KNOWN-GAP: {tag})") if tag in KNOWN_GAPS else fails.append(tag))
    except Exception as e:
        tag = f"{label}:aoti"
        note = "KNOWN-GAP" if tag in KNOWN_GAPS else "FAILED"
        print(f"[C aoti    ] {note} -> {type(e).__name__}: {str(e)[:160]}")
        if tag not in KNOWN_GAPS:
            fails.append(f"{tag}-error")

    return fails


def check_dynamic(is_periodic: bool, label: str):
    """Dynamic-shape gate on a single structure (B=1), which is how LAMMPS calls
    the model: the atom count changes between calls, so a deployment artifact
    must generalize over N rather than specialize to the traced size.

    Export/compile at N1, then EXECUTE at N2 != N1 and compare to eager at N2.
    """
    print(f"\n---------------- {label} | dynamic shapes, B=1 ----------------")
    fails = []
    N1, N2 = 12, 20

    vec = ForceWrapper(Les({"is_periodic": is_periodic})).to(DEVICE)
    a1 = make_single(DEVICE, DEVICE_DTYPE, N1, is_periodic)
    a2 = make_single(DEVICE, DEVICE_DTYPE, N2, is_periodic, seed=11)
    Ee2, Fe2 = _run(vec, *a2)  # eager reference at the *second* size

    # --- D. torch.compile(dynamic=True): run at N1 then N2 (no recompile-to-wrong)
    try:
        cvec = torch.compile(vec, fullgraph=True, dynamic=True)
        _run(cvec, *a1)
        Ed, Fd = _run(cvec, *a2)
        okE, okF = _close(Ed, Ee2, 1e-4, 1e-4), _close(Fd, Fe2, 1e-3, 1e-3)
        print(f"[D dyn-comp] N{N1}->N{N2} vs eager | dE={float((Ed-Ee2).abs().max()):.2e} "
              f"dF={float((Fd-Fe2).abs().max()):.2e} | E {'OK' if okE else 'FAIL'} F {'OK' if okF else 'FAIL'}")
        if not (okE and okF):
            fails.append(f"{label}:dynamic-compile")
    except Exception as e:
        print(f"[D dyn-comp] FAILED -> {type(e).__name__}: {str(e)[:180]}")
        fails.append(f"{label}:dynamic-compile-error")

    # --- E. AOTInductor with a dynamic atom dimension -----------------------
    tag = f"{label}:dynamic-aoti"
    try:
        nat = torch.export.Dim("natoms", min=2, max=1 << 20)
        dyn = {"positions": {0: nat}, "latent_charges": {0: nat},
               "cell": None, "batch": {0: nat}}
        r1 = a1[0].detach().requires_grad_(True)
        ep = torch.export.export(vec, (r1, a1[1], a1[2], a1[3]), dynamic_shapes=dyn)
        aoti = torch._inductor.aoti_load_package(
            torch._inductor.aoti_compile_and_package(ep))
        r2 = a2[0].detach().requires_grad_(True)
        Ea, Fa = aoti(r2, a2[1], a2[2], a2[3])   # executed at N2 != traced N1
        okE, okF = _close(Ea, Ee2, 1e-4, 1e-4), _close(Fa, Fe2, 1e-3, 1e-3)
        print(f"[E dyn-aoti] N{N1}->N{N2} vs eager | dE={float((Ea-Ee2).abs().max()):.2e} "
              f"dF={float((Fa-Fe2).abs().max()):.2e} | E {'OK' if okE else 'FAIL'} F {'OK' if okF else 'FAIL'}")
        if not (okE and okF):
            (print(f"             (KNOWN-GAP: {tag})") if tag in KNOWN_GAPS
             else fails.append(tag))
    except Exception as e:
        note = "KNOWN-GAP" if tag in KNOWN_GAPS else "FAILED"
        print(f"[E dyn-aoti] {note} -> {type(e).__name__}: {str(e)[:180]}")
        if tag not in KNOWN_GAPS:
            fails.append(f"{tag}-error")

    return fails


def check_mixed_dtype(is_periodic: bool, label: str):
    """Regression: LAMMPS/ASE hand over a float64 cell alongside float32
    positions. The compute dtype must follow the positions, not the cell."""
    fails = []
    r, q, cell, batch = make_single("cpu", torch.float32, 12, is_periodic)
    vec = ForceWrapper(Les({"is_periodic": is_periodic}))
    E_ref, F_ref = _run(vec, r, q, cell, batch)
    try:
        E_mix, F_mix = _run(vec, r, q, cell.double(), batch)   # float64 cell
        okE, okF = _close(E_mix, E_ref, 1e-5, 1e-5), _close(F_mix, F_ref, 1e-4, 1e-5)
        print(f"[F dtype   ] f64 cell + f32 pos | dE={float((E_mix-E_ref).abs().max()):.2e} "
              f"dF={float((F_mix-F_ref).abs().max()):.2e} | E {'OK' if okE else 'FAIL'} F {'OK' if okF else 'FAIL'}")
        if not (okE and okF):
            fails.append(f"{label}:mixed-dtype")
    except Exception as e:
        print(f"[F dtype   ] FAILED -> {type(e).__name__}: {str(e)[:140]}")
        fails.append(f"{label}:mixed-dtype-error")
    return fails


# TERMS to extend as vectorized multipoles land:
#   monopole  -> now (latent_charges)
#   dipole    -> latent_dipoles      (Phase 2)
#   quadrupole-> latent_quads        (Phase 3)
#   induced   -> latent_kappas/alphas(Phase 4)
def main():
    all_fails = []
    all_fails += check_case(is_periodic=True, label="periodic (reciprocal)")
    all_fails += check_dynamic(is_periodic=True, label="periodic (reciprocal)")
    all_fails += check_mixed_dtype(is_periodic=True, label="periodic (reciprocal)")
    all_fails += check_case(is_periodic=False, label="non-periodic (realspace)")
    all_fails += check_dynamic(is_periodic=False, label="non-periodic (realspace)")
    all_fails += check_mixed_dtype(is_periodic=False, label="non-periodic (realspace)")

    print("\n==================== SUMMARY ====================")
    if all_fails:
        print("FAILURES:", all_fails)
        sys.exit(1)
    print("All eager / compile / AOTInductor energy+force checks passed.")


if __name__ == "__main__":
    main()
