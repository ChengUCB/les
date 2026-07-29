"""
Device-aware validation harness for the vectorized, torch.compile / AOTInductor
friendly Ewald (monopoles + dipoles).

Runs unchanged on CPU / MPS (locally) and CUDA (cluster): just `git pull` on a
GPU box and run. For every configuration it checks ENERGY *and* FORCES:

  A. eager vectorized      == loop-based reference on `main`   [physics]
  B. torch.compile         == eager vectorized                 [compile]
  C. AOTInductor           == eager vectorized                 [export]
  D. torch.compile dynamic == eager, traced at N1 run at N2     [dynamic]
  E. AOTInductor dynamic   == eager, traced at N1 run at N2     [dynamic export]
  F. float64 cell + float32 positions == matched dtype          [dtype]

FORCES are the strict gate. Issue ChengUCB/NequIP-LES#15 showed AOTInductor can
match the energy while breaking the gradient, so we always compare F = -dE/dr.

Each case runs with monopoles only and with monopoles+dipoles.

Run:
    PYTHONPATH=src python test/test_dipole_ewald_vectorization_compilation.py
"""
import importlib.util
import os
import subprocess
import sys
import tempfile

import torch

import les
from les import Les
# Fail fast (and show which les is loaded) so we never silently test an OLD
# installed les that lacks the vectorized Ewald -- e.g. one that nequip-les
# pulled into site-packages.
from les.module import Ewald_vectorized
print(f"[info] les from: {les.__file__}")

# Compiling a forward that calls torch.autograd.grad (force = -dE/dr) requires
# dynamo to trace through the autograd op. AOTInductor handles it natively.
try:
    torch._dynamo.config.trace_autograd_ops = True
except Exception:
    pass

# Known gaps: reported but do NOT hard-fail, so "green" means "everything
# currently supported works". Remove a tag once fixed.
KNOWN_GAPS = set()

SIGMA, DL = 1.0, 2.0
NORM_FACTOR = 90.4756   # unified across implementations for the physics gate


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device()
DEVICE_DTYPE = torch.float32   # MPS is float32-only; CPU physics gate uses float64
print(f"[info] device={DEVICE}, device_dtype={DEVICE_DTYPE}")


# ----------------------------------------------------------------------------
# reference: the loop-based full-multipole Ewald from origin/main
# ----------------------------------------------------------------------------
def load_main_reference():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = tempfile.mkdtemp(prefix="les_ref_")
    for fn in ("make_kernels.py", "ewald.py"):
        src = subprocess.run(["git", "show", f"origin/main:src/les/module/{fn}"],
                             cwd=repo, capture_output=True, text=True, check=True).stdout
        src = src.replace("from les.module.make_kernels import make_kernels",
                          "from make_kernels import make_kernels")
        open(os.path.join(d, fn), "w").write(src)
    sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location("ref_ewald", os.path.join(d, "ewald.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Ewald


try:
    RefEwald = load_main_reference()
    print("[info] physics reference: loop-based Ewald from origin/main")
except Exception as e:
    RefEwald = None
    print(f"[warn] could not load origin/main reference ({type(e).__name__}); "
          f"physics gate will be skipped")


# ----------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------
def make_batched(device, dtype, periodic, n_q=1, seed=42):
    torch.manual_seed(seed)
    n_per = [5, 6, 7]
    if periodic:
        cells = torch.stack([
            torch.tensor([[10., 0, 0], [0, 10., 0], [0, 0, 10.]]),
            torch.tensor([[10., 2., 1.], [0, 9., 1.5], [0, 0, 10.]]),
            torch.tensor([[8., 0, 0], [0, 12., 0], [0, 0, 9.]]),
        ], 0).to(device=device, dtype=dtype)
        rs = [torch.rand(n, 3, dtype=dtype).to(device) @ cells[i]
              for i, n in enumerate(n_per)]
    else:
        # zero cells so the reference also takes its realspace (det<1e-6) branch
        cells = torch.zeros(len(n_per), 3, 3, device=device, dtype=dtype)
        rs = [torch.rand(n, 3, dtype=dtype, device=device) * 8.0 for n in n_per]
    r = torch.cat(rs, 0)
    N = sum(n_per)
    q = torch.rand(N, n_q, dtype=dtype, device=device) * 2 - 1
    u = torch.rand(N, n_q, 3, dtype=dtype, device=device) * 0.5
    batch = torch.cat([torch.full((n,), i, device=device, dtype=torch.long)
                       for i, n in enumerate(n_per)])
    return r, q, u, cells, batch


def make_single(device, dtype, n, periodic, n_q=1, seed=7):
    """One structure (B=1, batch all zeros) -- what a LAMMPS pair_style feeds."""
    torch.manual_seed(seed)
    if periodic:
        cell = torch.tensor([[10., 2., 1.], [0, 9., 1.5], [0, 0, 10.]],
                            dtype=dtype, device=device).unsqueeze(0)
        r = torch.rand(n, 3, dtype=dtype, device=device) @ cell[0]
    else:
        cell = torch.zeros(1, 3, 3, dtype=dtype, device=device)
        r = torch.rand(n, 3, dtype=dtype, device=device) * 8.0
    q = torch.rand(n, n_q, dtype=dtype, device=device) * 2 - 1
    u = torch.rand(n, n_q, 3, dtype=dtype, device=device) * 0.5
    batch = torch.zeros(n, dtype=torch.long, device=device)
    return r, q, u, cell, batch


# ----------------------------------------------------------------------------
# wrappers returning (E, F): forces must be validated everywhere
# ----------------------------------------------------------------------------
def _les(is_periodic):
    return Les({"is_periodic": is_periodic, "sigma": SIGMA, "dl": DL})


class WrapQ(torch.nn.Module):
    def __init__(self, is_periodic):
        super().__init__()
        self.les = _les(is_periodic)

    def forward(self, positions, latent_charges, cell, batch):
        E = self.les(positions=positions, latent_charges=latent_charges,
                     cell=cell, batch=batch, compute_bec=False)["E_lr"].sum()
        F = -torch.autograd.grad(E, positions)[0]
        # detach E: torch.compile refuses to return a tensor whose grad_fn was
        # consumed by the inner autograd.grad.
        return E.detach(), F


class WrapQU(torch.nn.Module):
    def __init__(self, is_periodic):
        super().__init__()
        self.les = _les(is_periodic)

    def forward(self, positions, latent_charges, latent_dipoles, cell, batch):
        E = self.les(positions=positions, latent_charges=latent_charges,
                     latent_dipoles=latent_dipoles, cell=cell, batch=batch,
                     compute_bec=False)["E_lr"].sum()
        F = -torch.autograd.grad(E, positions)[0]
        return E.detach(), F


def build(is_periodic, with_dipole, device):
    w = (WrapQU if with_dipole else WrapQ)(is_periodic).to(device)
    w.les.ewald.norm_factor = NORM_FACTOR      # match the reference
    return w


def pack(r, q, u, cell, batch, with_dipole):
    r = r.detach().requires_grad_(True)
    return (r, q, u, cell, batch) if with_dipole else (r, q, cell, batch)


def _close(a, b, rtol, atol):
    return torch.allclose(a, b, rtol=rtol, atol=atol)


def _report(tag, got, ref, rtol, atol, label, fails, note=""):
    (Eg, Fg), (Er, Fr) = got, ref
    okE, okF = _close(Eg, Er, rtol, atol), _close(Fg, Fr, rtol * 10, atol * 10)
    print(f"[{tag:9s}] {note:22s} dE={float((Eg-Er).abs().max()):.2e} "
          f"dF={float((Fg-Fr).abs().max()):.2e} | E {'OK' if okE else 'FAIL'} "
          f"F {'OK' if okF else 'FAIL'}")
    if not (okE and okF):
        fails.append(f"{label}:{tag.strip()}")


# ----------------------------------------------------------------------------
# gates
# ----------------------------------------------------------------------------
def check_physics(is_periodic, with_dipole, label, n_q=1):
    """A: vectorized == loop-based reference from main, in float64 on CPU."""
    fails = []
    if RefEwald is None:
        return fails
    r, q, u, cells, batch = make_batched("cpu", torch.float64, is_periodic, n_q=n_q)
    vec = build(is_periodic, with_dipole, "cpu")
    Ev, Fv = vec(*pack(r, q, u, cells, batch, with_dipole))

    ref = RefEwald(sigma=SIGMA, dl=DL, remove_self_interaction=True,
                   norm_factor=NORM_FACTOR)
    r_ref = r.detach().requires_grad_(True)
    outs = []
    for i in torch.unique(batch):
        m = batch == i
        uu = u[m] if with_dipole else None
        if is_periodic:
            res = ref.compute_potential_triclinic(r_ref[m], q[m], cells[i], u=uu)
        else:
            res = ref.compute_potential_realspace(r_ref[m], q[m], u=uu)
        outs.append(res["pot"])
    Er = torch.stack(outs, 0).sum(dim=1).sum()
    Fr = -torch.autograd.grad(Er, r_ref)[0]
    _report("A eager", (Ev, Fv), (Er.detach(), Fr), 1e-9, 1e-9, label, fails,
            note=f"vs main ref (n_q={n_q})")
    return fails


def check_compile_export(is_periodic, with_dipole, label):
    """B/C: torch.compile and AOTInductor vs eager, static shapes, B=3."""
    fails = []
    r, q, u, cells, batch = make_batched(DEVICE, DEVICE_DTYPE, is_periodic)
    vec = build(is_periodic, with_dipole, DEVICE)
    args = pack(r, q, u, cells, batch, with_dipole)
    ref = vec(*args)

    try:
        cvec = torch.compile(vec, fullgraph=True, dynamic=False)
        _report("B compile", cvec(*args), ref, 1e-4, 1e-4, label, fails,
                note="compiled vs eager")
    except Exception as e:
        print(f"[B compile] FAILED -> {type(e).__name__}: {str(e)[:150]}")
        fails.append(f"{label}:compile-error")

    tag = f"{label}:aoti"
    try:
        ep = torch.export.export(vec, args)
        aoti = torch._inductor.aoti_load_package(
            torch._inductor.aoti_compile_and_package(ep))
        _report("C aoti", aoti(*args), ref, 1e-4, 1e-4, label, fails,
                note="aoti vs eager")
    except Exception as e:
        note = "KNOWN-GAP" if tag in KNOWN_GAPS else "FAILED"
        print(f"[C aoti   ] {note} -> {type(e).__name__}: {str(e)[:150]}")
        if tag not in KNOWN_GAPS:
            fails.append(f"{tag}-error")
    return fails


def check_dynamic(is_periodic, with_dipole, label):
    """D/E: dynamic shapes on a single structure (B=1), as LAMMPS calls it.
    Trace/compile at N1, then EXECUTE at N2 != N1 and compare to eager at N2."""
    fails = []
    N1, N2 = 12, 20
    vec = build(is_periodic, with_dipole, DEVICE)
    a1 = pack(*make_single(DEVICE, DEVICE_DTYPE, N1, is_periodic), with_dipole)
    a2 = pack(*make_single(DEVICE, DEVICE_DTYPE, N2, is_periodic, seed=11), with_dipole)
    ref = vec(*a2)

    try:
        cvec = torch.compile(vec, fullgraph=True, dynamic=True)
        cvec(*a1)
        _report("D dyn-cmp", cvec(*a2), ref, 1e-4, 1e-4, label, fails,
                note=f"N{N1}->N{N2} vs eager")
    except Exception as e:
        print(f"[D dyn-cmp] FAILED -> {type(e).__name__}: {str(e)[:150]}")
        fails.append(f"{label}:dynamic-compile-error")

    tag = f"{label}:dynamic-aoti"
    try:
        nat = torch.export.Dim("natoms", min=2, max=1 << 20)
        names = (["positions", "latent_charges", "latent_dipoles", "cell", "batch"]
                 if with_dipole else ["positions", "latent_charges", "cell", "batch"])
        dyn = {n: ({0: nat} if n != "cell" else None) for n in names}
        ep = torch.export.export(vec, a1, dynamic_shapes=dyn)
        aoti = torch._inductor.aoti_load_package(
            torch._inductor.aoti_compile_and_package(ep))
        _report("E dyn-aoti", aoti(*a2), ref, 1e-4, 1e-4, label, fails,
                note=f"N{N1}->N{N2} vs eager")
    except Exception as e:
        note = "KNOWN-GAP" if tag in KNOWN_GAPS else "FAILED"
        print(f"[E dyn-aoti] {note} -> {type(e).__name__}: {str(e)[:150]}")
        if tag not in KNOWN_GAPS:
            fails.append(f"{tag}-error")
    return fails


def check_mixed_dtype(is_periodic, with_dipole, label):
    """F: LAMMPS/ASE hand over a float64 cell with float32 positions; the compute
    dtype must follow the positions."""
    fails = []
    r, q, u, cell, batch = make_single("cpu", torch.float32, 12, is_periodic)
    vec = build(is_periodic, with_dipole, "cpu")
    ref = vec(*pack(r, q, u, cell, batch, with_dipole))
    try:
        got = vec(*pack(r, q, u, cell.double(), batch, with_dipole))
        _report("F dtype", got, ref, 1e-5, 1e-5, label, fails,
                note="f64 cell + f32 pos")
    except Exception as e:
        print(f"[F dtype  ] FAILED -> {type(e).__name__}: {str(e)[:150]}")
        fails.append(f"{label}:mixed-dtype-error")
    return fails


# TERMS still to add: quadrupoles (-Q), induced charges (-iq), induced dipoles
# (-iu) and epsilon_r scaling -> extend build()/pack() and the reference call.
def main():
    fails = []
    for is_periodic, pname in ((True, "periodic"), (False, "realspace")):
        for with_dipole in (False, True):
            terms = "q+u" if with_dipole else "q"
            label = f"{pname} [{terms}]"
            print(f"\n================ {label} ================")
            fails += check_physics(is_periodic, with_dipole, label, n_q=1)
            if with_dipole:      # multi-channel latents exercise the self-term
                fails += check_physics(is_periodic, with_dipole, label, n_q=2)
            fails += check_compile_export(is_periodic, with_dipole, label)
            fails += check_dynamic(is_periodic, with_dipole, label)
            fails += check_mixed_dtype(is_periodic, with_dipole, label)

    print("\n==================== SUMMARY ====================")
    if fails:
        print("FAILURES:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("All physics / compile / AOTInductor / dynamic / dtype checks passed "
          "(energy + forces).")


if __name__ == "__main__":
    main()
