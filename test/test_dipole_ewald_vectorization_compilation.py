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
# module-level so a gate can flip it: the remove_self_interaction=False branch
# adds the r-independent self terms, which the default gates never reach
REMOVE_SELF_INTERACTION = True


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
    quad = torch.rand(N, n_q, 3, 3, dtype=dtype, device=device) * 0.4
    quad = 0.5 * (quad + quad.transpose(-1, -2))
    batch = torch.cat([torch.full((n,), i, device=device, dtype=torch.long)
                       for i, n in enumerate(n_per)])
    return r, q, u, quad, cells, batch


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
    quad = torch.rand(n, n_q, 3, 3, dtype=dtype, device=device) * 0.4
    quad = 0.5 * (quad + quad.transpose(-1, -2))
    batch = torch.zeros(n, dtype=torch.long, device=device)
    return r, q, u, quad, cell, batch


# ----------------------------------------------------------------------------
# wrappers returning (E, F): forces must be validated everywhere
# ----------------------------------------------------------------------------
def _les(is_periodic):
    return Les({"is_periodic": is_periodic, "sigma": SIGMA, "dl": DL,
                "remove_self_interaction": REMOVE_SELF_INTERACTION})


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


class WrapQUQuad(torch.nn.Module):
    def __init__(self, is_periodic):
        super().__init__()
        self.les = _les(is_periodic)

    def forward(self, positions, latent_charges, latent_dipoles, latent_quads, cell, batch):
        E = self.les(positions=positions, latent_charges=latent_charges,
                     latent_dipoles=latent_dipoles, latent_quads=latent_quads,
                     cell=cell, batch=batch, compute_bec=False)["E_lr"].sum()
        F = -torch.autograd.grad(E, positions)[0]
        return E.detach(), F


# term sets exercised end to end; extend as further multipoles land
WRAPPERS = {"q": WrapQ, "q+u": WrapQU, "q+u+Q": WrapQUQuad}
ARG_NAMES = {
    "q": ["positions", "latent_charges", "cell", "batch"],
    "q+u": ["positions", "latent_charges", "latent_dipoles", "cell", "batch"],
    "q+u+Q": ["positions", "latent_charges", "latent_dipoles", "latent_quads",
              "cell", "batch"],
}


def build(is_periodic, terms, device):
    w = WRAPPERS[terms](is_periodic).to(device)
    w.les.ewald.norm_factor = NORM_FACTOR      # match the reference
    return w


def pack(r, q, u, quad, cell, batch, terms):
    r = r.detach().requires_grad_(True)
    if terms == "q":
        return (r, q, cell, batch)
    if terms == "q+u":
        return (r, q, u, cell, batch)
    return (r, q, u, quad, cell, batch)


def _close(a, b, rtol, atol):
    return torch.allclose(a, b, rtol=rtol, atol=atol)


def _devices():
    """Devices a torch.compile gate may run on: the selected one, then the CPU.
    The Metal backend sometimes fails to build the largest kernels (e.g. its
    atomics header); retrying on the CPU keeps the gate covered instead of
    losing it, and the output says which backend ran."""
    if DEVICE.type == "cpu":
        return [DEVICE]
    return [DEVICE, torch.device("cpu")]


# AOTInductor device. Repeated MPS exports in one process eventually exhaust the
# Metal shader compiler and *segfault* (each export works standalone), which no
# try/except can recover from -- so on MPS the export gates run on the CPU. CUDA
# is the deployment backend and is exported natively.
AOTI_DEVICE = torch.device("cpu") if DEVICE.type == "mps" else DEVICE


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
def check_physics(is_periodic, terms, label, n_q=1):
    """A: vectorized == loop-based reference from main, in float64 on CPU."""
    fails = []
    if RefEwald is None:
        return fails
    r, q, u, quad, cells, batch = make_batched("cpu", torch.float64, is_periodic, n_q=n_q)
    vec = build(is_periodic, terms, "cpu")
    Ev, Fv = vec(*pack(r, q, u, quad, cells, batch, terms))

    ref = RefEwald(sigma=SIGMA, dl=DL, remove_self_interaction=True,
                   norm_factor=NORM_FACTOR)
    r_ref = r.detach().requires_grad_(True)
    outs = []
    for i in torch.unique(batch):
        m = batch == i
        uu = u[m] if "u" in terms else None
        QQ = quad[m] if "Q" in terms else None
        if is_periodic:
            res = ref.compute_potential_triclinic(r_ref[m], q[m], cells[i], u=uu, quad=QQ)
        else:
            res = ref.compute_potential_realspace(r_ref[m], q[m], u=uu, quad=QQ)
        outs.append(res["pot"])
    Er = torch.stack(outs, 0).sum(dim=1).sum()
    Fr = -torch.autograd.grad(Er, r_ref)[0]
    _report("A eager", (Ev, Fv), (Er.detach(), Fr), 1e-9, 1e-9, label, fails,
            note=f"vs main ref (n_q={n_q})")
    return fails


def check_vs_legacy(is_periodic, terms, label, sigma=SIGMA, dl=DL, n_q=1):
    """A2: the vectorized model must equal the in-repo LEGACY model through the
    same Les interface -- that is the fallback users actually get when
    'is_periodic' is omitted, and it exercises the full Les plumbing.

    Also the place to vary sigma/dl: with sigma=1 every sigma**k factor is 1, so
    a wrong exponent in the self terms is invisible at the default settings.
    """
    fails = []
    r, q, u, quad, cells, batch = make_batched("cpu", torch.float64, is_periodic, n_q=n_q)
    kw = {"sigma": sigma, "dl": dl, "remove_self_interaction": REMOVE_SELF_INTERACTION}

    def run(les_args):
        m = WRAPPERS[terms](None)               # placeholder, replaced below
        m.les = Les(les_args)
        m.les.ewald.norm_factor = NORM_FACTOR
        return m(*pack(r, q, u, quad, cells, batch, terms))

    # legacy needs a non-degenerate cell to pick reciprocal space, and a zero
    # cell for real space -- make_batched already provides exactly that
    got = run({"is_periodic": is_periodic, **kw})
    ref = run(kw)                                # no is_periodic -> legacy Ewald
    _report("A2 legacy", got, ref, 1e-9, 1e-9, label, fails,
            note=f"vs legacy sigma={sigma} dl={dl}")
    return fails


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


def check_input_shapes(is_periodic, terms, label):
    """H: the convenience input shapes must give exactly the same answer as the
    canonical ones -- q as [N] instead of [N, 1], u as [N, 3], quad as [N, 3, 3],
    and batch=None instead of an explicit all-zeros batch. These normalization
    branches are what callers with a single latent channel actually hit."""
    fails = []
    r, q, u, quad, cell, batch = make_single("cpu", torch.float64, 12, is_periodic)
    vec = build(is_periodic, terms, "cpu")
    ref = vec(*pack(r, q, u, quad, cell, batch, terms))
    try:
        got = vec(*pack(r, q.squeeze(-1), u.squeeze(1), quad.squeeze(1),
                        cell, None, terms))
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
        r, q, u, quad, cell, batch = make_single("cpu", torch.float64, n, is_periodic)
        if sep is not None:
            r = r.clone()
            r[1] = r[0] + sep          # two atoms almost coincident
        kw = {"sigma": SIGMA, "dl": DL,
              "remove_self_interaction": REMOVE_SELF_INTERACTION}

        def run(les_args):
            m = WRAPPERS[terms](None)
            m.les = Les(les_args)
            m.les.ewald.norm_factor = NORM_FACTOR
            return m(*pack(r, q, u, quad, cell, batch, terms))

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


def check_compile_export(is_periodic, terms, label, n_q=1):
    """B/C: torch.compile and AOTInductor vs eager, static shapes, B=3."""
    fails = []

    def _build(dev):
        r, q, u, quad, cells, batch = make_batched(dev, DEVICE_DTYPE, is_periodic, n_q=n_q)
        vec = build(is_periodic, terms, dev)
        args = pack(r, q, u, quad, cells, batch, terms)
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
    N1, N2 = 12, 20

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


def check_mixed_dtype(is_periodic, terms, label):
    """F: LAMMPS/ASE hand over a float64 cell with float32 positions; the compute
    dtype must follow the positions."""
    fails = []
    r, q, u, quad, cell, batch = make_single("cpu", torch.float32, 12, is_periodic)
    vec = build(is_periodic, terms, "cpu")
    ref = vec(*pack(r, q, u, quad, cell, batch, terms))
    try:
        got = vec(*pack(r, q, u, quad, cell.double(), batch, terms))
        _report("F dtype", got, ref, 1e-5, 1e-5, label, fails,
                note="f64 cell + f32 pos")
    except Exception as e:
        print(f"[F dtype  ] FAILED -> {type(e).__name__}: {str(e)[:150]}")
        fails.append(f"{label}:mixed-dtype-error")
    return fails


def check_no_self_removal(is_periodic, terms, label):
    """G: with remove_self_interaction=False the r-independent self terms are
    added. They cancel out of the forces, so only an energy comparison catches a
    miscompiled reduction there (this gate exists because one such inductor bug
    silently over-counted the quadrupole self term by 9x)."""
    global REMOVE_SELF_INTERACTION
    fails = []
    REMOVE_SELF_INTERACTION = False
    try:
        r, q, u, quad, cells, batch = make_batched(DEVICE, DEVICE_DTYPE, is_periodic)
        vec = build(is_periodic, terms, DEVICE)
        args = pack(r, q, u, quad, cells, batch, terms)
        ref = vec(*args)
        cvec = torch.compile(vec, fullgraph=True, dynamic=False)
        _report("G no-rsi", cvec(*args), ref, 1e-4, 1e-4, label, fails,
                note="compiled vs eager")
    except Exception as e:
        print(f"[G no-rsi ] FAILED -> {type(e).__name__}: {str(e)[:150]}")
        fails.append(f"{label}:no-self-removal-error")
    finally:
        REMOVE_SELF_INTERACTION = True
    return fails


# TERMS still to add: induced charges (-iq), induced dipoles (-iu) and
# epsilon_r scaling -> extend WRAPPERS/ARG_NAMES/pack and the reference call.
def main():
    fails = []
    for is_periodic, pname in ((True, "periodic"), (False, "realspace")):
        for terms in ("q", "q+u", "q+u+Q"):
            label = f"{pname} [{terms}]"
            print(f"\n================ {label} ================")
            fails += check_physics(is_periodic, terms, label, n_q=1)
            if terms != "q":     # multi-channel latents exercise the self-terms
                fails += check_physics(is_periodic, terms, label, n_q=2)
            # vs the in-repo legacy module, at default and at non-unit sigma/dl
            fails += check_vs_legacy(is_periodic, terms, label)
            fails += check_vs_legacy(is_periodic, terms, label, sigma=1.3, dl=1.5)
            fails += check_vs_legacy(is_periodic, terms, label, sigma=0.7, dl=2.5, n_q=2)
            fails += check_input_shapes(is_periodic, terms, label)
            fails += check_edge_cases(is_periodic, terms, label)
            fails += check_compile_export(is_periodic, terms, label)
            if terms != "q":     # multi-channel latents through the codegen path
                fails += check_compile_export(is_periodic, terms, label, n_q=2)
            fails += check_dynamic(is_periodic, terms, label)
            fails += check_mixed_dtype(is_periodic, terms, label)
            fails += check_no_self_removal(is_periodic, terms, label)

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
