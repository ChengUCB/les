"""Shared pieces of the vectorized-Ewald validation harness.

Used by test_ewald_vectorized_physics.py (fast, eager, float64) and
test_ewald_vectorized_compile.py (slow, torch.compile / AOTInductor).
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
# this harness compiles the same wrapper classes under many configurations; the
# default cache_size_limit of 8 is reached and, with fullgraph=True, dynamo then
# hard-fails instead of falling back. Raise it rather than resetting between
# gates, which would force a full recompile every time.
for _lim, _val in (("cache_size_limit", 128), ("accumulated_cache_size_limit", 4096)):
    try:
        setattr(torch._dynamo.config, _lim, _val)
    except Exception:
        pass

# Known gaps: reported but do NOT hard-fail, so "green" means "everything
# currently supported works". Remove a tag once fixed.
KNOWN_GAPS = {
    # AOTInductor export of the PERIODIC path fails ("fake tensor in the exported
    # program constant's list") once the latent multipoles are differentiated as
    # well as the positions -- some op on the reciprocal-space path reuses its own
    # output in its backward. It does not block deployment: `nequip-compile`
    # exports periodic models fine, because inference only needs dE/dr. Kept as a
    # gate so it is not forgotten, and so a fix is noticed.
    "periodic [latent-grads]:latent-grad-aoti",
}

SIGMA, DL = 1.0, 2.0
# atoms per configuration for the batched inputs; the compile test shrinks this
N_PER = [5, 6, 7]
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
    n_per = N_PER
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
    kappa = torch.rand(N, n_q, dtype=dtype, device=device) * 0.3
    alpha = torch.rand(N, n_q, dtype=dtype, device=device) * 0.3
    batch = torch.cat([torch.full((n,), i, device=device, dtype=torch.long)
                       for i, n in enumerate(n_per)])
    return r, q, u, quad, kappa, alpha, cells, batch


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
    kappa = torch.rand(n, n_q, dtype=dtype, device=device) * 0.3
    alpha = torch.rand(n, n_q, dtype=dtype, device=device) * 0.3
    batch = torch.zeros(n, dtype=torch.long, device=device)
    return r, q, u, quad, kappa, alpha, cell, batch


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


class WrapFull(torch.nn.Module):
    """every term the vectorized module supports, including the induced ones"""
    def __init__(self, is_periodic):
        super().__init__()
        self.les = _les(is_periodic)

    def forward(self, positions, latent_charges, latent_dipoles, latent_quads,
                latent_kappas, latent_alphas, cell, batch):
        E = self.les(positions=positions, latent_charges=latent_charges,
                     latent_dipoles=latent_dipoles, latent_quads=latent_quads,
                     latent_kappas=latent_kappas, latent_alphas=latent_alphas,
                     cell=cell, batch=batch, compute_bec=False)["E_lr"].sum()
        F = -torch.autograd.grad(E, positions)[0]
        return E.detach(), F


class WrapLatentGrad(torch.nn.Module):
    """Differentiates with respect to the latent multipoles, not just positions.

    In a trained model every multipole is a network output, so the deployed graph
    also contains their backward. The wrappers above differentiate positions
    only, and that blind spot hid a real failure: the backward of the [N, N]
    gather in the real-space pair grid is a scatter-add that the inductor CPU
    backend could not vectorize, so no non-periodic model could be exported with
    AOTInductor even though every gate here passed.

    All gradients are returned as one flat vector so the shared comparison helper
    can check them in the slot it uses for forces.
    """
    def __init__(self, is_periodic):
        super().__init__()
        self.les = _les(is_periodic)

    def forward(self, positions, latent_charges, latent_dipoles, latent_quads,
                latent_kappas, latent_alphas, cell, batch):
        E = self.les(positions=positions, latent_charges=latent_charges,
                     latent_dipoles=latent_dipoles, latent_quads=latent_quads,
                     latent_kappas=latent_kappas, latent_alphas=latent_alphas,
                     cell=cell, batch=batch, compute_bec=False)["E_lr"].sum()
        grads = torch.autograd.grad(
            E, [positions, latent_charges, latent_dipoles, latent_quads,
                latent_kappas, latent_alphas])
        return E.detach(), torch.cat([g.reshape(-1) for g in grads])


# term sets exercised end to end; extend as further multipoles land
WRAPPERS = {"q": WrapQ, "q+u": WrapQU, "q+u+Q": WrapQUQuad,
            "q+u+Q+k+a": WrapFull, "all-grads": WrapLatentGrad}
ARG_NAMES = {
    "q": ["positions", "latent_charges", "cell", "batch"],
    "q+u": ["positions", "latent_charges", "latent_dipoles", "cell", "batch"],
    "q+u+Q": ["positions", "latent_charges", "latent_dipoles", "latent_quads",
              "cell", "batch"],
    "q+u+Q+k+a": ["positions", "latent_charges", "latent_dipoles", "latent_quads",
                  "latent_kappas", "latent_alphas", "cell", "batch"],
    "all-grads": ["positions", "latent_charges", "latent_dipoles", "latent_quads",
                  "latent_kappas", "latent_alphas", "cell", "batch"],
}


def build(is_periodic, terms, device):
    w = WRAPPERS[terms](is_periodic).to(device)
    w.les.ewald.norm_factor = NORM_FACTOR      # match the reference
    return w


def pack(r, q, u, quad, kappa, alpha, cell, batch, terms):
    r = r.detach().requires_grad_(True)
    if terms == "q":
        return (r, q, cell, batch)
    if terms == "q+u":
        return (r, q, u, cell, batch)
    if terms == "q+u+Q":
        return (r, q, u, quad, cell, batch)
    return (r, q, u, quad, kappa, alpha, cell, batch)


def pack_all_grads(r, q, u, quad, kappa, alpha, cell, batch):
    """Args for WrapLatentGrad: every multipole is a differentiable input."""
    return (r.detach().requires_grad_(True),
            q.detach().requires_grad_(True),
            u.detach().requires_grad_(True),
            quad.detach().requires_grad_(True),
            kappa.detach().requires_grad_(True),
            alpha.detach().requires_grad_(True),
            cell, batch)


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
