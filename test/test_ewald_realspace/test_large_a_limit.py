"""
Tests that f_qu, f_uu, f_Qu, and f_QQ from make_kernels agree with the
successive derivatives of 1/r in the limit of large a (sigma -> 0).

As sigma -> 0:  erf(ar) -> 1,  exp(-a^2 r^2) -> 0  for any finite r > 0.
The Ewald kernels reduce to the bare Coulomb multipole tensors:

    f_qu[i,j,c]         ->  -nc * T1  = nc * r̂_c / r^2
    f_uu[i,j,c,d]       ->  +nc * T2  = nc * (3 r̂r̂ - δ) / r^3
    f_Qu[i,j,c,d,e]     ->  +nc * T3  = nc * (-15 r̂r̂r̂ + 3 [δ r̂]_sym) / r^4
    f_QQ[i,j,c,d,e,f]   ->  +nc * T4  = nc * (105 r̂r̂r̂r̂ - 15 [δ r̂r̂]_sym + 3 [δδ]_sym) / r^5

Derivatives are with respect to r_j, with r = |r_j - r_i|.
Error is measured as max |kernel - expected| / max |expected| over all
off-diagonal pairs.
"""
import torch
import pytest

from les.module import Ewald
from les.module.make_kernels import make_kernels, bare_multipole_tensors


@pytest.fixture
def r_raw():
    return torch.tensor([
        [ 0.06633400,  0.00000000,  0.00370100],
        [-0.52638300, -0.76932700, -0.02936600],
        [-0.52638300,  0.76932700, -0.02936600],
        [-0.06632200,  0.00000000,  2.38157100],
        [ 0.48094200, -0.00000000,  3.18454900],
        [ 0.57162900, -0.00000000,  1.64857000],
    ], dtype=torch.float64)


def _setup_small_sigma(sigma=0.001):
    ep = Ewald(dl=1.0, sigma=sigma, remove_self_interaction=False)
    nc = float(ep.norm_factor) / (2.0 * float(torch.pi))
    return ep.sigma, nc


def _rel_err(kernel, expected, n_atoms):
    """Max absolute error over off-diagonal pairs, normalised by max |expected|."""
    mask_off = ~torch.eye(n_atoms, dtype=torch.bool)
    extra = kernel.dim() - 2
    idx = (slice(None), slice(None)) + (None,) * extra
    m = mask_off[idx].expand_as(kernel)
    abs_err = (kernel - expected)[m].abs().max().item()
    scale   = expected[m].abs().max().item()
    return abs_err / scale


def test_f_qu_large_a_limit(r_raw):
    """f_qu -> -nc * T1  as sigma -> 0."""
    sigma, nc = _setup_small_sigma()
    r = r_raw.double()
    _, f_qu, _, _, _ = make_kernels(r, sigma, nc)
    T1, _, _, _     = bare_multipole_tensors(r)

    expected = -T1 * nc  # f_qu = nc * r̂/r^2,  T1 = -r̂/r^2
    rel = _rel_err(f_qu, expected, r.shape[0])
    print(f"f_qu large-a: rel err = {rel:.3e}")
    assert rel < 1e-3, f"f_qu large-a limit FAILED (rel err {rel:.3e})"


def test_f_uu_large_a_limit(r_raw):
    """f_uu -> +nc * T2  as sigma -> 0."""
    sigma, nc = _setup_small_sigma()
    r = r_raw.double()
    _, _, f_uu, _, _ = make_kernels(r, sigma, nc)
    _, T2, _, _     = bare_multipole_tensors(r)

    expected = T2 * nc
    rel = _rel_err(f_uu, expected, r.shape[0])
    print(f"f_uu large-a: rel err = {rel:.3e}")
    assert rel < 1e-3, f"f_uu large-a limit FAILED (rel err {rel:.3e})"


def test_f_Qu_large_a_limit(r_raw):
    """f_Qu -> +nc * T3  as sigma -> 0."""
    sigma, nc = _setup_small_sigma()
    r = r_raw.double()
    _, _, _, f_Qu, _ = make_kernels(r, sigma, nc)
    _, _, T3, _     = bare_multipole_tensors(r)

    expected = T3 * nc
    rel = _rel_err(f_Qu, expected, r.shape[0])
    print(f"f_Qu large-a: rel err = {rel:.3e}")
    assert rel < 1e-3, f"f_Qu large-a limit FAILED (rel err {rel:.3e})"


def test_f_QQ_large_a_limit(r_raw):
    """f_QQ -> +nc * T4  as sigma -> 0."""
    sigma, nc = _setup_small_sigma()
    r = r_raw.double()
    _, _, _, _, f_QQ = make_kernels(r, sigma, nc)
    _, _, _, T4     = bare_multipole_tensors(r)

    expected = T4 * nc
    rel = _rel_err(f_QQ, expected, r.shape[0])
    print(f"f_QQ large-a: rel err = {rel:.3e}")
    assert rel < 1e-3, f"f_QQ large-a limit FAILED (rel err {rel:.3e})"
