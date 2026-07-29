"""
Tests that each Ewald realspace kernel is the autograd derivative of the previous one.

All derivatives are with respect to r_raw[j] (where r_ij = r_raw[j] - r_raw[i]):

    d(f_qq[i,j])       / d(r_raw[j,c])   = -f_qu[i,j,c]
    d(f_qu[i,j,c])     / d(r_raw[j,d])   = -f_uu[i,j,c,d]
    d(f_uu[i,j,c,d])   / d(r_raw[j,e])   = -f_Qu[i,j,c,d,e]
    d(f_Qu[i,j,a,b,c]) / d(r_raw[j,d])   = -f_QQ[i,j,a,b,c,d]

Each kernel is defined as f_n = (-d/dr_j)^n phi, so every step introduces a minus sign.
"""
import torch
import pytest

from les.module import Ewald
from les.module.make_kernels import make_kernels


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


def _setup(sigma=2.0):
    ep = Ewald(dl=1.0, sigma=sigma, remove_self_interaction=False)
    nc = float(ep.norm_factor) / (2.0 * float(torch.pi))
    return ep.sigma, nc


# ─────────────────────────────────────────────────────────────────────────────
# Test 0: d(f_qq[i,j]) / d(r_raw[j,c]) = -f_qu[i,j,c]
# ─────────────────────────────────────────────────────────────────────────────
def test_f_qu_is_autograd_derivative_of_f_qq(r_raw):
    sigma, nc = _setup()
    kw = dict(compute_u=True, compute_Q=False)

    _, f_qu_analytical, _, _, _ = make_kernels(r_raw, sigma, nc, **kw)

    r_grad = r_raw.clone().requires_grad_(True)
    jac = torch.autograd.functional.jacobian(
        lambda r: make_kernels(r, sigma, nc, **kw)[0], r_grad)  # [n,n, n,3]

    n = r_raw.shape[0]
    max_err_j = max_err_i = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            max_err_j = max(max_err_j, (jac[i,j,j,:] + f_qu_analytical[i,j]).abs().max().item())
            max_err_i = max(max_err_i, (jac[i,j,i,:] - f_qu_analytical[i,j]).abs().max().item())

    print(f"f_qu: max |d(f_qq)/d(r_j) + f_qu| = {max_err_j:.3e}")
    print(f"f_qu: max |d(f_qq)/d(r_i) - f_qu| = {max_err_i:.3e}")
    assert max_err_j < 1e-8, f"d(f_qq[i,j])/d(r[j,c]) != -f_qu[i,j,c]  (max err {max_err_j:.3e})"
    assert max_err_i < 1e-8, f"d(f_qq[i,j])/d(r[i,c]) != +f_qu[i,j,c]  (max err {max_err_i:.3e})"


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: d(f_qu[i,j,c]) / d(r_raw[j,d]) = -f_uu[i,j,c,d]
# ─────────────────────────────────────────────────────────────────────────────
def test_f_uu_is_autograd_derivative_of_f_qu(r_raw):
    sigma, nc = _setup()
    kw = dict(compute_u=True, compute_Q=False)

    _, _, f_uu_analytical, _, _ = make_kernels(r_raw, sigma, nc, **kw)

    r_grad = r_raw.clone().requires_grad_(True)
    jac = torch.autograd.functional.jacobian(
        lambda r: make_kernels(r, sigma, nc, **kw)[1], r_grad)  # [n,n,3, n,3]

    n = r_raw.shape[0]
    max_err_j = max_err_i = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            max_err_j = max(max_err_j, (jac[i,j,:,j,:] + f_uu_analytical[i,j]).abs().max().item())
            max_err_i = max(max_err_i, (jac[i,j,:,i,:] - f_uu_analytical[i,j]).abs().max().item())

    print(f"f_uu: max |d(f_qu)/d(r_j) + f_uu| = {max_err_j:.3e}")
    print(f"f_uu: max |d(f_qu)/d(r_i) - f_uu| = {max_err_i:.3e}")
    assert max_err_j < 1e-8, f"d(f_qu[i,j,c])/d(r[j,d]) != -f_uu[i,j,c,d]  (max err {max_err_j:.3e})"
    assert max_err_i < 1e-8, f"d(f_qu[i,j,c])/d(r[i,d]) != +f_uu[i,j,c,d]  (max err {max_err_i:.3e})"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: d(f_uu[i,j,c,d]) / d(r_raw[j,e]) = -f_Qu[i,j,c,d,e]
# ─────────────────────────────────────────────────────────────────────────────
def test_f_Qu_is_autograd_derivative_of_f_uu(r_raw):
    sigma, nc = _setup()
    kw = dict(compute_u=False, compute_Q=True)

    _, _, _, f_Qu_analytical, _ = make_kernels(r_raw, sigma, nc, **kw)

    r_grad = r_raw.clone().requires_grad_(True)
    jac = torch.autograd.functional.jacobian(
        lambda r: make_kernels(r, sigma, nc, **kw)[2], r_grad)  # [n,n,3,3, n,3]

    n = r_raw.shape[0]
    max_err_j = max_err_i = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            max_err_j = max(max_err_j, (jac[i,j,:,:,j,:] + f_Qu_analytical[i,j]).abs().max().item())
            max_err_i = max(max_err_i, (jac[i,j,:,:,i,:] - f_Qu_analytical[i,j]).abs().max().item())

    print(f"f_Qu: max |d(f_uu)/d(r_j) + f_Qu| = {max_err_j:.3e}")
    print(f"f_Qu: max |d(f_uu)/d(r_i) - f_Qu| = {max_err_i:.3e}")
    assert max_err_j < 1e-8, f"d(f_uu[i,j,c,d])/d(r[j,e]) != -f_Qu[i,j,c,d,e]  (max err {max_err_j:.3e})"
    assert max_err_i < 1e-8, f"d(f_uu[i,j,c,d])/d(r[i,e]) != +f_Qu[i,j,c,d,e]  (max err {max_err_i:.3e})"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: d(f_Qu[i,j,a,b,c]) / d(r_raw[j,d]) = -f_QQ[i,j,a,b,c,d]
# ─────────────────────────────────────────────────────────────────────────────
def test_f_QQ_is_autograd_derivative_of_f_Qu(r_raw):
    sigma, nc = _setup()
    kw = dict(compute_u=False, compute_Q=True)

    _, _, _, _, f_QQ_analytical = make_kernels(r_raw, sigma, nc, **kw)

    r_grad = r_raw.clone().requires_grad_(True)
    jac = torch.autograd.functional.jacobian(
        lambda r: make_kernels(r, sigma, nc, **kw)[3], r_grad)  # [n,n,3,3,3, n,3]

    n = r_raw.shape[0]
    max_err_j = max_err_i = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            max_err_j = max(max_err_j, (jac[i,j,:,:,:,j,:] + f_QQ_analytical[i,j]).abs().max().item())
            max_err_i = max(max_err_i, (jac[i,j,:,:,:,i,:] - f_QQ_analytical[i,j]).abs().max().item())

    print(f"f_QQ: max |d(f_Qu)/d(r_j) + f_QQ| = {max_err_j:.3e}")
    print(f"f_QQ: max |d(f_Qu)/d(r_i) - f_QQ| = {max_err_i:.3e}")
    assert max_err_j < 1e-8, f"d(f_Qu[i,j,a,b,c])/d(r[j,d]) != -f_QQ[i,j,a,b,c,d]  (max err {max_err_j:.3e})"
    assert max_err_i < 1e-8, f"d(f_Qu[i,j,a,b,c])/d(r[i,d]) != +f_QQ[i,j,a,b,c,d]  (max err {max_err_i:.3e})"
