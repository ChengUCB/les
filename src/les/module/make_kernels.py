import torch

def make_kernels(r, sigma, norm_const, compute_u=True, compute_Q=True):
    """
    Compute the Ewald real-space interaction kernels.

      f_qq [n,n]            charge-charge            (always computed)
      f_qu [n,n,3]          charge-dipole            (if compute_u)
      f_uu [n,n,3,3]        dipole-dipole            (if compute_u or compute_Q)
      f_Qu [n,n,3,3,3]      quadrupole-dipole        (if compute_Q)
      f_QQ [n,n,3,3,3,3]    quadrupole-quadrupole    (if compute_Q)

    Unrequested kernels are returned as None to save compute.
    f_uu is returned under either flag because the quadrupole energy
    pot_Qq contracts with f_uu.
    """
    n = r.shape[0]
    a = 1.0 / (sigma * 2.0 ** 0.5)
    sqrt_pi = torch.pi ** 0.5
    device, dtype = r.device, r.dtype

    mask_off = ~torch.eye(n, dtype=torch.bool, device=device)

    r_ij = r.unsqueeze(0) - r.unsqueeze(1)   # r_ij[i,j] = r[j] - r[i]
    r_ij_norm = torch.norm(r_ij, dim=-1)

    rinv = torch.zeros_like(r_ij_norm)
    rinv[mask_off] = 1.0 / r_ij_norm[mask_off]

    erf = torch.zeros_like(r_ij_norm)
    erf[mask_off] = torch.special.erf(r_ij_norm[mask_off] * a)

    f_qq = erf * rinv * norm_const                                                  # [n,n]

    f_qu = None
    f_uu = None
    f_Qu = None
    f_QQ = None

    if not (compute_u or compute_Q):
        return f_qq, f_qu, f_uu, f_Qu, f_QQ

    rinv2 = rinv * rinv
    rinv3 = rinv2 * rinv
    gauss = torch.exp(-(a * r_ij_norm) ** 2) * mask_off
    rhat = r_ij * rinv[..., None]
    eye = torch.eye(3, device=device, dtype=dtype)

    # s1, s2 feed f_qu, f_uu (and f_Qu, f_QQ via s2)
    s1 = erf * rinv3 - (2.0 * a / sqrt_pi) * gauss * rinv2
    s2 = (3.0 * erf * rinv3
          - (6.0 * a / sqrt_pi) * gauss * rinv2
          - (4.0 * a ** 3 / sqrt_pi) * gauss)

    rr = rhat[..., :, None] * rhat[..., None, :]
    f_uu = (s2[:, :, None, None] * rr - s1[:, :, None, None] * eye[None, None]) * norm_const  # [n,n,3,3]

    if compute_u:
        f_qu = s1[..., None] * r_ij * norm_const                                    # [n,n,3]

    if compute_Q:
        rinv4 = rinv3 * rinv
        rinv5 = rinv4 * rinv
        # s3 = 2*s2/r - ds2/dr  (sign chosen so all radial coefficients lead with +erf)
        s3 = (15.0 * erf * rinv4
              - (30.0 * a / sqrt_pi) * gauss * rinv3
              - (20.0 * a ** 3 / sqrt_pi) * gauss * rinv
              - (8.0 * a ** 5 / sqrt_pi) * gauss * r_ij_norm)
        # s4 = ds3/dr - 3*s3/r
        s4 = (105.0 * erf * rinv5
              - (210.0 * a / sqrt_pi) * gauss * rinv4
              - (140.0 * a ** 3 / sqrt_pi) * gauss * rinv2
              - (56.0 * a ** 5 / sqrt_pi) * gauss
              - (16.0 * a ** 7 / sqrt_pi) * gauss * r_ij_norm ** 2)

        rrr = torch.einsum('nmi,nmj,nmk->nmijk', rhat, rhat, rhat)
        term_delta_r = (torch.einsum('ab,ijc->ijabc', eye, rhat)
                        + torch.einsum('ac,ijb->ijabc', eye, rhat)
                        + torch.einsum('bc,ija->ijabc', eye, rhat))
        f_Qu = (s3[..., None, None, None] * rrr
                - (s2 * rinv)[..., None, None, None] * term_delta_r) * norm_const   # [n,n,3,3,3]

        rrrr = torch.einsum('ija,ijb,ijc,ijd->ijabcd', rhat, rhat, rhat, rhat)
        term_delta_rr = (torch.einsum('ab,ijc,ijd->ijabcd', eye, rhat, rhat)
                         + torch.einsum('ac,ijb,ijd->ijabcd', eye, rhat, rhat)
                         + torch.einsum('ad,ijb,ijc->ijabcd', eye, rhat, rhat)
                         + torch.einsum('bc,ija,ijd->ijabcd', eye, rhat, rhat)
                         + torch.einsum('bd,ija,ijc->ijabcd', eye, rhat, rhat)
                         + torch.einsum('cd,ija,ijb->ijabcd', eye, rhat, rhat))
        term_delta_delta = (torch.einsum('ab,cd->abcd', eye, eye)
                            + torch.einsum('ac,bd->abcd', eye, eye)
                            + torch.einsum('ad,bc->abcd', eye, eye))
        term_delta_delta = term_delta_delta.unsqueeze(0).unsqueeze(0)
        f_QQ = (s4[..., None, None, None, None] * rrrr
                - (s3 * rinv)[..., None, None, None, None] * term_delta_rr
                + (s2 * rinv2)[..., None, None, None, None] * term_delta_delta) * norm_const  # [n,n,3,3,3,3]

    return f_qq, f_qu, f_uu, f_Qu, f_QQ


def bare_multipole_tensors(r):
    """
    Bare 1/r multipole derivative tensors for all pairs (zero on i==j).
    Derivatives are taken with respect to r_j where r = r_j - r_i.

        T1[i,j,c]       = d(1/r)/dr_c            = -r̂_c / r^2
        T2[i,j,c,d]     = d^2(1/r)/dr^2          = (3 r̂r̂ - δ) / r^3
        T3[i,j,c,d,e]   = d^3(1/r)/dr^3          = (-15 r̂r̂r̂ + 3 [δ r̂]_sym) / r^4
        T4[i,j,c,d,e,f] = d^4(1/r)/dr^4          = (105 r̂r̂r̂r̂ - 15 [δ r̂r̂]_sym + 3 [δδ]_sym) / r^5
    """
    n = r.shape[0]
    device, dtype = r.device, r.dtype
    eye = torch.eye(3, device=device, dtype=dtype)
    mask_off = ~torch.eye(n, dtype=torch.bool, device=device)

    r_ij = r.unsqueeze(0) - r.unsqueeze(1)
    dist = torch.norm(r_ij, dim=-1)
    rinv = torch.zeros_like(dist)
    rinv[mask_off] = 1.0 / dist[mask_off]
    rinv2 = rinv * rinv
    rinv3 = rinv2 * rinv
    rinv4 = rinv3 * rinv
    rinv5 = rinv4 * rinv
    rhat = r_ij * rinv[..., None]

    T1 = -rhat * rinv2[..., None]                                                          # [n,n,3]

    outer = rhat[..., :, None] * rhat[..., None, :]
    T2 = (3.0 * outer - eye[None, None]) * rinv3[..., None, None]                          # [n,n,3,3]

    rrr = torch.einsum('nmi,nmj,nmk->nmijk', rhat, rhat, rhat)
    delta_r = (torch.einsum('ab,ijc->ijabc', eye, rhat)
               + torch.einsum('ac,ijb->ijabc', eye, rhat)
               + torch.einsum('bc,ija->ijabc', eye, rhat))
    T3 = (-15.0 * rrr + 3.0 * delta_r) * rinv4[..., None, None, None]                      # [n,n,3,3,3]

    rrrr = torch.einsum('ija,ijb,ijc,ijd->ijabcd', rhat, rhat, rhat, rhat)
    delta_rr = (torch.einsum('ab,ijc,ijd->ijabcd', eye, rhat, rhat)
                + torch.einsum('ac,ijb,ijd->ijabcd', eye, rhat, rhat)
                + torch.einsum('ad,ijb,ijc->ijabcd', eye, rhat, rhat)
                + torch.einsum('bc,ija,ijd->ijabcd', eye, rhat, rhat)
                + torch.einsum('bd,ija,ijc->ijabcd', eye, rhat, rhat)
                + torch.einsum('cd,ija,ijb->ijabcd', eye, rhat, rhat))
    delta_delta = (torch.einsum('ab,cd->abcd', eye, eye)
                   + torch.einsum('ac,bd->abcd', eye, eye)
                   + torch.einsum('ad,bc->abcd', eye, eye))
    delta_delta = delta_delta.unsqueeze(0).unsqueeze(0)
    T4 = (105.0 * rrrr - 15.0 * delta_rr + 3.0 * delta_delta) * rinv5[..., None, None, None, None]  # [n,n,3,3,3,3]

    return T1, T2, T3, T4
