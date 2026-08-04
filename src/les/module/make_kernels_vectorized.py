import torch
from typing import Optional

__all__ = ['multipole_pair_energy', 'multipole_potential_field']


class _ExpSaveInput(torch.autograd.Function):
    """
    exp() whose backward recomputes from the saved *input*.
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return grad_out * torch.exp(x)


def multipole_pair_energy(q: torch.Tensor,
                          u: Optional[torch.Tensor],
                          quad: Optional[torch.Tensor],
                          r_ij: torch.Tensor,
                          dist: torch.Tensor,
                          dist_sq: torch.Tensor,
                          rinv: torch.Tensor,
                          erf_val: torch.Tensor,
                          sigma: float,
                          ) -> torch.Tensor:
    """
    Per-pair energy of the dipole/quadrupole real-space terms, on an [N, N] grid.

    make_kernels.py builds the full f_qu/f_uu/f_Qu/f_QQ tensors; here the same
    kernels are contracted with the multipoles analytically, so the [N, N, 3, 3,
    3, 3] intermediates are never formed and every op stays friendly to
    torch.compile and AOTInductor export.

    r_ij is r_j - r_i (the convention of the reference kernels). Returns the
    contribution to be added to the monopole-monopole term, before the
    cross-batch/self mask and the 1/(4 pi eps0) prefactor.
    """
    n_q = q.shape[1]
    N = q.shape[0]
    a = 1.0 / (sigma * (2.0 ** 0.5))
    sqrt_pi = torch.pi ** 0.5

    rinv2 = rinv * rinv
    rinv3 = rinv2 * rinv
    # the Gaussian argument depends on the positions, so use the export-safe exp
    # TorchScript cannot compile a custom autograd.Function, so it gets the plain
    # exp; is_scripting() folds to a constant at compile time and the other branch is
    # pruned. Eager and AOTInductor keep _ExpSaveInput, whose backward recomputes exp
    # from the saved input rather than reusing the output (which breaks AOTI export).
    if torch.jit.is_scripting():
        gauss = torch.exp(-(a * a) * dist_sq)
    else:
        gauss = _ExpSaveInput.apply(-(a * a) * dist_sq)
    s1 = erf_val * rinv3 - (2.0 * a / sqrt_pi) * gauss * rinv2
    s2 = (3.0 * erf_val * rinv3
          - (6.0 * a / sqrt_pi) * gauss * rinv2
          - (4.0 * a ** 3 / sqrt_pi) * gauss)

    # the multipoles enter by broadcasting over the pair grid rather than by
    # gathering with an [N, N] index: the backward of such a gather is a
    # scatter-add, which the inductor CPU backend fails to vectorize (an
    # AssertionError in its atomic_add codegen), and broadcasting backs up to a
    # plain reduction instead
    q_j = q.unsqueeze(0)                                             # [1, N, n_q]
    r_ij_e = r_ij.unsqueeze(2)                                       # [N, N, 1, 3]
    rhat = r_ij * rinv.unsqueeze(-1)                                 # [N, N, 3]
    rhat_e = rhat.unsqueeze(2)                                       # [N, N, 1, 3]

    extra = torch.zeros_like(dist)                                   # [N, N]
    uj_dot_r = torch.zeros(0, device=q.device, dtype=q.dtype)
    u_j = torch.zeros(0, device=q.device, dtype=q.dtype)

    if u is not None:
        u_i = u.unsqueeze(1)                                         # [N, 1, n_q, 3]
        u_j = u.unsqueeze(0)                                         # [1, N, n_q, 3]

        # charge-dipole: q_j (u_i . f_qu),  f_qu = s1 r_ij
        ui_dot_r = (u_i * r_ij_e).sum(dim=-1)                        # [N, N, n_q]
        extra = extra + (s1.unsqueeze(-1) * ui_dot_r * q_j).sum(dim=-1)

        # dipole-dipole: -1/2 u_j . f_uu . u_i,  f_uu = s2 rhat rhat - s1 I
        uj_dot_r = (u_j * r_ij_e).sum(dim=-1)                        # [N, N, n_q]
        ui_dot_uj = (u_i * u_j).sum(dim=-1)                          # [N, N, n_q]
        uu_term = (s2.unsqueeze(-1) * ui_dot_r * uj_dot_r * rinv2.unsqueeze(-1)
                   - s1.unsqueeze(-1) * ui_dot_uj)                   # [N, N, n_q]
        extra = extra - 0.5 * uu_term.sum(dim=-1)

    if quad is not None:
        rinv4 = rinv3 * rinv
        rinv5 = rinv4 * rinv
        s3 = (15.0 * erf_val * rinv4
              - (30.0 * a / sqrt_pi) * gauss * rinv3
              - (20.0 * a ** 3 / sqrt_pi) * gauss * rinv
              - (8.0 * a ** 5 / sqrt_pi) * gauss * dist)
        s4 = (105.0 * erf_val * rinv5
              - (210.0 * a / sqrt_pi) * gauss * rinv4
              - (140.0 * a ** 3 / sqrt_pi) * gauss * rinv2
              - (56.0 * a ** 5 / sqrt_pi) * gauss
              - (16.0 * a ** 7 / sqrt_pi) * gauss * dist_sq)

        # f_uu/f_Qu/f_QQ are symmetric in their index groups, so Q only enters
        # through Qr = (Q + Q^T) rhat, QRR = Q : rhat rhat and QTr = tr(Q).
        # The j-side reductions follow from rhat_ji = -rhat_ij.
        Qsym = quad + quad.transpose(-1, -2)                         # [N, n_q, 3, 3]
        QTr = quad.diagonal(dim1=-2, dim2=-1).sum(dim=-1)            # [N, n_q]
        # broadcast-multiply-and-sum, not matmul: a broadcast matmul on the
        # position-dependent path breaks AOTInductor export
        Qr_i = (Qsym.unsqueeze(1) * rhat.unsqueeze(2).unsqueeze(2)).sum(dim=-1)  # [N, N, n_q, 3]
        Qr_j = -Qr_i.transpose(0, 1)
        QRR_i = 0.5 * (Qr_i * rhat_e).sum(dim=-1)                    # [N, N, n_q]
        QRR_j = QRR_i.transpose(0, 1)
        QTr_i = QTr.unsqueeze(1)                                     # [N, 1, n_q]
        QTr_j = QTr.unsqueeze(0)                                     # [1, N, n_q]

        # charge-quadrupole: q_j (1/2 Q_i : f_uu)
        e_phi_Q = 0.5 * (s2.unsqueeze(-1) * QRR_i - s1.unsqueeze(-1) * QTr_i)
        extra = extra + (q_j * e_phi_Q).sum(dim=-1)

        # quadrupole-quadrupole: 1/8 Q_i : f_QQ : Q_j
        Qflat = quad.reshape(N, n_q, 9).permute(1, 0, 2)             # [n_q, N, 9]
        QflatT = quad.transpose(-1, -2).reshape(N, n_q, 9).permute(1, 0, 2)
        QQ = torch.bmm(Qflat, Qflat.transpose(1, 2)).permute(1, 2, 0)    # Q_i : Q_j
        QQT = torch.bmm(Qflat, QflatT.transpose(1, 2)).permute(1, 2, 0)  # Q_i : Q_j^T
        QrQr = (Qr_i * Qr_j).sum(dim=-1)                             # [N, N, n_q]
        qq_term = (s4.unsqueeze(-1) * QRR_i * QRR_j
                   - (s3 * rinv).unsqueeze(-1) * (QTr_i * QRR_j + QRR_i * QTr_j + QrQr)
                   + (s2 * rinv2).unsqueeze(-1) * (QTr_i * QTr_j + QQ + QQT))
        extra = extra + 0.125 * qq_term.sum(dim=-1)

        if u is not None:
            # dipole-quadrupole: -u_j . (1/2 Q_i : f_Qu)
            uj_dot_rhat = uj_dot_r * rinv.unsqueeze(-1)              # [N, N, n_q]
            uj_dot_Qr = (u_j * Qr_i).sum(dim=-1)                     # [N, N, n_q]
            Qu_term = (s3.unsqueeze(-1) * QRR_i * uj_dot_rhat
                       - (s2 * rinv).unsqueeze(-1) * (QTr_i * uj_dot_rhat + uj_dot_Qr))
            extra = extra - 0.5 * Qu_term.sum(dim=-1)

    return extra


def multipole_potential_field(q: torch.Tensor,
                             u: Optional[torch.Tensor],
                             quad: Optional[torch.Tensor],
                             r_ij: torch.Tensor,
                             dist: torch.Tensor,
                             dist_sq: torch.Tensor,
                             rinv: torch.Tensor,
                             erf_val: torch.Tensor,
                             sigma: float,
                             norm_const: float,
                             keep_pair: torch.Tensor,   # [N, N] bool
                             ):
    """
    Electrostatic potential and field at every atom, from the real-space kernels.

    These feed the induced charge (-kappa * phi) and induced dipole (alpha * E)
    terms, so unlike the pair energy they must come out in physical units --
    norm_const is applied here.

    Sums over the source index i are written as matmul/bmm rather than a masked
    reduction: fused reductions over a large axis are miscompiled by the Metal
    inductor backend, and einsum breaks AOTInductor export.

    Excluded pairs are removed with `torch.where`, never by multiplying with a 0/1
    mask: a masked-out entry can still carry an inf (its separation is zero), and
    inf * 0 is NaN, which no later masking can clean up again -- see
    ChengUCB/les#11. `where` also blocks the backward, so a non-finite gradient in
    the excluded branch cannot reach the parameters either.

    Returns (phi [N, n_q], field [N, n_q, 3]).
    """
    zero = torch.zeros((), device=q.device, dtype=q.dtype)
    m2 = keep_pair                                    # [N, N]
    m3 = keep_pair.unsqueeze(-1)                      # [N, N, 1]
    m4 = m3.unsqueeze(-1)                             # [N, N, 1, 1]
    N, n_q = q.shape
    a = 1.0 / (sigma * (2.0 ** 0.5))
    sqrt_pi = torch.pi ** 0.5

    rinv2 = rinv * rinv
    rinv3 = rinv2 * rinv
    # see multipole_pair_energy
    if torch.jit.is_scripting():
        gauss = torch.exp(-(a * a) * dist_sq)
    else:
        gauss = _ExpSaveInput.apply(-(a * a) * dist_sq)
    s1 = erf_val * rinv3 - (2.0 * a / sqrt_pi) * gauss * rinv2
    s2 = (3.0 * erf_val * rinv3
          - (6.0 * a / sqrt_pi) * gauss * rinv2
          - (4.0 * a ** 3 / sqrt_pi) * gauss)

    rhat = r_ij * rinv.unsqueeze(-1)                                  # [N, N, 3]
    # contract over the source index i: [N(j), n_q, N(i)] @ [N(j), N(i), 3]
    r_ij_j = r_ij.transpose(0, 1)                                     # [N(j), N(i), 3]
    rhat_j = rhat.transpose(0, 1)                                     # [N(j), N(i), 3]

    # --- monopole: phi = sum_i q_i erf(ar)/r,  field = sum_i q_i s1 r_ij ---
    phi = torch.matmul(torch.where(m2, erf_val * rinv, zero).transpose(0, 1), q)     # [N, n_q]
    w_q = (q.unsqueeze(1) * torch.where(m2, s1, zero).unsqueeze(-1))                 # [N(i), N(j), n_q]
    field = torch.bmm(w_q.permute(1, 2, 0), r_ij_j)                    # [N(j), n_q, 3]

    if u is not None:
        u_i = u.unsqueeze(1)                                           # [N, 1, n_q, 3]
        ui_dot_r = (u_i * r_ij.unsqueeze(2)).sum(dim=-1)               # [N, N, n_q]
        # phi += sum_i u_i . f_qu,  f_qu = s1 r_ij
        phi = phi + (torch.where(m2, s1, zero).unsqueeze(-1) * ui_dot_r).sum(dim=0)
        # field += sum_i f_uu . u_i = s2 (u_i.rhat) rhat - s1 u_i
        w_u = torch.where(m2, s2, zero).unsqueeze(-1) * ui_dot_r * rinv.unsqueeze(-1)  # [N,N,n_q]
        field = field + torch.bmm(w_u.permute(1, 2, 0), rhat_j)
        field = field - torch.matmul(torch.where(m2, s1, zero).transpose(0, 1),
                                     u.reshape(N, n_q * 3)).reshape(N, n_q, 3)

    if quad is not None:
        rinv4 = rinv3 * rinv
        s3 = (15.0 * erf_val * rinv4
              - (30.0 * a / sqrt_pi) * gauss * rinv3
              - (20.0 * a ** 3 / sqrt_pi) * gauss * rinv
              - (8.0 * a ** 5 / sqrt_pi) * gauss * dist)
        Qsym = quad + quad.transpose(-1, -2)                           # [N, n_q, 3, 3]
        QTr = quad.diagonal(dim1=-2, dim2=-1).sum(dim=-1)              # [N, n_q]
        Qr_i = (Qsym.unsqueeze(1) * rhat.unsqueeze(2).unsqueeze(2)).sum(dim=-1)  # [N,N,n_q,3]
        QRR_i = 0.5 * (Qr_i * rhat.unsqueeze(2)).sum(dim=-1)           # [N, N, n_q]
        QTr_i = QTr.unsqueeze(1)                                       # [N, 1, n_q]

        # phi += sum_i 1/2 Q_i : f_uu
        phi = phi + (0.5 * torch.where(m3, s2.unsqueeze(-1) * QRR_i
                                       - s1.unsqueeze(-1) * QTr_i, zero)).sum(dim=0)
        # field += sum_i 1/2 Q_i : f_Qu
        w_Q = 0.5 * torch.where(m3, s3.unsqueeze(-1) * QRR_i
                                - (s2 * rinv).unsqueeze(-1) * QTr_i, zero)
        field = field + torch.bmm(w_Q.permute(1, 2, 0), rhat_j)
        field = field - (0.5 * torch.where(m4, (s2 * rinv).unsqueeze(-1).unsqueeze(-1)
                                          * Qr_i, zero)).sum(dim=0)

    return phi * norm_const, field * norm_const
