import torch

__all__ = ['segment_onehot', 'segment_softmax', 'segment_logsumexp', 'quadratic_energy']


def segment_onehot(batch: torch.Tensor, n_batch: int,
                   dtype: torch.dtype) -> torch.Tensor:
    """[n_batch, n_atoms] indicator of the structure each atom belongs to."""
    return (batch.unsqueeze(0)
            == torch.arange(n_batch, device=batch.device).unsqueeze(1)).to(dtype)


def _segment_max(scores: torch.Tensor, onehot: torch.Tensor) -> torch.Tensor:
    """Largest score within each structure, [n_batch].

    Slots outside a structure are pushed below every real score with a finite value,
    not -inf: a structure holding no atoms would otherwise leave -inf here, and 0 * inf
    in the broadcast back onto atoms is NaN for every structure in the batch.
    """
    floor = -(scores.abs().max() + 1.0)
    wide = onehot * scores.unsqueeze(0) + (1.0 - onehot) * floor    # [n_batch, n_atoms]
    return wide.max(dim=1).values


def segment_softmax(scores: torch.Tensor, onehot: torch.Tensor) -> torch.Tensor:
    """Softmax of `scores` within each structure, [n_atoms].

    Shifted by the per-structure maximum, so adding a common offset to a structure's
    scores leaves the result unchanged -- the gauge freedom the frontier models rely on.
    """
    onehot_t = onehot.transpose(0, 1)
    shifted = scores - torch.matmul(onehot_t, _segment_max(scores, onehot))
    weights = torch.exp(shifted)
    return weights / torch.matmul(onehot_t, torch.matmul(onehot, weights))


def segment_logsumexp(scores: torch.Tensor, onehot: torch.Tensor) -> torch.Tensor:
    """log-sum-exp of `scores` within each structure, [n_batch]."""
    smax = _segment_max(scores, onehot)
    shifted = scores - torch.matmul(onehot.transpose(0, 1), smax)
    return smax + torch.log(torch.matmul(onehot, torch.exp(shifted)))


def quadratic_energy(K: torch.Tensor, z: torch.Tensor,
                     onehot: torch.Tensor) -> torch.Tensor:
    """1/2 z^T K z per structure, [n_batch].

    K must already carry this module's energy convention, i.e. come from
    compute_A_matrix(self_interaction = not remove_self_interaction). No self-energy
    correction is applied here -- energy_from_A is the routine that expects the other
    convention and undoes it.
    """
    return 0.5 * torch.matmul(onehot, z * torch.matmul(K, z))
