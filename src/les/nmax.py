"""Resizing the reciprocal-space grid of a trained model.

``N_max`` fixes how far the integer k-grid runs; the physical cutoff is set by ``dl``
alone (``|k| <= 2*pi/dl``). Since ``n_i = k . a_i / (2*pi)``, the grid only has to reach
that sphere:

    N_max >= max_i |a_i| / dl

It is not a learned quantity and the buffers holding the grid are registered
non-persistently, so it is absent from every checkpoint. Raising it on a trained model
is therefore free -- above the requirement the extra k-vectors are masked out and the
energy and forces are unchanged to the last bit. Below it the reciprocal sum is silently
truncated, which is what happens when a cell is replicated for a larger simulation than
the one the model was fitted on.
"""
import math

import torch

__all__ = ['required_N_max', 'set_N_max']


def _cell_tensor(cell):
    """Accept a torch tensor, a numpy array or an ase Cell / Atoms."""
    if hasattr(cell, 'get_cell'):          # an ase Atoms
        cell = cell.get_cell()
    if hasattr(cell, '__array__') and not torch.is_tensor(cell):
        cell = torch.as_tensor(cell.__array__())
    cell = torch.as_tensor(cell).to(dtype=torch.float64)
    if cell.dim() == 2:
        cell = cell.unsqueeze(0)
    if cell.dim() != 3 or cell.shape[-2:] != (3, 3):
        raise ValueError(f"cell must be [3, 3] or [n_structures, 3, 3], got {tuple(cell.shape)}")
    return cell


def required_N_max(cell, dl: float = 2.0, margin: int = 0) -> int:
    """Smallest N_max that reaches the dl cutoff for this cell: ceil(max|a_i| / dl).

    cell may be a tensor, a numpy array, an ase Cell, or an ase Atoms.
    """
    lengths = _cell_tensor(cell).norm(dim=-1)          # [n_structures, 3]
    return int(math.ceil(float(lengths.max()) / float(dl))) + int(margin)


def set_N_max(model, N_max=None, cell=None, margin: int = 0, verbose: bool = False) -> int:
    """Set the k-grid extent on every LES Ewald inside `model`, and return what was set.

        set_N_max(calc.model, 25)                    # explicitly
        set_N_max(calc.model, cell=atoms.cell)       # from the cell you are about to run

    `model` may be a Les, an Ewald, or any module containing one, so a host MLIP can be
    passed straight in. Give either N_max or cell; with `cell`, `dl` is read from the
    module itself. `margin` adds headroom, e.g. for a cell that will grow under NPT.

    Modules on the legacy path (``is_periodic=None``) size their grid per structure at
    call time and are skipped; the count of modules changed is reported by `verbose`.
    """
    if (N_max is None) == (cell is None):
        raise ValueError("give exactly one of N_max or cell")

    targets = [m for m in model.modules() if hasattr(m, 'set_N_max')]
    if not targets:
        raise ValueError("no LES Ewald with a resizable k-grid found in this model; "
                         "the legacy path (is_periodic=None) sizes its grid per call")

    resolved = None
    for m in targets:
        want = int(N_max) if N_max is not None else required_N_max(cell, m.dl, margin)
        if resolved is None:
            resolved = want
        old = m.N_max
        m.set_N_max(want)
        if verbose:
            print(f"[les] {type(m).__name__}: N_max {old} -> {want} "
                  f"({(2 * want + 1) ** 3} k-vectors)")
    return resolved
