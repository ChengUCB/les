# Using the LES library directly

You do not need NequIP to use LES. The library is a single `torch.nn.Module` that takes
positions, a cell, and per-atom multipoles, and returns the long-range energy. This page is
for adding LES to your own model. If your MLIP already has LES, use its own page:
[MACE](mace.md), [CACE](cace.md), [NequIP or Allegro](nequip.md).

## Install

```bash
git clone https://github.com/ChengUCB/les.git
cd les && pip install -e .
```

## The module

```python
import torch
from les import Les

les = Les(les_arguments={})

out = les(
    positions=positions,          # [n_atoms, 3]
    cell=cell,                    # [n_structures, 3, 3]
    latent_charges=q,             # [n_atoms] -- your model's output
    batch=batch,                  # [n_atoms], which structure each atom belongs to
    compute_energy=True,
)

energy = out["E_lr"]              # add this to your short-range energy
```

`les_arguments` is the same dictionary documented in the
[Ewald parameters](#ewald-parameters) below -- it can also be a path to a YAML file holding it. Forces come from autograd on `positions` as usual; nothing special is needed.

## Inputs

| argument | shape | notes |
|---|---|---|
| `positions` | `[n_atoms, 3]` | required |
| `cell` | `[n_structures, 3, 3]` | required, even for isolated systems -- see below |
| `batch` | `[n_atoms]` | defaults to a single structure |
| `latent_charges` | `[n_atoms]` | your model's per-atom scalar |
| `latent_dipoles` | `[n_atoms, 3]` | optional, must be equivariant |
| `latent_quads` | `[n_atoms, 3, 3]` | optional, must be equivariant |
| `latent_kappas` | `[n_atoms]` | optional |
| `latent_alphas` | `[n_atoms]` or `[n_atoms, 3, 3]` | optional, dipole polarizability |
| `atomic_numbers` | `[n_atoms]` | needed by `use_fixed_atomic_charges` / `use_atomic_alpha` |
| `e_ext` | | optional external field |
| `desc` | `[n_atoms, n_features]` | descriptors, instead of `latent_charges` -- requires `use_atomwise: True`, and LES predicts the charges itself with its own MLP |

Pass either `latent_charges` or `desc`; passing neither raises error.

## Outputs

```python
{
    "E_lr":            ...,   # [n_structures] long-range energy
    "latent_charges":  ...,   # [n_atoms], with induced charge added if kappa was used
    "latent_dipoles":  ...,   # with induced dipole added if alpha was used
    "latent_quads":    ...,
    "latent_alphas":   ...,
    "BEC":             ...,   # only if compute_bec=True
}
```

The returned `latent_charges` is what you should write out if you want to inspect the
charges: it includes the induced contribution, while the tensor you passed in does not.

## Flags

* `compute_energy` (default `True`) -- the Ewald sum. Turn it off to get charges only.
* `compute_field` -- also return the electrostatic potential and field per atom.
* `compute_bec` -- Born effective charges. `bec_output_index` restricts them to one
  Cartesian direction, which is three times cheaper when that is all you need.


## Ewald parameters

These belong to the library and are shared by every host MLIP:

| key | default | meaning |
|---|---|---|
| `is_periodic` | `None` | `True` = periodic, `False` = non-periodic, `None` = the legacy implementation, which decides per structure. Only the vectorized implementation (`True`/`False`) can be compiled or exported. |
| `sigma` | `1.0` | Width (Å) of the Gaussian each latent charge is smeared over, and the Ewald splitting parameter. |
| `dl` | `2.0` | Resolution of the reciprocal-space sum (Å): the cutoff is `k_max = 2*pi/dl`. The default corresponds to `k_c = pi`. |
| `N_max` | `10` | Extent of the integer k-grid per direction. Keep `N_max * dl` above the cell's longest side. Periodic only. |
| `remove_self_interaction` | `True` | Subtract each charge's interaction with its own Gaussian. |

```{note}
We have checked that the default `sigma` and `dl` converge in essentially every case we have
tried, and they are what the published fits use. Changing them is not recommended.
```

For what the choice of implementation means for `torch.compile` and deployment, see
[Ewald implementations](https://nequip-les.readthedocs.io/en/latest/guide/ewald.html) in the NequIP-LES documentation.

## Hyperparameters

The defaults usually work. The one worth trying differently is `remove_self_interaction`:

> `remove_self_interaction=True` is the default and is the most robust choice.
> `remove_self_interaction=False` can sometimes yield a bit better training accuracy, but is
> less robust when training on finite systems and then extrapolating to periodic systems.

## Other MLIPs with LES

LES is already integrated into several packages, so check whether yours is covered before
wiring it in yourself:

| package | how to use it |
|---|---|
| MACE | [this guide](mace.md) -- [ACEsuit/mace](https://github.com/ACEsuit/mace) |
| CACE | [this guide](cace.md) -- [BingqingCheng/cace](https://github.com/BingqingCheng/cace) |
| NequIP / Allegro | [this guide](nequip.md) -- [ChengUCB/NequIP-LES](https://github.com/ChengUCB/NequIP-LES) |
| MatGL | [ChengUCB/matgl](https://github.com/ChengUCB/matgl) |

Training scripts and trained models for all of them:
[les_fit](https://github.com/ChengUCB/les_fit) and
[extended_les_fit](https://github.com/ChengUCB/extended_les_fit), including **MACELES-OFF**
trained on SPICE.
