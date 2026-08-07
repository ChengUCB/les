# MACE

LES is implemented in the main [MACE](https://github.com/ACEsuit/mace) repository. Training is
the ordinary `run_train.py` with two extra flags.

Complete example, bulk water with the full multipole and response set:
[`MLIPs/MACE-LES/water/maceles-uQiqiu-r-4.5-nl-1`](https://github.com/ChengUCB/extended_les_fit/tree/main/MLIPs/MACE-LES/water/maceles-uQiqiu-r-4.5-nl-1).

## Training

From that directory's `fit.sh`:

```bash
python mace/scripts/run_train.py \
    --model="MACELES" \             # the LES-augmented model
    --les_arguments='les.yaml' \    # LES options, in a separate file
    --train_file="../train-H2O_RPBE-D3.xyz" \
    --hidden_irreps='128x0e + 128x1o' \
    --r_max=4.5 \
    --num_interactions=1 \
    ...
```

Only the first two lines are LES-specific. Everything else is MACE as documented in its own
repository.

## `les.yaml`

The options that
[`les.yaml`](https://github.com/ChengUCB/extended_les_fit/blob/main/MLIPs/MACE-LES/water/maceles-uQiqiu-r-4.5-nl-1/les.yaml)
sets for that model:

```yaml
use_dipole: True                      # default: False -- latent dipoles (1o)
use_quad: True                        # default: False -- latent quadrupoles (2e)
use_induced_charge: True              # default: False -- induced charge, needs a hardness kappa
use_induced_dipole: True              # default: False -- induced dipole, needs a polarizability alpha
use_anisotropic_polarizability: True  # default: False -- alpha as a tensor rather than a scalar
alpha_irreps: '0e+1o+2e'              # which irreps make up alpha
alpha_1o_nonlinear_readout: False     # MACE-specific readout choices for the 1o part of alpha
make_alpha_positive: False
alpha_1o_linear_w_pos: True
```

Anything not listed keeps its default, so a monopole-only model needs no `les.yaml` at all. The
Ewald parameters (`sigma`, `dl`, `is_periodic`, `N_max`) also go in this file and default to the
values in the [Ewald parameters](library.md#ewald-parameters).

```{note}
The key names are not identical across host MLIPs -- MACE uses `use_quad` where NequIP-LES uses
`use_quadrupole`, and the `alpha_1o_*` options exist only in MACE. Read the host package's
example rather than copying a config across packages.
```

## Molecular dynamics

Two ASE-based examples, both loading the trained model with `compute_bec=True` so the Born
effective charges are available:

* NVT: [`md_MACE.py`](https://github.com/ChengUCB/extended_les_fit/blob/main/MD_simulations/water/md_MACE.py)
* NVT under a static electric field:
  [`md_MACE_e_ext.py`](https://github.com/ChengUCB/extended_les_fit/blob/main/MD_simulations/water/md_MACE_e_ext.py) --
  takes the field along $z$ as its argument and adds the electrostatic forces
  $F_{i\beta} = Z^*_{i\alpha\beta} \mathcal{E}^0_\alpha$ obtained from the predicted BECs

```python
from mace.calculators import MACECalculator

calculator = MACECalculator(model_paths='H20.model', device='cuda',
                            compute_bec=True)
bec = atoms.calc.results['bec']
```
