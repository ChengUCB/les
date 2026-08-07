# CACE

LES is implemented in [CACE](https://github.com/BingqingCheng/cace). A CACE model is assembled
in Python from output modules, so LES enters as two of them.

Complete example, bulk water with the full multipole and response set:
[`fit_cace_les.py`](https://github.com/ChengUCB/extended_les_fit/blob/main/MLIPs/CACE-LES/water/caceles-uQiqiu-r-4.5-nl-1-nu3/fit_cace_les.py).

## The two modules

**`TensorReadout`** predicts the latent multipoles from the CACE representation. Which
quantities exist is decided by which keys you ask for at each rotation order:

```python
from cace.modules import TensorReadout

multipoles = TensorReadout(max_l=2,                     # highest rotation order read out
                           l0_key='kappas',             # scalar    -> hardness (induced charge)
                           l1_key='dipoles',            # vector    -> latent dipoles
                           l2_key=['alphas', 'quads'],  # rank-2    -> polarizability, quadrupoles
                           l0_output_scale=0.1,          # keeps the response a correction early on
                           l1_output_scale=1.,
                           l2_output_scale=1.,
                           )
```

Leaving a key out leaves that term out of the model: `l1_key` alone gives monopoles and dipoles,
no quadrupoles and no response.

**`LesWrapper`** takes those keys and evaluates the Ewald sum:

```python
from cace.modules import LesWrapper

les_e = LesWrapper(dipole_key='dipoles',
                   kappa_key='kappas',
                   alpha_key='alphas',
                   quad_key='quads',
                   energy_key='ewald_potential',   # where the long-range energy is written
                   add_scalar_alpha=True,          # add a scalar to the diagonal of alpha
                   compute_bec=False,              # default: False -- turn on to get BECs
                   )
```

The latent charges themselves need no key: they come from the representation's invariant
features. Any `*_key` you omit removes that term.

## Assembling the model

The long-range energy is a separate output that is added to the short-range one:

```python
sr_energy = cace.modules.atomwise.Atomwise(output_key='SR_energy', ...)

e_add = cace.modules.FeatureAdd(feature_keys=['SR_energy', 'ewald_potential'],
                                output_key='CACE_energy')

forces = cace.modules.Forces(energy_key='CACE_energy', forces_key='CACE_forces')

cace_nnp = NeuralNetworkPotential(
    representation=cace_representation,
    output_modules=[multipoles, les_e, sr_energy, e_add, forces],
)
```

Forces are taken from the *total* energy, so they include the long-range contribution. The
Ewald parameters (`sigma`, `dl`, `is_periodic`, `N_max`) are `LesWrapper` arguments and default
to the values in [Ewald parameters](library.md#ewald-parameters).
