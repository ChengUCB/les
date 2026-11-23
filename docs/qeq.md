# Charge equilibration (Qeq) and the MetalWall electrode

With `use_qeq: True` the charges of the selected atoms are not taken from the network. They are the
charges that minimise

```
U(q) = E_ewald(q) + Σ_i ( χ_i q_i + ½ J_i q_i² ) − Σ_i v_i q_i ,      Σ_i q_i = Q
```

- `E_ewald`: the LES electrostatic energy of all charges, with the Gaussian self energy of the equilibrated charges
- `χ_i`: electronegativity (V), `J_i`: hardness (eV/e²), `v_i`: applied potential (V), `Q`: total charge (e)

The other atoms keep their latent charges. One linear solve per call gives the charges; `E_lr` is `U` at
its minimum and forces come from autograd as usual.

The Siepmann–Sprik electrode (MetalWall) is the same solve with `χ = 0` and `J = 0`:
`qeq_siepmann_sprik: True`.

## Settings (`les_arguments`)

| key | default | meaning |
|---|---|---|
| `use_qeq` | `False` | turn charge equilibration on |
| `qeq_atomic_numbers` | `None` | elements whose atoms are equilibrated, e.g. `[78]`; `None`: all atoms |
| `qeq_siepmann_sprik` | `False` | Siepmann–Sprik electrode: `χ = 0`, `J = 0`, and `qeq_charge_scope` defaults to `'selected'` |
| `qeq_hardness` | `1.0` | hardness `J` (eV/e²) |
| `qeq_hardness_trainable` | `True` | one trainable `J` per element, starting from `qeq_hardness`; `False`: the constant |
| `qeq_chi` | `None` | electronegativity (V): a number for all elements, or `{atomic number: value}` (0 for the others) |
| `qeq_chi_trainable` | `False` | one trainable `χ` per element, starting from `qeq_chi` |
| `qeq_charge_scope` | `'system'` | what `total_charge` refers to. `'system'`: all atoms, the equilibrated atoms compensate the others. `'selected'`: the equilibrated atoms alone |
| `qeq_onsite_energy` | `True` | add `Σ(χq + ½Jq²)` to `E_lr`. `False`: `E_lr` is the electrostatic energy alone, as in 4G-HDNNP |
| `qeq_at_sites` | `True` | with charge offsets: solve with the equilibrated atoms at their sites `r + δ`. `False`: solve at the nuclei, then displace |
| `qeq_cache` | `False` | in eval mode, reuse the inverse matrix while the equilibrated atoms, the cell and the hardness are unchanged (MD with a fixed electrode) |

With charge offsets (`charge_offset_mode: True`):

| key | default | meaning |
|---|---|---|
| `charge_offset_atomic_numbers` | `None` | elements whose atoms carry an offset, e.g. `[78]`; `None`: all atoms |
| `charge_offset_axis` | `None` | an axis `n`, e.g. `[0, 0, 1]`: keep only the component of the offset along it, `δ = (v·n) n`. With `use_qeq` this applies to the equilibrated atoms only |

## Inputs and outputs of `Les.forward`

| argument | shape | meaning |
|---|---|---|
| `atomic_numbers` | `[n_atoms]` | required with `use_qeq` |
| `latent_charges` | `[n_atoms]` | charges of the atoms that are not equilibrated (ignored on the equilibrated atoms) |
| `latent_chi` | `[n_atoms]` | atom-wise electronegativity (V), added to `qeq_chi` |
| `latent_hardness` | `[n_atoms]` | atom-wise hardness, `softplus` of it is added to `J` |
| `total_charge` | `[n_structures]` | `Q` (e); `None`: 0 |
| `qeq_mask` | `[n_atoms]`, bool | restrict the equilibrated atoms further |
| `applied_potential` | `[n_atoms]` | `v_i` (V) at the nuclei; with offsets, `e_ext` (V/Å) moves it to the sites |

| output | shape | meaning |
|---|---|---|
| `latent_charges` | `[n_atoms]` | all charges, the equilibrated ones included |
| `qeq_lambda` | `[n_structures]` | chemical potential of the equilibrated atoms |
| `E_lr` | `[n_structures]` | `U` at its minimum |

## Examples

MetalWall on Pt, the electrolyte charges from the network:

```python
les = Les({"is_periodic": True, "use_qeq": True, "qeq_siepmann_sprik": True, "qeq_atomic_numbers": [78]})
out = les(positions=r, cell=cell, batch=batch, atomic_numbers=Z, latent_charges=q)
```

The same with charge offsets on every atom, on Pt only along z:

```python
les = Les({"is_periodic": True, "use_qeq": True, "qeq_siepmann_sprik": True, "qeq_atomic_numbers": [78],
           "charge_offset_mode": True, "charge_offset_axis": [0, 0, 1]})
out = les(positions=r, cell=cell, batch=batch, atomic_numbers=Z, latent_charges=q, latent_offsets=v)
```

Qeq on all atoms with a learned electronegativity and one trainable hardness per element:

```python
les = Les({"is_periodic": True, "use_qeq": True})
out = les(positions=r, cell=cell, batch=batch, atomic_numbers=Z, latent_chi=chi)
```

## Notes

- `E_lr` is the energy the charges minimise when `qeq_onsite_energy` is `True` (or `qeq_siepmann_sprik`) and the
  solve is at the sites. Otherwise the gradient goes through the solve; forces are exact in both cases.
- The Gaussian self energy of an equilibrated charge acts as its hardness, so it stays in `E_lr` also with
  `remove_self_interaction: True`.
- `qeq_cache` with an energy the charges do not minimise (`qeq_at_sites: False` with offsets, or
  `qeq_onsite_energy: False`): a reused call gives the exact energy and exact forces on the atoms that are not
  equilibrated, but incomplete forces on the equilibrated atoms and an incomplete stress. Use it for a fixed electrode.
- One charge channel; `latent_alphas` and `latent_kappas` are not supported together with `use_qeq`.
- Works with both Ewald modules, periodic and open boundaries, batches, and TorchScript. Models saved before
  load with `use_qeq` off.
- Tests: `PYTHONPATH=src python -m pytest test/test_qeq.py -q`
