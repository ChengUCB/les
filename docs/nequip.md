# NequIP and Allegro

LES for [NequIP](https://github.com/mir-group/nequip) and
[Allegro](https://github.com/mir-group/allegro) lives in its own extension package,
[NequIP-LES](https://github.com/ChengUCB/NequIP-LES), which has its own documentation:

**[nequip-les.readthedocs.io](https://nequip-les.readthedocs.io/en/latest/)**

| | |
|---|---|
| turning a config into a LES one | [Usage](https://nequip-les.readthedocs.io/en/latest/guide/usage.html) |
| every option, with defaults | [`les_args` reference](https://nequip-les.readthedocs.io/en/latest/guide/les_args.html) |
| which Ewald implementation, and what compilation needs | [Ewald implementations](https://nequip-les.readthedocs.io/en/latest/guide/ewald.html) |
| what can and cannot be compiled or deployed | [What works](https://nequip-les.readthedocs.io/en/latest/guide/deployment.html) |
| running in LAMMPS | [LAMMPS](https://nequip-les.readthedocs.io/en/latest/guide/lammps.html) |

Both backbones are supported by the same package; `base_model: nequip` or
`base_model: allegro` is the only difference.
