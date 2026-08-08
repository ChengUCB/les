# How LES works

## The problem

A short-range MLIP writes the energy as a sum of local atomic contributions inside a cutoff,

$$E^\text{sr} = \sum_{i=1}^{N} E_i$$

with $r_c$ typically around 5 to 6 Å. Electrostatics does not fit in that form: a Coulomb
interaction decays as $1/r$.

## The idea

LES splits the energy in two,

$$E = E^\text{sr} + E^\text{lr}$$

and builds $E^\text{lr}$ from a **latent atomic charge** $q_i^\text{les}$ or other latent quantites, predicted by a small
network from the same local invariant features the model already uses for $E_i$.

Two design principles make this work, and they are the whole method
([Kim & Cheng 2026](https://doi.org/10.1063/5.0316886)):

1. **Use a Coulomb functional form with environment-dependent charges.** The physics is in
   the functional form, so the asymptotics are right by construction and the charge stays
   interpretable.
2. **Do not train on DFT partial charges.** There is no unique mapping from an electron
   density to atomic charges, so fitting one
   choice fixes an arbitrary electrostatic description and is not necessary.

So nothing supervises $q_i^\text{les}$. The only training signal is the total energy and the
forces, and the charges are whatever makes those come out right. This is why they are called
*latent* ([King et al. 2025](https://www.nature.com/articles/s41467-025-63852-x)).

They are not arbitrary, either. Because $E^\text{lr}$ depends on them only through a physical
In practice they come out physically meaningful: models trained on energies
and forces alone predict molecular dipoles, Born effective charges, IR spectra and ionic
conductivities they never saw
([Zhong et al. 2025](https://doi.org/10.1038/s41524-025-01911-z)).

Coulomb interactions between atomic charges are screened by the fast electronic background,
described as a homogeneous dielectric medium with relative permittivity $\varepsilon_e$. Learning
from forces absorbs that screening automatically, so the learned quantities are *scaled* physical
ones,

$$q_i^\text{les} = \frac{q_i}{\sqrt{\varepsilon_e}}, \qquad
\mathbf{u}_i^\text{les} = \frac{\mathbf{u}_i}{\sqrt{\varepsilon_e}}, \qquad
\mathbf{Q}_i^\text{les} = \frac{\mathbf{Q}_i}{\sqrt{\varepsilon_e}}$$

and only the vacuum permittivity enters the Ewald sum. $\varepsilon_e$ therefore never has to be
considered while training or predicting energies and forces -- it appears only when recovering
unscaled physical quantities such as BECs and polarizabilities ([Kim, King, Park et al. 2026](https://arxiv.org/abs/2605.05746)):

* $\varepsilon_e = 1$ for a system in vacuum, such as an isolated molecule;
* $\varepsilon_e = \varepsilon_\infty$, the high-frequency (electronic) dielectric constant, for a
  homogeneous bulk system with no induced-dipole term -- this is the original LES case;
* $\varepsilon_e = \varepsilon_\infty / (1 + \chi^\text{les})$ when induced dipoles are included,
  since they contribute their own susceptibility $\chi^\text{les} = \sum_i \alpha_i^\text{les} / (\varepsilon_0 V)$.

## The Ewald sum

Each latent charge is smeared into a Gaussian of width $\sigma$ (about 1 Å). That single
choice makes both halves of the calculation finite.

**Isolated systems** are summed pairwise in real space, where the smeared charges replace the
bare $1/r$ with an error function:

$$E^\text{lr} = \frac{1}{2}\frac{1}{4\pi\varepsilon_0}\sum_{i=1}^{N}\sum_{j=1}^{N}
\bigl[1-\varphi(r_{ij})\bigr]\frac{q_i^\text{les} q_j^\text{les}}{r_{ij}},
\qquad \varphi(r) = \operatorname{erfc}\!\left(\frac{r}{\sqrt{2}\,\sigma}\right)$$

At short range the kernel tends to a constant rather than diverging -- the Gaussian overlap
removes the singularity. This matters for more than numerics: the long-range term is *smooth
and weak* exactly where the short-range model is already accurate, so the two do not fight
over the same physics.

**Periodic systems** are summed in reciprocal space:

$$E^\text{lr} = \frac{1}{2\varepsilon_0 V}\sum_{0<k<k_c}\frac{e^{-\sigma^2k^2/2}}{k^2}
\bigl|S(\mathbf{k})\bigr|^2,
\qquad S(\mathbf{k}) = \sum_{i=1}^{N} q_i^\text{les}\, e^{i\mathbf{k}\cdot\mathbf{r}_i}$$

The Gaussian factor $e^{-\sigma^2k^2/2}$ truncates the sum by itself: large $k$ contributes
nothing, so a finite number of terms is exact to any tolerance you like. `dl` sets the cutoff
$k_c = 2\pi/\texttt{dl}$; the default `dl: 2.0` Å corresponds to $k_c = \pi$.

Charge neutrality is not imposed. For a neutral system the learned charges sum to very nearly
zero on their own, and any residual is absorbed as a uniform background -- the tinfoil boundary
condition already implicit in the reciprocal-space sum.

`remove_self_interaction` subtracts each charge's interaction with its own Gaussian, which is
an artefact of the smearing rather than physics.

## Multipoles and polarization response

A single scalar per atom is the leading term of a multipole expansion of the atomic charge
density. The extension adds the next ones and a linear response, all still learned from
energies and forces alone
([Kim, King, Park et al. 2026](https://arxiv.org/abs/2605.05746)).

**Fixed multipoles.** Latent dipoles $\mathbf{u}_i^\text{les}$ (an equivariant `1o` vector) and
traceless quadrupoles $\mathbf{Q}_i^\text{les}$ (`2e`) enter the same Ewald machinery through the
structure factor:

$$S(\mathbf{k}) = \sum_{i=1}^{N}\Bigl(q_i^\text{les} + i\,\mathbf{k}\cdot\mathbf{u}_i^\text{les}
- \tfrac{1}{2}\,\mathbf{k}\cdot\mathbf{Q}_i^\text{les}\cdot\mathbf{k}\Bigr)
e^{i\mathbf{k}\cdot\mathbf{r}_i}$$

Each successive order decays with one more factor of $1/r$, so the expansion is truncated at
dipole or quadrupole level.

**Induced response.** Instead of a global charge-equilibration solve, the residual non-local
effects are captured by *non-self-consistent* linear response: the induced terms respond once
to the field of the fixed multipoles, not to each other. With a hardness $\kappa_i^{-1}$ and a
polarizability $\boldsymbol{\alpha}_i$,

$$\Delta q_i = -\kappa_i \Phi(\mathbf{r}_i), \qquad
U_i^\text{iq} = -\tfrac{1}{2}\kappa_i\Phi^2(\mathbf{r}_i)$$

$$\Delta\mathbf{u}_i = \boldsymbol{\alpha}_i\cdot\mathbf{E}(\mathbf{r}_i), \qquad
U_i^\text{iu} = -\tfrac{1}{2}\mathbf{E}(\mathbf{r}_i)\cdot\boldsymbol{\alpha}_i\cdot\mathbf{E}(\mathbf{r}_i)$$

so the charge distribution is no longer a function of geometry alone -- it reacts to the
electrostatics it is itself generating.

The total is assembled as

$$U = U^\text{sr} + U^\text{elec} + \sum_i U_i^\text{iq} + \sum_i U_i^\text{iu}$$

**Naming.** Model names in the published fits encode which terms are on, and they map
one-to-one onto the `les_args` flags:

| suffix | term | flag |
|---|---|---|
| `-les` | monopoles only | (default) |
| `-u` | + dipoles | `use_dipole` |
| `-Q` | + quadrupoles | `use_quadrupole` |
| `-iq` | + induced charge | `use_induced_charge` |
| `-iu` | + induced dipole | `use_induced_dipole` |

So `nequiples-uQiqiu` is monopoles, dipoles, quadrupoles, induced charge and induced dipole --
the full set. Adding terms generally improves accuracy, with the largest single gain coming
from the monopole itself and diminishing returns after that.

## Born effective charges

The polarization of a configuration follows from the latent variables,

$$\mathbf{P} = \sum_i (q_i + \Delta q_i)\,\mathbf{r}_i + \sum_i (\mathbf{u}_i + \Delta\mathbf{u}_i)$$

and the Born effective charge tensor is its derivative with respect to an atomic position:

$$Z^*_{i\alpha\beta} = \frac{\partial P_\alpha}{\partial r_{i\beta}}$$

Since the latent variables are themselves differentiable functions of every position, autograd
gives this directly -- no finite differences, no extra training target. For a homogeneous
periodic system the charge part is taken in the $k\to0$ limit,

$$Z^*_{i\alpha\beta} = \frac{\partial P^u_\alpha}{\partial r_{i\beta}}
+ \lim_{k\to0}\Re\left[e^{-ikr_{i\alpha}}\frac{\partial P^q_\alpha(k)}{\partial r_{i\beta}}\right],
\qquad P^q_\alpha(k) = \sum_i \frac{\sqrt{\varepsilon_\infty}\,q_i^\text{les}}{ik} e^{ikr_{i\alpha}}$$

where $\varepsilon_\infty$ is what `epsilon_factor` sets. It is $\varepsilon_\infty$ here, not
$\varepsilon_e$: for a model without induced dipoles the two coincide, and when induced dipoles are
present $\varepsilon_e$ is derived from $\varepsilon_\infty$ by the relation above.

BECs are the strongest evidence that the latent charges are not a fitting artefact: they are a
response property that never appeared in training, they agree with DFT, and they converge *faster* with training-set size than the forces do ([Kim & Cheng 2026](https://doi.org/10.1063/5.0316886)).

```{warning}
The periodic expression needs a single high-frequency permittivity $\varepsilon_\infty$ for a
homogeneous bulk material. For heterogeneous systems -- interfaces between materials with
different $\varepsilon_\infty$ -- it is not obvious how to choose it, and extending LES-based
BEC extraction to such systems is an open problem.
```

## Cost, and what it cannot do

The long-range term is a small fraction of the short-range network's cost at typical settings:
the reciprocal sum is inexpensive next to the message passing, and MD timings with and without
LES nearly coincide. The augmentation has been called effectively a "free lunch".

Two limitations are worth knowing before you rely on it:

* The charges come from **local** features, so there is no explicit mechanism for truly
  long-range charge transfer through mobile carriers -- an induced surface charge on a
  macroscopic metal electrode, for instance. Coupling to an explicit metallic boundary model is
  the current remedy.
* Because forces are the autograd derivative of a **global** energy, distributing the force
  evaluation across GPUs or MPI ranks is not straightforward. This is why LES in LAMMPS is
  restricted to [one MPI rank](https://nequip-les.readthedocs.io/en/latest/guide/lammps.html#one-mpi-rank).

## Reading

| paper | what it covers |
|---|---|
| [Latent Ewald summation for machine learning of long-range interactions](https://www.nature.com/articles/s41524-025-01577-7) | the original paper, method |
| [Machine learning of charges and long-range interactions from energies and forces](https://www.nature.com/articles/s41467-025-63852-x) | benchmarks, why energies and forces alone suffice |
| [Machine learning interatomic potential can infer electrical response](https://doi.org/10.1038/s41524-025-01911-z) | BECs, IR spectra, finite-field MD |
| [A universal augmentation framework for long-range electrostatics in MLIPs](https://pubs.acs.org/doi/10.1021/acs.jctc.5c01400) | the MLIP-agnostic formulation this package implements |
| [Long-range electrostatics for MLIPs is easier than we thought](https://doi.org/10.1063/5.0316886) | the LES design principles, and where LES sits among the alternatives |
| [Polarizable atomic multipoles for learning long-range electrostatics](https://arxiv.org/abs/2605.05746) | the multipole and response terms |

Full entries under [Citation](citation.md).
