import torch
from torch import nn
from typing import Dict, Any, Union, Optional

from .module import (
    Atomwise,
    batch_slots,
    Ewald,
    Ewald_vectorized,
    BEC,
    FixedCharges,
    AtomicAlpha,
    Qeq,
    Nqe,
    Model0,
    Model0A,
    ModelB,
    quadratic_energy,
    segment_onehot,
)

__all__ = ['Les']

class Les(nn.Module):

    __constants__ = ['use_fixed_atomic_charges', 'use_atomic_alpha', 'use_atomwise', 'use_epsilon_r_scaling',
                     'use_qeq', 'use_nqe', 'use_model0', 'use_model0_a',
                     'use_model_b']
    def __init__(self, les_arguments: Union[Dict[str, Any], str] = {}):
        """
        LES model for long-range interations
        """
        super().__init__()

        if isinstance(les_arguments, str):
            import yaml
            with open(les_arguments, 'r') as file:
                les_arguments = yaml.safe_load(file)
                if les_arguments is None:
                    les_arguments = {}

        self._parse_arguments(les_arguments)

        self.atomwise: nn.Module = (
            Atomwise(
                n_layers=self.n_layers,
                n_hidden=self.n_hidden,
                add_linear_nn=self.add_linear_nn,
                output_scaling_factor=self.output_scaling_factor,
            )
            if self.use_atomwise
            else _DummyAtomwise()
        )

        if self.use_fixed_atomic_charges:
            self.fixed_charges = FixedCharges(normalization_factor=self.fixed_atomic_charges_scaling_factor)
        if self.use_atomic_alpha:
            self.atomic_alpha = AtomicAlpha()

        if self.is_periodic is not None:
            self.ewald = Ewald_vectorized( # torch.compile compatible
                sigma=self.sigma, # default 1.0
                dl=self.dl, # default 2.0
                is_periodic=self.is_periodic, # True: periodic, False: non-periodic
                N_max=self.N_max, # default 10, recommend N_max * dl > cell norm for accuracy
                remove_self_interaction=self.remove_self_interaction,
                use_epsilon_r_scaling=self.use_epsilon_r_scaling,
                )
        else:
            self.ewald = Ewald( # legacy module, not torch.compile compatible, supports mixed datasets
                sigma=self.sigma,
                dl=self.dl,
                remove_self_interaction=self.remove_self_interaction,
                use_epsilon_r_scaling=self.use_epsilon_r_scaling,
                )

        if self.use_qeq:
            if self.is_periodic is None:
                raise ValueError("use_qeq requires is_periodic to be True or False: charge "
                                 "equilibration needs the vectorized Ewald, which is also the "
                                 "only one that can be compiled or exported")
            if self.use_nqe:
                raise ValueError("use_qeq and use_nqe are mutually exclusive: charge "
                                 "equilibration already constrains the total charge through "
                                 "its Lagrange multiplier")
            if self.qeq_max_atoms_per_structure is not None and self.is_periodic:
                raise ValueError(
                    "qeq_max_atoms_per_structure is a non-periodic optimization: the "
                    "reciprocal path is dominated by its [n_atoms, K] structure factors, "
                    "not by A, so laying A out per structure would not pay there")
            self.qeq = Qeq(
                j_init=self.qeq_j_init,
                chi_positive=self.qeq_chi_positive,
                masked_charge_scope=(self.qeq_charge_scope == 'masked'),
                )
        if self.use_nqe:
            self.nqe = Nqe(eps=self.nqe_eps,
                           masked_charge_scope=(self.nqe_charge_scope == 'masked'))

        n_charge_models = sum([bool(self.use_qeq), bool(self.use_nqe),
                               bool(self.use_model0), bool(self.use_model0_a),
                               bool(self.use_model_b)])
        if n_charge_models > 1:
            raise ValueError("use_qeq, use_nqe, use_model0, use_model0_a and "
                             "use_model_b each replace the charge source, so at most "
                             "one may be enabled")
        if self.use_model0 or self.use_model0_a or self.use_model_b:
            if self.is_periodic is None:
                raise ValueError("the frontier models require is_periodic to be True or "
                                 "False: the legacy path decides the boundary condition "
                                 "per structure, which a single batch-wide coupling "
                                 "matrix cannot represent")
            frontier_args = dict(beta=self.frontier_beta,
                                 epsilon_e=self.frontier_epsilon_e,
                                 constrain_charge=self.frontier_constrain_charge,
                                 carrier_sign=self.frontier_carrier_sign)
            if self.use_model0:
                self.model0 = Model0(**frontier_args)
            elif self.use_model0_a:
                # Model 0-a always constrains the reference charge, so it takes no
                # constrain_charge argument
                self.model0_a = Model0A(
                    beta=self.frontier_beta, epsilon_e=self.frontier_epsilon_e,
                    carrier_sign=self.frontier_carrier_sign,
                    charge_switch_width=self.frontier_charge_switch_width)
            else:
                self.model_b = ModelB(**frontier_args)

        self.bec = BEC(
             remove_mean=self.remove_mean,
             epsilon_factor=self.epsilon_factor,
             )

    def _parse_arguments(self, les_arguments: Dict[str, Any]):
        """
        Parse arguments for LES model
        """
        self.n_layers = les_arguments.get('n_layers', 3)
        self.n_hidden = les_arguments.get('n_hidden', [32, 16])
        self.add_linear_nn = les_arguments.get('add_linear_nn', True)
        self.output_scaling_factor = les_arguments.get('output_scaling_factor', 0.1)

        self.sigma = les_arguments.get('sigma', 1.0)
        self.dl = les_arguments.get('dl', 2.0)
        self.remove_self_interaction = les_arguments.get('remove_self_interaction', True)


        self.is_periodic = les_arguments.get('is_periodic', None) # set is_periodic (True/False) to use the vectorized, torch.compile
        self.N_max = les_arguments.get('N_max', 10)

        self.remove_mean = les_arguments.get('remove_mean', True)
        self.epsilon_factor = les_arguments.get('epsilon_factor', 1.)
        self.use_atomwise = les_arguments.get('use_atomwise', False)
        self.use_fixed_atomic_charges = les_arguments.get('use_fixed_atomic_charges', False)
        self.fixed_atomic_charges_scaling_factor = les_arguments.get('fixed_atomic_charges_scaling_factor', 0.5)
        self.use_atomic_alpha = les_arguments.get('use_atomic_alpha', False)
        self.use_epsilon_r_scaling = les_arguments.get('use_epsilon_r_scaling', False)

        self.use_qeq = les_arguments.get('use_qeq', False)
        self.use_nqe = les_arguments.get('use_nqe', False)
        self.qeq_j_init = les_arguments.get('qeq_j_init', 1.0)
        self.qeq_chi_positive = les_arguments.get('qeq_chi_positive', True)
        self.qeq_self_interaction = les_arguments.get('qeq_self_interaction', True)
        self.nqe_eps = les_arguments.get('nqe_eps', 1e-6)
        # 'system': total_charge is the whole structure's charge (default)
        # 'masked': it is the charge of the masked atoms alone
        self.qeq_charge_scope = les_arguments.get('qeq_charge_scope', 'system')
        # set to the largest atom count of any single structure to lay the Qeq system out
        # per structure instead of as one flat block -- non-periodic only
        self.qeq_max_atoms_per_structure = les_arguments.get(
            'qeq_max_atoms_per_structure', None)
        self.nqe_charge_scope = les_arguments.get('nqe_charge_scope', 'system')

        self.use_model0 = les_arguments.get('use_model0', False)
        self.use_model0_a = les_arguments.get('use_model0_a', False)
        self.use_model_b = les_arguments.get('use_model_b', False)
        self.frontier_beta = les_arguments.get('frontier_beta', 1.0)
        # electronic screening relating LES charges to physical ones,
        # z = q_physical / sqrt(frontier_epsilon_e). Separate from epsilon_factor,
        # which is the high-frequency permittivity BEC uses.
        self.frontier_epsilon_e = les_arguments.get('frontier_epsilon_e', 1.0)
        self.frontier_constrain_charge = les_arguments.get(
            'frontier_constrain_charge', False)
        # -1 for an added electron, +1 for an added hole
        self.frontier_carrier_sign = les_arguments.get('frontier_carrier_sign', -1)
        # width in e over which Model 0-a blends the addition and removal heads
        self.frontier_charge_switch_width = les_arguments.get(
            'frontier_charge_switch_width', 0.01)
        for key, val in (('qeq_charge_scope', self.qeq_charge_scope),
                         ('nqe_charge_scope', self.nqe_charge_scope)):
            if val not in ('system', 'masked'):
                raise ValueError(f"{key} must be 'system' or 'masked', got {val!r}")

    def __setstate__(self, state: Dict[str, Any]):
        # Backward compatibility
        for key, default in {
            'use_atomwise': False,
            'use_fixed_atomic_charges': False,
            'use_atomic_alpha': False,
            'use_epsilon_r_scaling': False,
            'is_periodic': None,
            'N_max': 10,
            'use_qeq': False,
            'use_nqe': False,
            'qeq_self_interaction': True,
            'use_model0': False,
            'use_model0_a': False,
            'use_model_b': False,
        }.items():
            state.setdefault(key, default)
        super().__setstate__(state)

    def forward(self,
               positions: torch.Tensor, # [n_atoms, 3]
               cell: torch.Tensor, # [batch_size, 3, 3]
               e_ext: Optional[torch.Tensor]= None,
               desc: Optional[torch.Tensor]= None, # [n_atoms, n_features]
               latent_charges: Optional[torch.Tensor] = None, # [n_atoms, ]
               latent_dipoles: Optional[torch.Tensor] = None, # [n_atoms, 3]
               latent_quads: Optional[torch.Tensor] = None, # [n_atoms, 3, 3]
               latent_kappas: Optional[torch.Tensor] = None, # [n_atoms, ]
               latent_alphas: Optional[torch.Tensor] = None, # [n_atoms, ]
               atomic_numbers: Optional[torch.Tensor] = None, # [n_atoms, ]
               batch: Optional[torch.Tensor] = None,
               compute_energy: bool = True,
               compute_field: bool = False,
               compute_bec: bool = False,
               bec_output_index: Optional[int] = None, # option to compute BEC components along only one direction
               latent_chi: Optional[torch.Tensor] = None, # [n_atoms, ] electronegativity, for use_qeq
               latent_fukui: Optional[torch.Tensor] = None, # [n_atoms, ] Fukui weights, for use_nqe
               total_charge: Optional[torch.Tensor] = None, # [batch_size, ] defaults to neutral
               qeq_mask: Optional[torch.Tensor] = None, # [n_atoms, ] bool, True = equilibrate
               nqe_mask: Optional[torch.Tensor] = None, # [n_atoms, ] bool, True = absorbs the deficit
               latent_frontier: Optional[torch.Tensor] = None, # [n_atoms, ] epsilon, for use_model0
               latent_local_addition: Optional[torch.Tensor] = None, # [n_atoms, ] a, for use_model_b
               reference_total_charge: Optional[torch.Tensor] = None, # [batch_size, ] physical Q_ref
               carrier_count: Optional[torch.Tensor] = None, # [batch_size, ] m per structure
               ) -> Dict[str, Optional[torch.Tensor]]:
        # check the input shapes
        if batch is None:
            batch = torch.zeros(positions.shape[0], dtype=torch.int64, device=positions.device)

        if latent_charges is not None:
            # check the shape of latent charges
            assert latent_charges.shape[0] == positions.shape[0]
        elif desc is not None and latent_charges is None:
            if not self.use_atomwise:
                raise ValueError("desc must be provided and use_atomwise must be True if latent_charges is not provided")
            # compute the latent charges
            assert desc.shape[0] == positions.shape[0]
            latent_charges = self.atomwise(desc, batch)
        elif self.use_qeq and latent_chi is not None:
            # Qeq produces the charges itself; latent_charges is then only the prior kept by
            # atoms excluded from the equilibration, and neutral is the natural default
            latent_charges = torch.zeros(positions.shape[0], dtype=positions.dtype,
                                         device=positions.device)
        else:
            raise ValueError("Either desc or latent_charges must be provided")

        if atomic_numbers is not None and self.use_fixed_atomic_charges:
            # the baseline is one value per atom, so it needs a channel axis when the
            # charges carry one -- a bare + would broadcast [n_atoms, 1] against
            # [n_atoms] into [n_atoms, n_atoms] and corrupt the energy
            fixed = self.fixed_charges(atomic_numbers)
            latent_charges = latent_charges + (fixed if latent_charges.dim() == 1
                                               else fixed.unsqueeze(-1))

        if atomic_numbers is not None and self.use_atomic_alpha and latent_alphas is not None:
            baseline_alphas = self.atomic_alpha(atomic_numbers)
            #print(f'baseline_alphas: {baseline_alphas}')
            if latent_alphas.dim() == 1:
                latent_alphas = latent_alphas + baseline_alphas
            elif latent_alphas.dim() == 3:
                latent_alphas = latent_alphas + baseline_alphas[:,None,None] * torch.eye(3, device=baseline_alphas.device).unsqueeze(0) # [n_atoms, 3, 3]


        # Charge equilibration replaces the charge source, so it has to run before the
        # Ewald sum: the equilibrated charges are what the long-range energy is built from.
        qeq_energy: Optional[torch.Tensor] = None
        frontier_energy: Optional[torch.Tensor] = None
        frontier_occupation: Optional[torch.Tensor] = None
        frontier_reference_charges: Optional[torch.Tensor] = None
        frontier_reference_energy: Optional[torch.Tensor] = None
        frontier_addition_energy: Optional[torch.Tensor] = None
        frontier_occupation_add: Optional[torch.Tensor] = None
        frontier_occupation_remove: Optional[torch.Tensor] = None
        frontier_charge_deficit: Optional[torch.Tensor] = None
        frontier_removal_gate: Optional[torch.Tensor] = None
        if self.use_qeq and latent_chi is not None:
            if atomic_numbers is None:
                raise ValueError("atomic_numbers must be provided when use_qeq is True")
            if latent_kappas is not None:
                raise ValueError("use_qeq does not accept latent_kappas: the induced charge "
                                 "-kappa*phi is added after the solve and would break the "
                                 "total charge constraint. Qeq is already the self-consistent "
                                 "version of that response")
            assert latent_charges.numel() == positions.shape[0], \
                'use_qeq supports a single charge channel only'
            if total_charge is None:
                tot_q = torch.zeros(cell.shape[0], dtype=positions.dtype,
                                    device=positions.device)
            else:
                tot_q = total_charge
            if qeq_mask is None:
                mask = torch.ones(positions.shape[0], dtype=torch.bool,
                                  device=positions.device)
            else:
                mask = qeq_mask
            # the second return value of Qeq is the Lagrange multiplier, i.e. the
            # structure's chemical potential. Qeq exposes it; Les keeps its output
            # dict unchanged.
            n_pad = self.qeq_max_atoms_per_structure
            if n_pad is not None:
                slot, valid = batch_slots(batch, cell.shape[0], n_pad)
                A = self.ewald.compute_A_matrix_padded(positions, batch, slot, valid,
                                                       self.qeq_self_interaction)
                latent_charges, _ = self.qeq(A, latent_chi, atomic_numbers, batch, tot_q,
                                             latent_charges.reshape(-1), mask, slot, valid)
            else:
                slot, valid = None, None
                A = self.ewald.compute_A_matrix(positions, cell, batch,
                                                self.qeq_self_interaction)
                latent_charges, _ = self.qeq(A, latent_chi, atomic_numbers, batch,
                                             tot_q, latent_charges.reshape(-1), mask)
            # With monopoles alone the long-range energy is exactly 1/2 q^T A q, so the
            # matrix already built for the solve gives it and the Ewald pass can be
            # skipped. In a periodic cell that is the difference between building the
            # k-space terms once or twice. Needs A in the self_interaction=True
            # convention, which is what qeq_self_interaction selects.
            if (compute_energy and self.qeq_self_interaction
                    and latent_dipoles is None and latent_quads is None
                    and latent_alphas is None):
                if slot is not None and valid is not None:
                    qeq_energy = self.ewald.energy_from_A_padded(A, latent_charges,
                                                                 batch, slot, valid)
                else:
                    qeq_energy = self.ewald.energy_from_A(A, latent_charges, batch,
                                                          cell.shape[0])
        elif self.use_nqe and latent_fukui is not None:
            if latent_kappas is not None:
                raise ValueError("use_nqe does not accept latent_kappas: the induced charge "
                                 "-kappa*phi is added after the redistribution and would "
                                 "break the total charge it just enforced")
            assert latent_charges.numel() == positions.shape[0], \
                'use_nqe supports a single charge channel only'
            if total_charge is None:
                tot_q = torch.zeros(cell.shape[0], dtype=positions.dtype,
                                    device=positions.device)
            else:
                tot_q = total_charge
            latent_charges = self.nqe(latent_charges.reshape(-1), latent_fukui, batch,
                                      tot_q, nqe_mask)
        elif self.use_model0 or self.use_model0_a or self.use_model_b:
            # This first version covers monopoles and one carrier. Rather than quietly
            # ignoring a term it cannot place, it refuses the combination.
            if (latent_kappas is not None or latent_alphas is not None
                    or latent_dipoles is not None or latent_quads is not None
                    or e_ext is not None or compute_field or compute_bec):
                raise ValueError("the frontier models handle monopoles only for now: "
                                 "latent_kappas, latent_alphas, latent_dipoles, "
                                 "latent_quads, e_ext, compute_field and compute_bec "
                                 "are not supported with use_model0 / use_model_b")
            if carrier_count is None:
                raise ValueError("carrier_count is required by the frontier models: "
                                 "give 0 or 1 per structure")
            assert latent_charges.numel() == positions.shape[0], \
                'the frontier models support a single charge channel only'
            z_raw = latent_charges.reshape(-1)
            n_batch = cell.shape[0]
            if self.use_model0:
                if latent_frontier is None:
                    raise ValueError("latent_frontier is required when use_model0 is set")
                latent_charges, frontier_reference_charges, frontier_occupation = \
                    self.model0(z_raw, latent_frontier, batch, carrier_count, n_batch,
                                reference_total_charge)
            elif self.use_model0_a:
                if latent_frontier is None:
                    raise ValueError("latent_frontier is required when use_model0_a is "
                                     "set: [n_atoms, 2], addition score then removal")
                if reference_total_charge is None:
                    raise ValueError("reference_total_charge is required by use_model0_a, "
                                     "which always constrains the reference charge; "
                                     "neutrality is not assumed on your behalf")
                (latent_charges, frontier_reference_charges, frontier_occupation,
                 frontier_occupation_add, frontier_occupation_remove,
                 frontier_charge_deficit, frontier_removal_gate) = self.model0_a(
                    z_raw, latent_frontier, batch, carrier_count, n_batch,
                    reference_total_charge)
            else:
                if latent_local_addition is None:
                    raise ValueError("latent_local_addition is required when "
                                     "use_model_b is set")
                # the matrix whose quadratic form is this module's own monopole energy,
                # reused for every candidate's electrostatic cost
                K = self.ewald.compute_A_matrix(positions, cell, batch,
                                                not self.remove_self_interaction)
                latent_charges, frontier_reference_charges, frontier_occupation, \
                    smooth_min = self.model_b(z_raw, latent_local_addition, K, batch,
                                              carrier_count, n_batch,
                                              reference_total_charge)
                if compute_energy:
                    onehot = segment_onehot(batch, n_batch, positions.dtype)
                    # the candidates' electrostatic cost already sits inside the smooth
                    # minimum, so the reference charges -- not the reported ones -- are
                    # what the electrostatic term is built from
                    frontier_reference_energy = quadratic_energy(
                        K, frontier_reference_charges, onehot)
                    frontier_addition_energy = (
                        carrier_count.reshape(-1).to(dtype=positions.dtype) * smooth_min)
                    frontier_energy = (frontier_reference_energy
                                       + frontier_addition_energy)

        # compute the long-range interactions
        if frontier_energy is not None:
            E_lr, q_induced, u_induced = frontier_energy, None, None
        elif qeq_energy is not None:
            E_lr, q_induced, u_induced = qeq_energy, None, None
        elif compute_energy:
            E_lr, q_induced, u_induced = self.ewald(q=latent_charges,
                              u=latent_dipoles,
                              kappa=latent_kappas,
                              alpha=latent_alphas,
                              quad=latent_quads,
                              r=positions,
                              cell=cell,
                              batch=batch,
                              compute_field=compute_field,
                              e_ext = e_ext,
                              )
        else:
            E_lr, q_induced, u_induced = None, None, None

        if latent_alphas is not None and u_induced is not None:
            if latent_dipoles is not None:
                if latent_dipoles.dim() == 2 and u_induced.dim() == 3:
                    latent_dipoles = latent_dipoles.unsqueeze(1) # [n_node, 1, 3]
                assert latent_dipoles.shape == u_induced.shape, f'latent_dipoles dimension error'
                latent_dipoles = latent_dipoles + u_induced
            else:
                latent_dipoles = u_induced

        if latent_kappas is not None and q_induced is not None:
            # reshape, not a bare +: the vectorized Ewald returns [n_atoms, n_q] while a
            # caller may have passed [n_atoms], and [N] + [N, 1] broadcasts to [N, N]
            latent_charges = latent_charges + q_induced.reshape(latent_charges.shape)

        # compute the BEC
        if compute_bec:
            bec = self.bec(q=latent_charges,
                           u=latent_dipoles,
                           r=positions,
                           cell=cell,
                           batch=batch,
                           output_index=bec_output_index,
		           )
        else:
            bec = None

        output = {
            'E_lr': E_lr,
            'latent_charges': latent_charges,
            'latent_dipoles': latent_dipoles,
            'latent_quads': latent_quads,
            'latent_alphas': latent_alphas,
            'BEC': bec,
            }
        if self.use_model0 or self.use_model0_a or self.use_model_b:
            output['occupation'] = frontier_occupation
            output['reference_charges'] = frontier_reference_charges
        if self.use_model0_a:
            output['occupation_add'] = frontier_occupation_add
            output['occupation_remove'] = frontier_occupation_remove
            output['reference_charge_deficit'] = frontier_charge_deficit
            output['removal_gate'] = frontier_removal_gate
        if self.use_model_b:
            # E_lr already sums these two; they are returned for diagnosis, not to be
            # added again
            output['E_reference_lr'] = frontier_reference_energy
            output['E_frontier'] = frontier_addition_energy
        return output

class _DummyAtomwise(nn.Module):
    def forward(self, desc: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        raise ValueError("set use_atomwise to True to use Atomwise module")
