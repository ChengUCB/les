import torch
from torch import nn
from typing import Dict, Any, Union, Optional, Tuple

from .module import (
    Atomwise,
    Ewald,
    Ewald_vectorized,
    BEC,
    FixedCharges,
    AtomicAlpha,
    Qeq,
)
from .util import grad

__all__ = ['Les']

class Les(nn.Module):

    __constants__ = ['use_fixed_atomic_charges', 'use_atomic_alpha', 'use_atomwise', 'use_epsilon_r_scaling',
                     'charge_offset_mode', 'use_qeq']
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

        self.bec = BEC(
             remove_mean=self.remove_mean,
             epsilon_factor=self.epsilon_factor,
             )

        if self.use_qeq:
            self.qeq = Qeq(hardness=self.qeq_hardness, hardness_trainable=self.qeq_hardness_trainable,
                           chi=self.qeq_chi, chi_trainable=self.qeq_chi_trainable)
            # atomic numbers that are equilibrated; empty: every atom
            self.register_buffer('qeq_z', torch.tensor(self.qeq_atomic_numbers or [], dtype=torch.long),
                                 persistent=False)
            # is E_lr, as a function of the equilibrated charges, the energy they minimise?
            self.qeq_variational = ((self.qeq_siepmann_sprik or self.qeq_onsite_energy)
                                    and (self.qeq_at_sites or not self.charge_offset_mode))
            # qeq_cache: inverse matrix of the last call and what it was built from
            self.register_buffer('qeq_cache_inverse', torch.zeros(0, 0), persistent=False)
            self.register_buffer('qeq_cache_key', torch.zeros(0), persistent=False)

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

        # charge offsets: the multipoles of atom i sit at r_i + delta_i, delta_i from latent_offsets
        self.charge_offset_mode = les_arguments.get('charge_offset_mode', False)
        charge_offset_max = les_arguments.get('charge_offset_max', 1.0) # soft cap on |delta| (Angstrom); 0 or None: no cap
        self.charge_offset_max = float(charge_offset_max) if charge_offset_max is not None else 0.0
        # elements whose atoms carry an offset, e.g. [78]; None: all atoms
        self.charge_offset_z = torch.tensor(les_arguments.get('charge_offset_atomic_numbers', None) or [], dtype=torch.long)
        # set an axis n, e.g. [0, 0, 1], to keep only the component of latent_offsets along it: delta = (v . n) n.
        # With use_qeq this applies to the equilibrated atoms (the slab); the other atoms keep their 3D offsets
        charge_offset_axis = les_arguments.get('charge_offset_axis', None)
        self.charge_offset_axis = [float(x) for x in charge_offset_axis] if charge_offset_axis is not None else [0.0, 0.0, 0.0]

        # charge equilibration (Qeq): the charges of the selected atoms minimise the electrostatic energy
        # plus chi q + 1/2 J q^2, the other atoms keep their latent charges
        self.use_qeq = les_arguments.get('use_qeq', False)
        self.qeq_siepmann_sprik = les_arguments.get('qeq_siepmann_sprik', False) # Siepmann-Sprik electrode: chi = 0, J = 0
        self.qeq_atomic_numbers = les_arguments.get('qeq_atomic_numbers', None) # elements to equilibrate; None: all
        self.qeq_hardness = les_arguments.get('qeq_hardness', 1.0) # J (eV/e^2)
        self.qeq_hardness_trainable = les_arguments.get('qeq_hardness_trainable', True) # one J per element; False: constant
        # electronegativity chi (V) per element: a number, or {atomic number: value} (0 for the others); with
        # qeq_chi_trainable one trainable value per element starting from it. latent_chi adds an atom-wise part
        self.qeq_chi = les_arguments.get('qeq_chi', None)
        self.qeq_chi_trainable = les_arguments.get('qeq_chi_trainable', False)
        # total_charge is that of the 'system' (the equilibrated atoms compensate the other charges) or of the
        # 'selected' atoms alone. Default: 'system'; with qeq_siepmann_sprik 'selected' (the electrode stays neutral)
        self.qeq_charge_scope = les_arguments.get('qeq_charge_scope', 'selected' if self.qeq_siepmann_sprik else 'system')
        self.qeq_onsite_energy = les_arguments.get('qeq_onsite_energy', True) # add chi q + 1/2 J q^2 to E_lr
        self.qeq_at_sites = les_arguments.get('qeq_at_sites', True) # with offsets: solve at r + delta; False: at r
        # in eval mode, reuse the inverse matrix while the equilibrated atoms, the cell and the hardness are
        # unchanged (MD with a fixed electrode).
        self.qeq_cache = les_arguments.get('qeq_cache', False)

    def __setstate__(self, state: Dict[str, Any]):
        # Backward compatibility
        for key, default in {
            'use_atomwise': False,
            'use_fixed_atomic_charges': False,
            'use_atomic_alpha': False,
            'use_epsilon_r_scaling': False,
            'is_periodic': None,
            'N_max': 10,
            'charge_offset_mode': False,
            'charge_offset_max': 1.0,
            'charge_offset_axis': [0.0, 0.0, 0.0],
            'charge_offset_z': torch.zeros(0, dtype=torch.long),
            'use_qeq': False,
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
               latent_offsets: Optional[torch.Tensor] = None, # [n_atoms, 3]
               latent_chi: Optional[torch.Tensor] = None, # [n_atoms, ] electronegativity (V), for use_qeq
               latent_hardness: Optional[torch.Tensor] = None, # [n_atoms, ] atom-wise hardness before softplus, for use_qeq
               total_charge: Optional[torch.Tensor] = None, # [batch_size, ] (e), for use_qeq; None: neutral
               qeq_mask: Optional[torch.Tensor] = None, # [n_atoms, ] bool, atoms to equilibrate; None: all
               applied_potential: Optional[torch.Tensor] = None, # [n_atoms, ] (V) at the nuclei, for use_qeq: energy -sum_i v_i q_i; with offsets, e_ext (V/A) moves it to the sites
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
        elif self.use_qeq:
            # the charges of the atoms that are not equilibrated
            latent_charges = torch.zeros(positions.shape[0], dtype=positions.dtype, device=positions.device)
        else:
            raise ValueError("Either desc or latent_charges must be provided")

        if atomic_numbers is not None and self.use_fixed_atomic_charges:
            latent_charges = latent_charges + self.fixed_charges(atomic_numbers)

        if atomic_numbers is not None and self.use_atomic_alpha and latent_alphas is not None:
            baseline_alphas = self.atomic_alpha(atomic_numbers)
            #print(f'baseline_alphas: {baseline_alphas}')
            if latent_alphas.dim() == 1:
                latent_alphas = latent_alphas + baseline_alphas
            elif latent_alphas.dim() == 3:
                latent_alphas = latent_alphas + baseline_alphas[:,None,None] * torch.eye(3, device=baseline_alphas.device).unsqueeze(0) # [n_atoms, 3, 3]

        # sites of the latent multipoles, r + delta with charge offsets: built in the autograd graph, never wrapped
        sites = positions
        offsets: Optional[torch.Tensor] = None
        if self.charge_offset_mode:
            if latent_offsets is None:
                raise ValueError("charge_offset_mode is True, so latent_offsets must be provided")
            latent_offsets = latent_offsets.to(dtype=positions.dtype)
            assert latent_offsets.shape == positions.shape, 'latent_offsets must be [n_atoms, 3]'
            if self.charge_offset_z.numel() > 0:
                if atomic_numbers is None:
                    raise ValueError("atomic_numbers must be provided when charge_offset_atomic_numbers is set")
                carries = torch.isin(atomic_numbers, self.charge_offset_z.to(atomic_numbers.device))
                latent_offsets = torch.where(carries.unsqueeze(1), latent_offsets, torch.zeros_like(latent_offsets))
            if self.charge_offset_axis != [0.0, 0.0, 0.0]:
                axis = torch.tensor(self.charge_offset_axis, dtype=positions.dtype, device=positions.device)
                axis = axis / axis.norm()
                projected = (latent_offsets @ axis).unsqueeze(1) * axis.unsqueeze(0)
                if self.use_qeq:
                    on_slab = self._qeq_selected(atomic_numbers, qeq_mask, positions.shape[0], positions.device)
                    latent_offsets = torch.where(on_slab.unsqueeze(1), projected, latent_offsets)
                else:
                    latent_offsets = projected
            offsets = self._cap_offsets(latent_offsets)
            sites = positions + offsets
        elif latent_offsets is not None:
            raise ValueError("latent_offsets given, but charge_offset_mode is False")

        # charge equilibration replaces the charges of the selected atoms, before the Ewald sum
        qeq_lambda: Optional[torch.Tensor] = None
        qeq_energy: Optional[torch.Tensor] = None
        if self.use_qeq:
            if latent_kappas is not None or latent_alphas is not None:
                raise ValueError("use_qeq does not support latent_kappas and latent_alphas")
            if applied_potential is not None and offsets is not None and e_ext is not None:
                # a displaced charge feels the applied potential at its site: v(r + delta) = v(r) + e_ext . delta
                applied_potential = applied_potential.reshape(-1).to(positions.dtype) + (offsets * e_ext.to(positions.dtype).reshape(1, 3)).sum(dim=-1)
            latent_charges, qeq_lambda, qeq_energy = self._equilibrate(
                latent_charges, positions, sites, cell, batch, atomic_numbers, latent_dipoles, latent_quads,
                latent_chi, latent_hardness, total_charge, qeq_mask, applied_potential)

        # compute the long-range interactions
        if compute_energy:
            E_lr, q_induced, u_induced = self.ewald(q=latent_charges,
                              u=latent_dipoles,
                              kappa=latent_kappas,
                              alpha=latent_alphas,
                              quad=latent_quads,
                              r=sites,
                              cell=cell,
                              batch=batch,
                              compute_field=compute_field,
                              e_ext = e_ext,
                              )
        else:
            E_lr, q_induced, u_induced = None, None, None
        if E_lr is not None and qeq_energy is not None:
            E_lr = E_lr + qeq_energy

        if latent_alphas is not None and u_induced is not None:
            if latent_dipoles is not None:
                if latent_dipoles.dim() == 2 and u_induced.dim() == 3:
                    latent_dipoles = latent_dipoles.unsqueeze(1) # [n_node, 1, 3]
                assert latent_dipoles.shape == u_induced.shape, f'latent_dipoles dimension error'
                latent_dipoles = latent_dipoles + u_induced
            else:
                latent_dipoles = u_induced

        if latent_kappas is not None and q_induced is not None:
            latent_charges = latent_charges + q_induced

        # compute the BEC
        if compute_bec:
            bec = self.bec(q=latent_charges,
                           u=latent_dipoles,
                           r=positions,
                           cell=cell,
                           batch=batch,
                           output_index=bec_output_index,
                           r_q=sites,
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
        if offsets is not None:
            output['latent_offsets'] = offsets
        if qeq_lambda is not None:
            output['qeq_lambda'] = qeq_lambda
        return output

    def _cap_offsets(self, v: torch.Tensor) -> torch.Tensor:
        # delta = v / sqrt(1 + |v|^2 / d^2), smooth at v = 0; pow, not rsqrt, for AOTInductor export
        if self.charge_offset_max > 0.0:
            return v * (1.0 + (v * v).sum(dim=-1, keepdim=True) / self.charge_offset_max ** 2).pow(-0.5)
        return v

    def _qeq_selected(self, atomic_numbers: Optional[torch.Tensor], qeq_mask: Optional[torch.Tensor],
                      n: int, device: torch.device) -> torch.Tensor:
        # the atoms that are equilibrated: those of qeq_atomic_numbers (all if not set), restricted by qeq_mask
        if atomic_numbers is None:
            raise ValueError("atomic_numbers must be provided when use_qeq is True")
        selected = torch.ones(n, dtype=torch.bool, device=device)
        if self.qeq_z.numel() > 0:
            selected = torch.isin(atomic_numbers, self.qeq_z)
        if qeq_mask is not None:
            selected = selected & qeq_mask
        return selected

    def _equilibrate(self,
                     latent_charges: torch.Tensor,
                     positions: torch.Tensor,
                     sites: torch.Tensor,
                     cell: torch.Tensor,
                     batch: torch.Tensor,
                     atomic_numbers: Optional[torch.Tensor],
                     latent_dipoles: Optional[torch.Tensor],
                     latent_quads: Optional[torch.Tensor],
                     latent_chi: Optional[torch.Tensor],
                     latent_hardness: Optional[torch.Tensor],
                     total_charge: Optional[torch.Tensor],
                     qeq_mask: Optional[torch.Tensor],
                     applied_potential: Optional[torch.Tensor],
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Charges of the selected atoms from Qeq. Returns all charges, lambda [batch_size] and the
        # energy [batch_size] to add to the Ewald energy of these charges.
        n = positions.shape[0]
        n_batch = cell.shape[0]
        dtype = positions.dtype
        device = positions.device
        assert latent_charges.numel() == n, 'use_qeq supports one charge channel'
        if atomic_numbers is None:
            raise ValueError("atomic_numbers must be provided when use_qeq is True")

        selected = self._qeq_selected(atomic_numbers, qeq_mask, n, device)
        batch_s = batch[selected]

        # with offsets, the selected atoms are equilibrated at their sites, or at their nuclei
        r_solve = sites if self.qeq_at_sites else torch.where(selected.unsqueeze(1), positions, sites)

        # potential on every atom of all that is not equilibrated: dE/dq at zero equilibrated charge
        q_fixed = torch.where(selected, torch.zeros_like(positions[:, 0]), latent_charges.reshape(-1).to(dtype))
        probe = positions[:, 0] * 0.0   # in the graph of the positions: torch.compile cannot create a leaf
        if not probe.requires_grad:
            probe = probe.requires_grad_(True)
        E_fixed = self.ewald(q=q_fixed + probe, u=latent_dipoles, quad=latent_quads,
                             r=r_solve, cell=cell, batch=batch)[0]
        phi = grad(y=E_fixed, x=probe)[selected]

        chi = torch.zeros_like(phi)
        J = torch.zeros_like(phi)
        if not self.qeq_siepmann_sprik:
            J = self.qeq.hardness(atomic_numbers[selected],
                                  latent_hardness[selected] if latent_hardness is not None else None).to(dtype)
            chi = self.qeq.electronegativity(atomic_numbers[selected],
                                             latent_chi.reshape(-1)[selected] if latent_chi is not None else None).to(dtype)
        b = chi + phi
        if applied_potential is not None:
            b = b - applied_potential.reshape(-1)[selected].to(dtype)

        if total_charge is not None:
            Q = total_charge.reshape(-1).to(dtype)
        else:
            Q = torch.zeros(n_batch, dtype=dtype, device=device)
        if self.qeq_charge_scope == 'system':
            Q = Q - torch.zeros_like(Q).index_add(0, batch, q_fixed)

        use_cache = self.qeq_cache and not self.training
        key = torch.cat([r_solve[selected].reshape(-1), cell.reshape(-1).to(dtype), J, batch_s.to(dtype)]).detach()
        if use_cache and key.shape == self.qeq_cache_key.shape and torch.equal(key, self.qeq_cache_key):
            q_selected, qeq_lambda = self.qeq.reuse(self.qeq_cache_inverse, b, batch_s, Q)
        else:
            A = self.ewald.compute_A_matrix(r_solve[selected], cell, batch_s)
            q_selected, qeq_lambda, K_inv = self.qeq(A, b, J, batch_s, Q, self.qeq_variational or use_cache)
            if use_cache:
                self.qeq_cache_inverse = K_inv
                self.qeq_cache_key = key
        q = q_fixed.clone()
        q[selected] = q_selected

        # the Gaussian self energy of an equilibrated charge is part of its hardness: it stays in the energy
        energy = torch.zeros_like(q_selected)
        if self.remove_self_interaction:
            energy = energy + q_selected ** 2 * (self.ewald.norm_factor / (self.sigma * (2.0 * torch.pi) ** 1.5))
        if self.qeq_onsite_energy:
            energy = energy + chi * q_selected + 0.5 * J * q_selected ** 2
        qeq_energy = torch.zeros_like(Q).index_add(0, batch_s, energy)
        if applied_potential is not None:
            qeq_energy = qeq_energy - torch.zeros_like(Q).index_add(0, batch, applied_potential.reshape(-1).to(dtype) * q)
        return q.reshape(latent_charges.shape), qeq_lambda, qeq_energy

class _DummyAtomwise(nn.Module):
    def forward(self, desc: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        raise ValueError("set use_atomwise to True to use Atomwise module")
