import torch
from torch import nn
from typing import Dict, Any, Union, Optional

from .module import (
    Atomwise,
    Ewald,
    BEC,
    FixedCharges,
)

__all__ = ['Les','LesVector']

class Les(nn.Module):

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

        self.fixed_charges = FixedCharges()

        self.ewald = Ewald(
            sigma=self.sigma,
            dl=self.dl,
            remove_self_interaction=self.remove_self_interaction,
            )

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

        self.remove_mean = les_arguments.get('remove_mean', True)
        self.epsilon_factor = les_arguments.get('epsilon_factor', 1.)
        self.use_atomwise = les_arguments.get('use_atomwise', False)
        self.use_fixed_charges = les_arguments.get('use_fixed_charges', True)

    def forward(self, 
               positions: torch.Tensor, # [n_atoms, 3]
               cell: torch.Tensor, # [batch_size, 3, 3]
               e_ext: Optional[torch.Tensor]= None, # [n_atoms, n_features]
               desc: Optional[torch.Tensor]= None, # [n_atoms, n_features]
               latent_charges: Optional[torch.Tensor] = None, # [n_atoms, ]
               latent_dipoles: Optional[torch.Tensor] = None, # [n_atoms, 3]
               latent_kappas: Optional[torch.Tensor] = None, # [n_atoms, ]
               latent_alphas: Optional[torch.Tensor] = None, # [n_atoms, ]
               atomic_numbers: Optional[torch.Tensor] = None, # [n_atoms, ]
               batch: Optional[torch.Tensor] = None,
               compute_energy: bool = True,
               compute_field: bool = False,
               compute_bec: bool = False,
               bec_output_index: Optional[int] = None, # option to compute BEC components along only one direction
               ) -> Dict[str, Optional[torch.Tensor]]:
        """
        arguments:
        desc: torch.Tensor
        Descriptors for the atoms. Shape: (n_atoms, n_features)
        latent_charges: torch.Tensor
        One can also directly input the latent charges. Shape: (n_atoms, )
        positions: torch.Tensor
            positions of the atoms. Shape: (n_atoms, 3)
        cell: torch.Tensor
            cell of the system. Shape: (batch_size, 3, 3)
        batch: torch.Tensor
            batch of the system. Shape: (n_atoms,)
        """
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
        else:
            raise ValueError("Either desc or latent_charges must be provided")

        if atomic_numbers is not None and self.use_fixed_charges:
            latent_charges = latent_charges + self.fixed_charges(atomic_numbers)[:,None]

        # compute the long-range interactions
        if compute_energy:
            # print("LES:",latent_charges.shape,latent_dipoles.shape,latent_kappas.shape,latent_alphas.shape)
            ewald_out = self.ewald(q=latent_charges,
                              u=latent_dipoles,
                              e_ext=e_ext,
                              kappa=latent_kappas,
                              alpha=latent_alphas,
                              r=positions,
                              cell=cell,
                              batch=batch,
                              compute_field=compute_field,
                              )
        else:
            ewald_out = {"E_lr": None, "q_induced": None, "u_induced": None}
        E_lr = ewald_out["E_lr"]
        q_induced = ewald_out["q_induced"]
        u_induced = ewald_out["u_induced"]

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
		           )
        else:
            bec = None

        output = {
            'E_lr': E_lr,
            'latent_charges': latent_charges,
            'latent_dipoles': latent_dipoles,
            'q_induced': q_induced,
            'u_induced': u_induced,
            'BEC': bec,
            }
        return output 

class _DummyAtomwise(nn.Module):
    def forward(self, desc: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        raise ValueError("set use_atomwise to True to use Atomwise module")

class LesVector(nn.Module):

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

        self.fixed_charges = FixedCharges()

        self.ewald = Ewald(
            sigma=self.sigma,
            dl=self.dl,
            remove_self_interaction=self.remove_self_interaction,
            )

        self.bec = BEC(
             remove_mean=self.remove_mean,
             epsilon_factor=self.epsilon_factor,
             )

    def _parse_arguments(self, les_arguments: Dict[str, Any]):
        """
        Parse arguments for LES model
        """
        self.output_scaling_factor = les_arguments.get('output_scaling_factor', 0.1)

        self.sigma = les_arguments.get('sigma', 1.0)
        self.dl = les_arguments.get('dl', 2.0)
        self.remove_self_interaction = les_arguments.get('remove_self_interaction', True)

        self.remove_mean = les_arguments.get('remove_mean', True)
        self.epsilon_factor = les_arguments.get('epsilon_factor', 1.)
        self.use_fixed_charges = les_arguments.get('use_fixed_charges', True)

    def forward(self, 
               positions: torch.Tensor, # [n_atoms, 3]
               cell: torch.Tensor, # [batch_size, 3, 3]
               latent_charges: torch.Tensor, # [n_atoms, ]
               e_ext : torch.Tensor, #[3]
               n_scf: int = 0,
               latent_dipoles: Optional[torch.Tensor] = None, # [n_atoms, 3]
               latent_kappas: Optional[torch.Tensor] = None, # [n_atoms, 3]
               latent_alphas: Optional[torch.Tensor] = None, # [n_atoms, 3, 3]
               atomic_numbers: Optional[torch.Tensor] = None, # [n_atoms, ]
               batch: Optional[torch.Tensor] = None,
               compute_energy: bool = True,
               compute_field: bool = False,
               compute_bec: bool = False,
               bec_output_index: Optional[int] = None, # option to compute BEC components along only one direction
               ) -> Dict[str, Optional[torch.Tensor]]:
        """
        arguments:
        desc: torch.Tensor
        Descriptors for the atoms. Shape: (n_atoms, n_features)
        latent_charges: torch.Tensor
        One can also directly input the latent charges. Shape: (n_atoms, )
        positions: torch.Tensor
            positions of the atoms. Shape: (n_atoms, 3)
        cell: torch.Tensor
            cell of the system. Shape: (batch_size, 3, 3)
        batch: torch.Tensor
            batch of the system. Shape: (n_atoms,)
        """
        # check the input shapes
        if batch is None:
            batch = torch.zeros(positions.shape[0], dtype=torch.int64, device=positions.device)
            
        assert latent_charges.shape[0] == positions.shape[0]

        if atomic_numbers is not None and self.use_fixed_charges:
            latent_charges = latent_charges + self.fixed_charges(atomic_numbers)

        # compute the long-range interactions
        # different kappa/alpha for each iteration
        for i in range(n_scf+1):
            ewald_out = self.ewald(q=latent_charges[:,None],
                                r=positions,
                                e_ext=e_ext,
                                cell=cell,
                                batch=batch,
                                u=latent_dipoles[:,None,:],
                                compute_field=True,
                                )
            efield = ewald_out["field"].squeeze() + e_ext[None,:]
            q_induced = torch.einsum("ni,ni->n",latent_kappas[:,i,:],efield)
            mu_induced = torch.einsum("nij,nj->ni",latent_alphas[:,i,:,:],efield)
            latent_charges = latent_charges + q_induced
            latent_dipoles = latent_dipoles + mu_induced

        ewald_out = self.ewald(q=latent_charges,
                            r=positions,
                            e_ext=e_ext,
                            cell=cell,
                            batch=batch,
                            u=latent_dipoles,
                            compute_field=True,
                            )

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
                 'E_lr': ewald_out["E_lr"],
                 'E_ext': ewald_out["E_ext"],
                 'BEC':bec,
                 }

        return output 

class _DummyAtomwise(nn.Module):
    def forward(self, desc: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        raise ValueError("set use_atomwise to True to use Atomwise module")