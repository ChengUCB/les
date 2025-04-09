import torch
from torch import nn
from typing import Dict, Any, Union

from .module import (
    Atomwise,
    Ewald,
    BEC
)

__all__ = ['Les']

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

        self.atomwise = Atomwise(
        n_layers=self.n_layers,
        n_hidden=self.n_hidden,
        add_linear_nn=self.add_linear_nn,
        output_scaling_factor=self.output_scaling_factor, 
    )

        self.ewald = Ewald(
            sigma=self.sigma,
            dl=self.dl
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

        self.remove_mean = les_arguments.get('remove_mean', True)
        self.epsilon_factor = les_arguments.get('epsilon_factor', 1.)

    def forward(self, 
               desc: torch.Tensor, # [n_atoms, n_features]
               positions: torch.Tensor, # [n_atoms, 3]
               cell: torch.Tensor, # [batch_size, 3, 3]
               batch: torch.Tensor = None,
               compute_energy: bool = True,
               compute_bec: bool = False,
               bec_output_index: int = None, # option to compute BEC components along only one direction
               ) -> torch.Tensor:
        """
        arguments:
        desc: torch.Tensor
        Descriptors for the atoms. Shape: (n_atoms, n_features)
        positions: torch.Tensor
            positions of the atoms. Shape: (n_atoms, 3)
        cell: torch.Tensor
            cell of the system. Shape: (batch_size, 3, 3)
        batch: torch.Tensor
            batch of the system. Shape: (n_atoms,)
        """
        # check the input shapes
        assert desc.shape[0] == positions.shape[0]
        if batch == None:
            batch = torch.zeros(desc.shape[0], dtype=torch.int64, device=desc.device)

        # compute the latent charges
        latent_charges = self.atomwise(desc, batch)

        # compute the long-range interactions
        if compute_energy:
            E_lr = self.ewald(q=latent_charges,
                              r=positions,
                              cell=cell,
                              batch=batch,
                              )

        # compute the BEC
        if compute_bec:
            bec = self.bec(q=latent_charges,
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
            'BEC': bec,
            }
        return output 
