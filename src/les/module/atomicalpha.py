import torch
import torch.nn as nn
from typing import Dict

__all__ = ['AtomicAlpha']

# in atomic units (Bohr radius^3)
alpha_dict = {
    1: 4.50,   # H
    2: 1.38,   # He
    3: 164.11, # Li
    4: 37.74,  # Be
    5: 20.50,  # B
    6: 11.30,  # C
    7: 7.40,   # N
    8: 5.30,   # O
    9: 3.74,   # F
    10: 2.67,  # Ne
    11: 162.70,# Na
    12: 71.30, # Mg
    13: 60.00, # Al
    14: 37.30, # Si
    15: 25.00, # P
    16: 19.40, # S
    17: 14.60, # Cl
    18: 11.10, # Ar
    19: 290.00,# K
    20: 169.00,# Ca
    21: 120.00,# Sc
    22: 98.00, # Ti
    23: 84.00, # V
    24: 78.00, # Cr
    25: 63.00, # Mn
    26: 56.00, # Fe
    27: 50.00, # Co
    28: 49.00, # Ni
    29: 47.00, # Cu
    30: 38.70, # Zn
    31: 50.00, # Ga
    32: 40.00, # Ge
    33: 30.00, # As
    34: 28.90, # Se
    35: 21.90, # Br
    36: 16.80, # Kr
    37: 319.00,# Rb
    38: 197.00,# Sr
    39: 162.00,# Y
    40: 121.00,# Zr
    41: 106.00,# Nb
    42: 86.40, # Mo
    43: 80.00, # Tc
    44: 65.00, # Ru
    45: 58.00, # Rh
    46: 26.10, # Pd
    47: 55.00, # Ag
    48: 49.70, # Cd
    49: 70.00, # In
    50: 52.00, # Sn
    51: 43.00, # Sb
    52: 37.60, # Te
    53: 35.00, # I
    54: 27.30, # Xe
    55: 401.00,# Cs
    56: 273.00,# Ba
    57: 213.00,# La
    58: 204.00,# Ce
    59: 196.00,# Pr
    60: 190.00,# Nd
    61: 185.00,# Pm
    62: 180.00,# Sm
    63: 175.00,# Eu
    64: 160.00,# Gd
    65: 159.00,# Tb
    66: 157.00,# Dy
    67: 156.00,# Ho
    68: 153.00,# Er
    69: 151.00,# Tm
    70: 142.00,# Yb
    71: 148.00,# Lu
    72: 109.00,# Hf
    73: 88.00, # Ta
    74: 74.00, # W
    75: 65.00, # Re
    76: 57.00, # Os
    77: 51.00, # Ir
    78: 39.70, # Pt
    79: 36.00, # Au
    80: 33.90, # Hg
    81: 50.00, # Tl
    82: 47.00, # Pb
    83: 48.00, # Bi
    84: 45.00, # Po
    85: 38.00, # At
    86: 33.00, # Rn
}

class AtomicAlpha(nn.Module):
    def __init__(self,
                 alpha_dict: Dict[int, float] = alpha_dict,
                 normalization_factor: float = 0.1481847/14.3996, # bohr^3 -> e*A/(V/A)
                 ):
        super().__init__()
        self.alpha_dict = alpha_dict
        self.normalization_factor = normalization_factor

    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        alpha = torch.tensor([self.alpha_dict[atomic_number.item()] for atomic_number in atomic_numbers], device=atomic_numbers.device)
        return alpha * self.normalization_factor
