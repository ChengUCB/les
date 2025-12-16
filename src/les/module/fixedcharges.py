import torch
import torch.nn as nn
from typing import Dict

__all__ = ['FixedCharges']

typical_charge = {
     1: +1,   # H
     2:  0,   # He
     3: +1,   # Li
     4: +2,   # Be
     5: +3,   # B
     6: +4,   # C
     7: -3,   # N
     8: -2,   # O
     9: -1,   # F
    10:  0,   # Ne
    11: +1,   # Na
    12: +2,   # Mg
    13: +3,   # Al
    14: +4,   # Si
    15: +5,   # P
    16: -2,   # S
    17: -1,   # Cl
    18:  0,   # Ar
    19: +1,   # K
    20: +2,   # Ca
    21: +3,   # Sc
    22: +4,   # Ti
    23: +5,   # V
    24: +3,   # Cr
    25: +2,   # Mn
    26: +2,   # Fe
    27: +2,   # Co
    28: +2,   # Ni
    29: +1,   # Cu
    30: +2,   # Zn
    31: +3,   # Ga
    32: +4,   # Ge
    33: +5,   # As
    34: -2,   # Se
    35: -1,   # Br
    36:  0,   # Kr
    37: +1,   # Rb
    38: +2,   # Sr
    39: +3,   # Y
    40: +4,   # Zr
    41: +5,   # Nb
    42: +6,   # Mo
    43: +7,   # Tc
    44: +3,   # Ru
    45: +3,   # Rh
    46: +2,   # Pd
    47: +1,   # Ag
    48: +2,   # Cd
    49: +3,   # In
    50: +2,   # Sn
    51: +3,   # Sb
    52: -2,   # Te
    53: -1,   # I
    54:  0,   # Xe
    55: +1,   # Cs
    56: +2,   # Ba
    57: +3,   # La
    58: +3,   # Ce
    59: +3,   # Pr
    60: +3,   # Nd
    61: +3,   # Pm
    62: +3,   # Sm
    63: +2,   # Eu
    64: +3,   # Gd
    65: +3,   # Tb
    66: +3,   # Dy
    67: +3,   # Ho
    68: +3,   # Er
    69: +3,   # Tm
    70: +2,   # Yb
    71: +3,   # Lu
    72: +4,   # Hf
    73: +5,   # Ta
    74: +6,   # W
    75: +7,   # Re
    76: +4,   # Os
    77: +3,   # Ir
    78: +2,   # Pt
    79: +1,   # Au
    80: +2,   # Hg
    81: +1,   # Tl
    82: +2,   # Pb
    83: +3,   # Bi
    84: +2,   # Po
    85: -1,   # At
    86:  0,   # Rn
    87: +1,   # Fr
    88: +2,   # Ra
    89: +3,   # Ac
    90: +4,   # Th
    91: +5,   # Pa
    92: +6,   # U
    93: +5,   # Np
    94: +4,   # Pu
    95: +3,   # Am
    96: +3,   # Cm
    97: +3,   # Bk
    98: +3,   # Cf
    99: +3,   # Es
   100: +3,   # Fm
   101: +3,   # Md
   102: +2,   # No
   103: +3,   # Lr
   104: +4,   # Rf
   105: +5,   # Db
   106: +6,   # Sg
   107: +7,   # Bh
   108: +4,   # Hs
   109: +3,   # Mt
   110: +2,   # Ds
   111: +1,   # Rg
   112: +2,   # Cn
   113: +3,   # Nh
   114: +2,   # Fl
   115: +3,   # Mc
   116: +2,   # Lv
   117: -1,   # Ts
   118:  0    # Og
}

class FixedCharges(nn.Module):
    def __init__(self,
                 charge_dict: Dict[int, float] = typical_charge,
                 normalization_factor: float = 0.5,
                 ):
        super().__init__()
        self.charge_dict = charge_dict
        self.normalization_factor = normalization_factor

    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        charge = torch.tensor([self.charge_dict[atomic_number.item()] for atomic_number in atomic_numbers], device=atomic_numbers.device)
        return charge * self.normalization_factor
