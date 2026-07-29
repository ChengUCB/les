import unittest

import torch

from les.module.bec import BEC


class BECTests(unittest.TestCase):
    def test_triclinic_fixed_charges_project_to_cartesian_bec(self):
        cell = torch.tensor(
            [[[2.0, 0.3, 0.1],
              [0.0, 3.0, 0.2],
              [0.0, 0.0, 4.0]]],
            dtype=torch.double,
        )
        r = torch.tensor(
            [[0.2, 0.4, 0.6],
             [1.1, 1.2, 1.3]],
            dtype=torch.double,
            requires_grad=True,
        )
        q = torch.tensor([1.0, -1.0], dtype=torch.double)

        bec = BEC(remove_mean=False)(q=q, r=r, cell=cell)

        expected = q[:, None, None] * torch.eye(3, dtype=torch.double)
        torch.testing.assert_close(bec, expected, rtol=1e-12, atol=1e-12)

    def test_output_index_matches_full_tensor_row(self):
        cell = torch.tensor(
            [[[2.0, 0.3, 0.1],
              [0.0, 3.0, 0.2],
              [0.0, 0.0, 4.0]]],
            dtype=torch.double,
        )
        q = torch.tensor([1.0, -1.0], dtype=torch.double)
        r_full = torch.tensor(
            [[0.2, 0.4, 0.6],
             [1.1, 1.2, 1.3]],
            dtype=torch.double,
            requires_grad=True,
        )
        r_indexed = r_full.detach().clone().requires_grad_(True)

        bec = BEC(remove_mean=False)
        full = bec(q=q, r=r_full, cell=cell)
        indexed = bec(q=q, r=r_indexed, cell=cell, output_index=1)

        torch.testing.assert_close(indexed, full[:, 1, :], rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
