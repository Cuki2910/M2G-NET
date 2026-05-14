import unittest

import torch

import config as cfg
from src.fusion import RegularizedTaskGate, sparsemax


class GatePriorSmoothingTest(unittest.TestCase):
    def test_gate_prior_smoothing_matches_formula(self):
        old_prior_weight = cfg.GATE_PRIOR_WEIGHT
        cfg.GATE_PRIOR_WEIGHT = 0.1
        try:
            gate = RegularizedTaskGate(num_inputs=3, input_dim=2, temperature_init=1.0)
            with torch.no_grad():
                gate.gate.weight.copy_(
                    torch.tensor([
                        [0.2, -0.3, 0.1, 0.4, -0.2, 0.5],
                        [-0.1, 0.3, 0.2, -0.4, 0.6, -0.2],
                        [0.5, 0.1, -0.3, 0.2, -0.1, 0.4],
                    ])
                )
                gate.gate.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))

            view_reps = [
                torch.tensor([[1.0, 0.5], [0.2, -0.3]]),
                torch.tensor([[-0.5, 0.7], [1.1, 0.4]]),
                torch.tensor([[0.3, -0.8], [-0.6, 0.9]]),
            ]

            alpha = gate(view_reps)
            logits = gate.gate(torch.cat(view_reps, dim=-1))
            raw_alpha = sparsemax(logits / gate.temperature.clamp(min=0.1), dim=-1)
            expected = (
                (1.0 - cfg.GATE_PRIOR_WEIGHT) * raw_alpha
                + cfg.GATE_PRIOR_WEIGHT / len(view_reps)
            )
            lower_bound = cfg.GATE_PRIOR_WEIGHT / len(view_reps)

            self.assertTrue(torch.allclose(alpha, expected, atol=1e-7))
            self.assertTrue(torch.allclose(alpha.sum(dim=-1), torch.ones(alpha.shape[0]), atol=1e-7))
            self.assertGreaterEqual(float(alpha.detach().min()), lower_bound - 1e-7)
        finally:
            cfg.GATE_PRIOR_WEIGHT = old_prior_weight

    def test_gate_prior_weight_validation(self):
        with self.assertRaises(ValueError):
            RegularizedTaskGate(num_inputs=3, input_dim=2, prior_weight=-0.1)
        with self.assertRaises(ValueError):
            RegularizedTaskGate(num_inputs=3, input_dim=2, prior_weight=1.1)

        gate = RegularizedTaskGate(num_inputs=3, input_dim=2, prior_weight=0.5)
        with self.assertRaises(ValueError):
            gate.prior_weight = 1.5


if __name__ == "__main__":
    unittest.main()
