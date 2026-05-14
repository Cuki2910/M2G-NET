import unittest

import config as cfg
from src.checkpoint import apply_checkpoint_model_config, current_model_config
from src.model import TGMVMTGFNetV2


class CheckpointModelConfigTest(unittest.TestCase):
    def _vocab(self):
        return {
            "rider_type": 3,
            "gender": 3,
            "age_group": 4,
            "police_presence": 2,
            "traffic_condition": 3,
            "num_lanes": 4,
            "has_signal": 2,
            "weather": 3,
            "time_slot": 4,
            "weekend": 2,
            "intersection_type": 3,
            "num_sites": 5,
            "site_mapping": {1: 0, 2: 1, 3: 2, 4: 3},
            "unknown_site_id": 4,
        }

    def test_checkpoint_prior_weight_is_applied_to_gates(self):
        model = TGMVMTGFNetV2(self._vocab())
        ckpt = {"model_config": {**current_model_config(), "gate_prior_weight": 0.3}}

        apply_checkpoint_model_config(model, ckpt)

        self.assertTrue(all(gate.prior_weight == 0.3 for gate in model.fusion.gates))

    def test_architecture_mismatch_fails_fast(self):
        model = TGMVMTGFNetV2(self._vocab())
        ckpt = {"model_config": {**current_model_config(), "view_dim": cfg.VIEW_DIM + 1}}

        with self.assertRaises(ValueError):
            apply_checkpoint_model_config(model, ckpt)


if __name__ == "__main__":
    unittest.main()
