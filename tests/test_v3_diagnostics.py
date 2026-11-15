"""In-memory telemetry regression; never opens project data or fits an M5 model.

Run: .venv/bin/python -m unittest tests.test_v3_diagnostics -v
"""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from scripts import train


class MemoryDataset(Dataset):
    def __init__(self, offset=0):
        self.ids = torch.arange(12) + offset
        self.targets = ((torch.arange(12)[:, None] +
                         torch.arange(len(train.cfg.TASK_NAMES))[None, :]) % 2).float()
        self.masks = torch.ones_like(self.targets)
        self.masks[::3, -1] = 0

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return {"id": self.ids[index]}, self.targets[index], self.masks[index]


class TinyDropoutModel(torch.nn.Module):
    def __init__(self, vocab, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(len(train.cfg.TASK_NAMES)))
        self.bias = torch.nn.Parameter(torch.zeros(len(train.cfg.TASK_NAMES)))
        self.dropout = torch.nn.Dropout(0.4)
        self.calls = []

    def forward(self, views, use_site_intercept=True):
        x = ((views["id"] % 12).float() / 12)[:, None]
        logits = self.dropout(x.expand(-1, len(self.weight))) * self.weight + self.bias
        probabilities = torch.sigmoid(logits + (0.3 if use_site_intercept else 0))
        # Exercise NumPy restoration too, even if production eval is deterministic.
        draw = np.random.random()
        self.calls.append((self.training, use_site_intercept, views["id"].clone(),
                           probabilities.detach().clone(), draw))
        return list(probabilities.unbind(dim=1)), None, None


class DiagnosticsTests(unittest.TestCase):
    def fit(self, diagnostics=None):
        training, validation = MemoryDataset(), MemoryDataset(offset=100)
        epochs = []
        real_run_epoch = train.run_epoch

        def loaders(train_df, val_df, test_df, vocab):
            self.assertIs(train_df, training)
            self.assertIs(val_df, validation)
            self.assertIs(test_df, validation)
            return (DataLoader(training, batch_size=4, shuffle=True),
                    DataLoader(validation, batch_size=4, shuffle=False),
                    DataLoader(validation, batch_size=4, shuffle=False))

        def epoch(model, loader, loss_fn, optimizer=None, **kwargs):
            loss, metrics, task_losses = real_run_epoch(
                model, loader, loss_fn, optimizer, **kwargs)
            if optimizer is None:
                # Force a best epoch before stopping; keep real per-task metrics.
                metrics["macro"]["roc_auc"] = (0.7, 0.8, 0.6, 0.5)[len(epochs)]
                epochs.append((deepcopy(model.state_dict()), deepcopy(metrics)))
            return loss, metrics, task_losses

        with patch.object(train, "TGMVMTGFNetV2", TinyDropoutModel), \
                patch.object(train, "get_loaders", side_effect=loaders), \
                patch.object(train, "anneal_temperature"), \
                patch.object(train, "compute_pos_weights", return_value=np.ones(len(train.cfg.TASK_NAMES))), \
                patch.object(train, "run_epoch", side_effect=epoch), \
                patch.object(train, "load_data", side_effect=AssertionError("No dataset access")):
            kwargs = {} if diagnostics is None else {"diagnostics": diagnostics}
            result = train.fit_model_for_split(
                {}, training, validation, max_epochs=7, patience=2,
                use_site_intercept=True, validation_use_site_intercept=False,
                use_uncertainty=False, use_focal=False, split_seed=17, **kwargs)
        return result, epochs, torch.get_rng_state().clone(), deepcopy(np.random.get_state())

    def assert_numpy_state_equal(self, left, right):
        self.assertEqual(left[0], right[0])
        np.testing.assert_array_equal(left[1], right[1])
        self.assertEqual(left[2:], right[2:])

    def test_history_preserves_sampling_dropout_checkpoint_predictions_and_rng(self):
        baseline, baseline_epochs, torch_before, numpy_before = self.fit()
        diagnostics = {}
        observed, observed_epochs, torch_after, numpy_after = self.fit(diagnostics)
        base_model, base_loss, base_auc, base_epoch = baseline
        model, loss, auc, best_epoch = observed
        self.assertEqual((base_epoch, base_auc), (2, 0.8))
        self.assertEqual((best_epoch, auc), (base_epoch, base_auc))
        self.assertEqual(len(baseline_epochs), 4)
        self.assertEqual(len(observed_epochs), 4)
        self.assertFalse(model.training)
        self.assertTrue(torch.equal(torch_before, torch_after))
        self.assert_numpy_state_equal(numpy_before, numpy_after)
        for baseline_module, observed_module in ((base_model, model), (base_loss, loss)):
            for key, tensor in baseline_module.state_dict().items():
                self.assertTrue(torch.equal(tensor, observed_module.state_dict()[key]), key)
        for key, tensor in observed_epochs[best_epoch - 1][0].items():
            self.assertTrue(torch.equal(tensor, model.state_dict()[key]), key)
        self.assertTrue(any(not torch.equal(tensor, observed_epochs[-1][0][key])
                            for key, tensor in model.state_dict().items()))

        # All original forwards (including shuffle order/dropout outputs) stay exact.
        original_calls = len(base_model.calls)
        self.assertGreater(len(model.calls), original_calls)
        for left, right in zip(base_model.calls, model.calls[:original_calls]):
            self.assertEqual(left[:2], right[:2])
            self.assertTrue(torch.equal(left[2], right[2]))
            self.assertTrue(torch.equal(left[3], right[3]))
            self.assertEqual(left[4], right[4])
        final_calls = model.calls[original_calls:]
        self.assertTrue(all(not call[0] for call in final_calls))
        train_calls = [call for call in final_calls if (call[2] < 100).all()]
        val_calls = [call for call in final_calls if (call[2] >= 100).all()]
        self.assertTrue(train_calls and val_calls)
        self.assertTrue(all(call[1] for call in train_calls))
        self.assertTrue(all(not call[1] for call in val_calls))
        self.assertTrue(torch.equal(torch.cat([call[2] for call in train_calls]), torch.arange(12)))
        self.assertTrue(torch.equal(torch.cat([call[2] for call in val_calls]), torch.arange(12) + 100))

        history = diagnostics["history"]
        self.assertEqual([row["epoch"] for row in history], [1, 2, 3, 4])
        for row, (_, expected) in zip(history, observed_epochs):
            self.assertEqual(row["validation_metrics"], expected)
            self.assertEqual(set(row["training_metrics_dropout"]), set(train.cfg.TASK_NAMES) | {"macro"})
            self.assertEqual(set(row["validation_metrics"]), set(train.cfg.TASK_NAMES) | {"macro"})
        self.assertEqual(diagnostics["best"]["epoch"], best_epoch)
        for split, offset, use_site in (("train", 0, True), ("validation", 100, False)):
            loader = DataLoader(MemoryDataset(offset), batch_size=train.cfg.BATCH_SIZE, shuffle=False)
            y, probabilities, masks = train.collect_predictions(model, loader, use_site)
            expected = train.compute_all_metrics(y, probabilities, masks)
            self.assertEqual(diagnostics["best"][f"{split}_metrics"], expected)

        loader = DataLoader(MemoryDataset(), batch_size=4, shuffle=False)
        base_predictions = train.collect_predictions(base_model, loader, True)[1]
        predictions = train.collect_predictions(model, loader, True)[1]
        np.testing.assert_array_equal(base_predictions, predictions)


if __name__ == "__main__":
    unittest.main()
