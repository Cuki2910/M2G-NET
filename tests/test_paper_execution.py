"""Acceptance fails closed on artifact corruption, split leakage and source drift."""
import json
from pathlib import Path
import tempfile
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_paper_unit import verify, verify_sources
from scripts.generate_paper_simulation import sha256


class ExecutionTest(unittest.TestCase):
    def test_unit_acceptance_and_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            model = root / "random_seed_42/m2g_net"
            data.mkdir()
            model.mkdir(parents=True)
            (data / "data.csv").write_text("input\n")
            (data / "oracle.csv").write_text("latent\n")
            hashes = {name: sha256(data / name) for name in ("data.csv", "oracle.csv")}
            (data / "manifest.json").write_text(json.dumps({"file_hashes": hashes}))
            (root / "manifest.json").write_text(json.dumps({"status": "complete", "dataset_sha256": hashes["data.csv"], "source_sha256": {}}))
            evaluation = {"split_rows": {"train": [0, 1], "validation": [2], "test": [3]}}
            evaluation_path = model / "evaluation.json"
            evaluation_path.write_text(json.dumps(evaluation))
            complete = {"model": "m2g_net", "seed": 42, "fold": "random", "artifact_sha256": {"evaluation.json": sha256(evaluation_path)}}
            (model / "complete.json").write_text(json.dumps(complete))
            unit = {"output": str(root), "data_directory": str(data), "training_seed": 42, "model": "m2g_net", "expected_model_artifacts": ["evaluation.json"]}
            verify(unit)
            verify_sources({"source_sha256": {str(evaluation_path): sha256(evaluation_path)}})
            evaluation["split_rows"]["test"] = [1]
            evaluation_path.write_text(json.dumps(evaluation))
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                verify(unit)
            complete["artifact_sha256"]["evaluation.json"] = sha256(evaluation_path)
            (model / "complete.json").write_text(json.dumps(complete))
            with self.assertRaisesRegex(ValueError, "Split leakage"):
                verify(unit)
            with self.assertRaisesRegex(ValueError, "Frozen source changed"):
                verify_sources({"source_sha256": {str(evaluation_path): "invalid"}})


if __name__ == "__main__":
    unittest.main()
