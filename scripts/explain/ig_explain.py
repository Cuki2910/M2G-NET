"""Joint encoder-space Integrated Gradients with numerical completeness checks."""

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import captum
from captum.attr import IntegratedGradients
import numpy as np
import pandas as pd
import torch

import config as cfg
from src.checkpoint import dataset_sha256, load_model_bundle
from src.data_pipeline import get_loaders


class TaskPredictionWrapper(torch.nn.Module):
    def __init__(self, model, task_idx):
        super().__init__()
        self.model = model
        self.task_idx = task_idx

    def forward(self, *representations):
        representations = list(representations)
        if self.model.ablation == "no_site":
            representations[-1] = representations[-1] * 0
        individual = torch.cat(representations[:2], dim=-1)
        contextual = torch.cat(representations[2:], dim=-1)
        interaction = (None if self.model.ablation == "no_interaction"
                       else self.model.interaction(individual, contextual))
        fused, _, _ = self.model.fusion(representations, interaction)
        return self.model.heads[self.task_idx](fused[self.task_idx])


def integrated_view_attributions(model, views, task_idx, n_steps=256, use_site_intercept=True):
    if type(n_steps) is not int or n_steps < 2:
        raise ValueError("n_steps must be an integer of at least 2.")
    if type(task_idx) is not int or not 0 <= task_idx < cfg.NUM_TASKS:
        raise ValueError("Invalid task index.")
    training_states = {module: module.training for module in model.modules()}
    model.eval()
    try:
        with torch.no_grad():
            representations = tuple(
                model.encoders[name](views["site_obs"], views["site_id"], use_intercept=use_site_intercept)
                if name == "site" else model.encoders[name](views[name])
                for name in cfg.ALL_VIEW_NAMES)
        if any(not torch.isfinite(value).all() for value in representations):
            raise ValueError("Encoder representations must be finite.")
        reference = tuple(torch.zeros_like(value) for value in representations)
        forward = TaskPredictionWrapper(model, task_idx)
        with torch.enable_grad():
            attributes, delta = IntegratedGradients(forward).attribute(
                representations, baselines=reference, n_steps=n_steps,
                method="gausslegendre", internal_batch_size=representations[0].shape[0],
                return_convergence_delta=True)
        with torch.no_grad():
            difference = forward(*representations) - forward(*reference)
        if not all(torch.isfinite(value).all() for value in (*attributes, delta, difference)):
            raise ValueError("Attributions and prediction differences must be finite.")
        return torch.stack(attributes, dim=1).detach(), delta.detach(), difference.detach()
    finally:
        for module, training in training_states.items():
            module.training = training


def compute_ig_per_view(model, loader, task_idx, n_steps=256, n_samples=200,
                        use_site_intercept=True, atol=1e-4, rtol=1e-3):
    if type(n_samples) is not int or n_samples < 1:
        raise ValueError("n_samples must be a positive integer.")
    if type(task_idx) is not int or not 0 <= task_idx < cfg.NUM_TASKS:
        raise ValueError("Invalid task index.")
    if not np.isfinite([atol, rtol]).all() or min(atol, rtol) < 0:
        raise ValueError("Completeness tolerances must be finite and nonnegative.")
    attributes, deltas, differences, indices = [], [], [], []
    count, offset = 0, 0
    for views, targets, masks in loader:
        if count >= n_samples:
            break
        if masks.shape != targets.shape or masks.ndim != 2 or masks.shape[1] != cfg.NUM_TASKS:
            raise ValueError("Targets and observation masks must have shape [batch, num_tasks].")
        if not ((masks == 0) | (masks == 1)).all():
            raise ValueError("Observation masks must be binary and finite.")
        selected = torch.nonzero(masks[:, task_idx] == 1).flatten()[:n_samples - count]
        if selected.numel():
            batch_views = {name: value[selected] for name, value in views.items()}
            attribution, delta, difference = integrated_view_attributions(
                model, batch_views, task_idx, n_steps, use_site_intercept)
            attributes.append(attribution.cpu().numpy())
            deltas.append(delta.cpu().numpy())
            differences.append(difference.cpu().numpy())
            indices.append(selected.cpu().numpy() + offset)
            count += len(selected)
        offset += len(targets)
    if not count:
        raise ValueError("No observed labels for the requested task.")
    delta, difference = np.concatenate(deltas), np.concatenate(differences)
    if not np.all(np.abs(delta) <= atol + rtol * np.abs(difference)):
        raise RuntimeError(f"IG completeness tolerance failed (max residual {np.max(np.abs(delta)):.6g}); increase n_steps.")
    signed = np.concatenate(attributes)
    signed_views = signed.sum(axis=2)
    magnitudes = np.abs(signed_views).mean(axis=0)
    total = magnitudes.sum()
    return {"signed_attributions": signed, "signed_view_attributions": signed_views, "completeness_delta": delta,
            "output_difference": difference, "sample_indices": np.concatenate(indices),
            "mean_absolute_attribution": magnitudes,
            "normalized_importance": magnitudes / total if total else np.zeros_like(magnitudes)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Trusted checkpoint from the paper pipeline.")
    parser.add_argument("--output", required=True, help="New output directory; never overwritten.")
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args(argv)
    if args.n_steps < 2 or args.samples < 1 or not np.isfinite([args.atol, args.rtol]).all() or min(args.atol, args.rtol) < 0:
        parser.error("Steps >= 2, samples >= 1, and finite nonnegative tolerances are required.")
    torch.set_num_threads(1)
    bundle = load_model_bundle(args.checkpoint)
    if not bundle["checkpoint"].get("dataset_sha256") or not bundle["checkpoint"].get("split_rows"):
        raise ValueError("Auditable IG requires a checkpoint with a dataset hash and exact split rows.")
    _, _, loader = get_loaders(bundle["train_df"], bundle["val_df"], bundle["test_df"], bundle["vocab"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__).resolve().relative_to(Path.cwd()), Path("config.py"),
               *(path for path in sorted(Path("src").glob("*.py")) if not path.name.startswith("._"))]
    for source in sources:
        destination = output / "source" / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    manifest = {"status": "running", "result_kind": "development_not_publication_validation",
                "method": "joint_encoder_space_integrated_gradients", "baseline": "zero_encoder_outputs",
                "prediction_scale": "raw_model_probability", "quadrature": "gausslegendre",
                "view_order": cfg.ALL_VIEW_NAMES, "feature_level_attribution": False,
                "population_summary": "mean_absolute_signed_view_attribution",
                "arguments": vars(args), "checkpoint_sha256": dataset_sha256(args.checkpoint),
                "dataset_sha256": bundle["checkpoint"]["dataset_sha256"],
                "use_site_intercept": bundle["test_use_site_intercept"],
                "packages": {"torch": torch.__version__, "captum": captum.__version__, "numpy": np.__version__},
                "source_sha256": {str(source): dataset_sha256(source) for source in sources}}
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        rows = []
        for task_idx, task in enumerate(cfg.TASK_NAMES):
            result = compute_ig_per_view(bundle["model"], loader, task_idx, args.n_steps, args.samples,
                                         bundle["test_use_site_intercept"], args.atol, args.rtol)
            row_ids = bundle["test_df"]["_row_id"].to_numpy()[result.pop("sample_indices")]
            np.savez_compressed(output / f"{task}.npz", row_ids=row_ids, **result)
            for view_idx, view in enumerate(cfg.ALL_VIEW_NAMES):
                rows.append({"task": task, "view": view, "n_observed": len(row_ids),
                             "mean_absolute_attribution": float(result["mean_absolute_attribution"][view_idx]),
                             "normalized_importance": float(result["normalized_importance"][view_idx]),
                             "max_absolute_completeness_delta": float(np.max(np.abs(result["completeness_delta"])))})
        pd.DataFrame(rows).to_csv(output / "view_attributions.csv", index=False)
        manifest["status"] = "complete"
        manifest["artifact_sha256"] = {path.name: dataset_sha256(path) for path in sorted(output.iterdir())
                                      if path.is_file() and path.name != "manifest.json" and not path.name.startswith("._")}
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Validated IG artifacts: {output}")
    return output


if __name__ == "__main__":
    main()
