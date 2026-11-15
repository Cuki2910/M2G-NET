"""Select gate-prior smoothing using validation only, never test or out-of-site scores."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import config as cfg
from src.data_pipeline import load_data, get_loaders
from scripts.train import fit_model_for_split
from scripts.evaluate.gate_stability import extract_gate_weights, compute_gate_statistics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 2023, 777, 999])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--data", default=cfg.DATA_PATH)
    parser.add_argument("--label-policy", choices=["strict", "mask"], default=cfg.LABEL_POLICY)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--output-dir", default="outputs/gate_prior_validation_sweep")
    args = parser.parse_args(argv)
    if any(not 0 <= value <= 1 for value in args.lambdas):
        parser.error("Gate priors must be in [0, 1].")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    train, validation, _, encoders, vocab = load_data(
        seed=args.split_seed, data_path=args.data, label_policy=args.label_policy)
    _, loader, _ = get_loaders(train, validation, validation, vocab)
    rows, summaries = [], []
    for prior in args.lambdas:
        gates, scores = [], []
        for seed in args.seeds:
            model, _, score, epoch = fit_model_for_split(
                vocab, train, validation, encoders=encoders, split_seed=seed,
                gate_prior_weight=prior, max_epochs=args.max_epochs, patience=args.patience)
            gates.append(extract_gate_weights(model, loader))
            scores.append(score)
            rows.append({"lambda": prior, "seed": seed, "validation_roc_auc": score, "best_epoch": epoch})
        mean, std, cv = compute_gate_statistics(gates)
        summaries.append({"lambda": prior, "validation_roc_auc_mean": float(np.mean(scores)),
                          "validation_gate_mean_cv": float(cv.mean()) if len(gates) > 1 else None})
        np.savez_compressed(output / f"gates_lambda_{prior:g}.npz", mean=mean, std=std, cv=cv)
        pd.DataFrame(rows).to_csv(output / "runs.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary.csv", index=False)
    best = summary.loc[summary["validation_roc_auc_mean"].idxmax()].to_dict()
    from src.checkpoint import dataset_sha256
    from scripts.evaluate.paper_pipeline import save_json
    save_json(output / "selection.json", {
        "selection_metric": "validation_roc_auc_mean", "best": best,
        "arguments": vars(args), "dataset_sha256": dataset_sha256(args.data),
        "test_evaluated": False,
        "caveat": "Smoothed gate CV can shrink mechanically; it does not establish faithful explanations.",
    })
    print(summary.to_string(index=False))
    print(f"Validation-selected lambda: {best['lambda']}")
    return best


if __name__ == "__main__":
    main()
