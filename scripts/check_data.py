"""Validate source data without changing it; strict by default."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import config as cfg
from src.data_pipeline import prepare_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=cfg.DATA_PATH)
    parser.add_argument("--label-policy", choices=["strict", "mask"], default=cfg.LABEL_POLICY)
    parser.add_argument("--require-masks", action="store_true")
    args = parser.parse_args()
    raw = pd.read_csv(args.data)
    try:
        frame = prepare_labels(raw, args.label_policy, args.require_masks)
    except (ValueError, TypeError) as error:
        parser.exit(1, f"Data validation failed: {error}\n")
    print(f"Source records: {len(raw)}; sites: {frame[cfg.SITE_ID_COL].nunique()}; policy: {args.label_policy}")
    absent = [column for column in cfg.TASK_MASK_COLS.values() if column not in raw]
    if absent:
        print("WARNING: absent explicit masks; finite synthetic labels assumed observed except signal-inapplicable outcomes.")
        print("Turn-signal eligibility is unknown; do not claim validated partial-observation handling.")
    for task in cfg.TASK_NAMES:
        observed = frame[cfg.TASK_MASK_COLS[task]].eq(1)
        labels = frame.loc[observed, task]
        print(f"{task}: observed={int(observed.sum())}, positive={int(labels.sum())}, prevalence={labels.mean():.4f}")
    print("Source CSV unchanged.")


if __name__ == "__main__":
    main()
