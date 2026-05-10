import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd

import config as cfg


os.makedirs("outputs", exist_ok=True)


def task_mask(df, task):
    col = cfg.TASK_MASK_COLS.get(task)
    if col and col in df.columns:
        return df[col].astype(bool).to_numpy()
    return np.ones(len(df), dtype=bool)


def main():
    df = pd.read_csv(cfg.DATA_PATH)
    rows = []

    for task in cfg.TASK_NAMES:
        observed = task_mask(df, task)
        y = df.loc[observed, task].astype(float)
        rows.append({
            "task": task,
            "display_name": cfg.TASK_DISPLAY_NAMES.get(task, task),
            "n_total": len(df),
            "n_observed": int(observed.sum()),
            "observed_rate": float(observed.mean()),
            "n_positive": int(y.sum()),
            "positive_rate_observed": float(y.mean()) if len(y) else np.nan,
        })

    obs_table = pd.DataFrame(rows)
    obs_path = "outputs/dataset_task_observation.csv"
    obs_table.to_csv(obs_path, index=False)

    corr = pd.DataFrame(index=cfg.TASK_NAMES, columns=cfg.TASK_NAMES, dtype=float)
    cooccur = pd.DataFrame(index=cfg.TASK_NAMES, columns=cfg.TASK_NAMES, dtype=float)

    for t1 in cfg.TASK_NAMES:
        for t2 in cfg.TASK_NAMES:
            mask = task_mask(df, t1) & task_mask(df, t2)
            if mask.sum() < 2:
                corr.loc[t1, t2] = np.nan
                cooccur.loc[t1, t2] = np.nan
                continue
            y1 = df.loc[mask, t1].astype(float)
            y2 = df.loc[mask, t2].astype(float)
            corr.loc[t1, t2] = y1.corr(y2)
            cooccur.loc[t1, t2] = float(((y1 == 1) & (y2 == 1)).mean())

    corr_path = "outputs/label_correlation.csv"
    cooccur_path = "outputs/label_cooccurrence.csv"
    corr.to_csv(corr_path)
    cooccur.to_csv(cooccur_path)

    print("\n=== Dataset / Task Observation Table ===")
    print(obs_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {obs_path}")
    print(f"Saved: {corr_path}")
    print(f"Saved: {cooccur_path}")


if __name__ == "__main__":
    main()
