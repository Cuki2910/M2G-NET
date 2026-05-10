"""
Phase 9: Repeated Runs and Statistical Significance Testing
This script runs the TG-MVMT-GFNet and a baseline (XGBoost) for multiple 
random seeds to compute means, 95% Confidence Intervals, and a paired t-test.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import pandas as pd
from scipy import stats

import config as cfg
from src.data_pipeline import load_data
from scripts.train import train
from baselines.run_all_baselines import run_sklearn_baseline
from xgboost import XGBClassifier

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_ci(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def main():
    n_runs = 5
    seeds = [42, 101, 2023, 777, 999]
    
    gfnet_aucs = []
    baseline_aucs = []
    
    train_df, val_df, test_df, encoders, vocab = load_data()
    trainval_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    
    print(f"Starting repeated runs ({n_runs} runs)...")
    
    for i, seed in enumerate(seeds):
        print(f"\n--- Run {i+1}/{n_runs} (Seed: {seed}) ---")
        set_random_seed(seed)
        
        # 1. Train GFNet
        print("Training TG-MVMT-GFNet...")
        # Train using the default epochs and patience
        model, loss_fn, test_metrics, history = train(
            vocab, train_df, val_df, test_df, 
            max_epochs=cfg.MAX_EPOCHS, 
            patience=cfg.EARLY_STOPPING_PATIENCE
        )
        gfnet_auc = test_metrics["macro"]["roc_auc"]
        gfnet_aucs.append(gfnet_auc)
        
        # 2. Train Baseline (XGBoost)
        print("Training Baseline (XGBoost)...")
        baseline_kwargs = {
            "n_estimators": 100, 
            "use_label_encoder": False,
            "eval_metric": "logloss", 
            "random_state": seed, 
            "verbosity": 0
        }
        res_baseline = run_sklearn_baseline(
            XGBClassifier, baseline_kwargs,
            trainval_df, test_df, "XGBoost"
        )
        baseline_auc = res_baseline["macro"]["roc_auc"]
        baseline_aucs.append(baseline_auc)
        
        print(f"Run {i+1} Results -> GFNet AUC: {gfnet_auc:.4f}, XGBoost AUC: {baseline_auc:.4f}")
        
    print("\n\n=== REPEATED RUNS SUMMARY ===")
    
    # GFNet stats
    m_gf, lower_gf, upper_gf = compute_ci(gfnet_aucs)
    print(f"TG-MVMT-GFNet Macro ROC-AUC : {m_gf:.4f} (95% CI: [{lower_gf:.4f}, {upper_gf:.4f}])")
    
    # Baseline stats
    m_bs, lower_bs, upper_bs = compute_ci(baseline_aucs)
    print(f"XGBoost Baseline Macro ROC-AUC: {m_bs:.4f} (95% CI: [{lower_bs:.4f}, {upper_bs:.4f}])")
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(gfnet_aucs, baseline_aucs)
    print(f"\nPaired t-test (GFNet vs XGBoost):")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value    : {p_value:.4e}")
    
    if p_value < 0.05:
        print("Conclusion: The difference is statistically significant (p < 0.05).")
    else:
        print("Conclusion: The difference is NOT statistically significant (p >= 0.05).")

if __name__ == "__main__":
    main()
