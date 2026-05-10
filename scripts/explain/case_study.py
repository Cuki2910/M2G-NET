import sys
sys.path.insert(0, "..")

"""
Case Studies Generator
Generates narrative explanations for individual predictions.
Shows why model predicts high/low risk for specific riders.
"""

import sys, os
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import torch

import config as cfg
from src.data_pipeline import load_data, get_loaders
from src.model import TGMVMTGFNetV2
from src.loss import UncertaintyWeightedLoss


def load_best_model(vocab):
    model = TGMVMTGFNetV2(vocab)
    loss_fn = UncertaintyWeightedLoss()
    ckpt = torch.load(cfg.CHECKPOINT_PATH, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    loss_fn.load_state_dict(ckpt["loss_state"])
    model.eval()
    return model, loss_fn


def decode_sample(sample_idx, df, encoders):
    """Decode a sample back to human-readable format."""
    row = df.iloc[sample_idx]
    decoded = {}

    for col, encoder in encoders.items():
        if col in row.index:
            decoded[col] = encoder.inverse_transform([int(row[col])])[0]

    # Handle intersection_id (not encoded by LabelEncoder)
    decoded['intersection_id'] = int(row['intersection_id']) + 1  # Convert back to 1-indexed

    return decoded


def get_risk_level(prob):
    """Categorize risk probability."""
    if prob >= 0.7:
        return "Very High", "🔴"
    elif prob >= 0.5:
        return "High", "🟠"
    elif prob >= 0.3:
        return "Medium", "🟡"
    else:
        return "Low", "🟢"


def format_gate_weights(gate_weights):
    """Format gate weights as percentages."""
    view_names = ["Rider Role", "Rider Traits", "Road Context", "Environment", "Site", "Interaction"]
    formatted = []
    for i, (name, weight) in enumerate(zip(view_names, gate_weights)):
        if weight > 0.1:  # Only show significant weights
            formatted.append(f"{name} ({weight*100:.1f}%)")
    return ", ".join(formatted) if formatted else "Balanced"


def generate_case_study(sample_idx, df, encoders, vocab, model, test_loader):
    """Generate detailed case study for a single sample."""

    # Get decoded sample info
    decoded = decode_sample(sample_idx, df, encoders)

    # Get model predictions
    views_batch, targets_batch = None, None
    current_idx = 0

    for views, targets, masks in test_loader:
        batch_size = targets.shape[0]
        if current_idx <= sample_idx < current_idx + batch_size:
            local_idx = sample_idx - current_idx
            views_batch = {k: v[local_idx:local_idx+1] for k, v in views.items()}
            targets_batch = targets[local_idx:local_idx+1]
            break
        current_idx += batch_size

    if views_batch is None:
        return None

    # Get predictions and gate weights
    with torch.no_grad():
        preds, gate_weights, alpha = model(views_batch)

    predictions = {
        task: float(preds[i][0])
        for i, task in enumerate(cfg.TASK_NAMES)
    }

    ground_truth = {
        task: int(targets_batch[0][i])
        for i, task in enumerate(cfg.TASK_NAMES)
    }

    gate_weights_np = [gw[0].numpy() for gw in gate_weights]

    # Build narrative
    narrative = []
    narrative.append("="*80)
    narrative.append(f"CASE STUDY #{sample_idx + 1}")
    narrative.append("="*80)

    # Rider Profile
    narrative.append("\n[RIDER PROFILE]")
    narrative.append(f"Type: {decoded['rider_type']}")
    narrative.append(f"Gender: {decoded['gender']}, Age: {decoded['age_group']}")
    narrative.append(f"Time: {decoded['time_slot']}, Weather: {decoded['weather']}")
    narrative.append(f"Weekend: {decoded['weekend']}")

    # Context
    narrative.append("\n[CONTEXT]")
    narrative.append(f"Intersection: #{decoded['intersection_id']} ({decoded['intersection_type']})")
    narrative.append(f"Traffic: {decoded['traffic_condition']}, Lanes: {decoded['num_lanes']}")
    narrative.append(f"Signal: {decoded['has_signal']}, Police: {decoded['police_presence']}")

    # Predictions
    narrative.append("\n[MODEL PREDICTIONS]")
    for task in cfg.TASK_NAMES:
        prob = predictions[task]
        truth = ground_truth[task]
        risk_level, emoji = get_risk_level(prob)
        correct = "✓" if (prob >= 0.5) == truth else "✗"

        narrative.append(f"\n{task.replace('_', ' ').title()}:")
        narrative.append(f"  Predicted Risk: {prob:.1%} ({risk_level} {emoji})")
        narrative.append(f"  Ground Truth: {'Violation' if truth else 'Compliant'} {correct}")

    # Explanations
    narrative.append("\n[WHY THESE PREDICTIONS?]")

    for i, task in enumerate(cfg.TASK_NAMES):
        prob = predictions[task]
        gw = gate_weights_np[i]

        narrative.append(f"\n{task.replace('_', ' ').title()} ({prob:.1%}):")

        # Top contributing views
        top_views = np.argsort(gw)[::-1][:3]
        view_names = ["Rider Role", "Rider Traits", "Road Context", "Environment", "Site", "Interaction"]

        narrative.append(f"  Key Factors: {format_gate_weights(gw)}")

        # Generate explanation based on top view
        top_view_idx = top_views[0]
        top_view_name = view_names[top_view_idx]

        if top_view_idx == 0:  # Rider Role
            narrative.append(f"  → {decoded['rider_type']} riders have specific risk patterns")
        elif top_view_idx == 1:  # Rider Traits
            narrative.append(f"  → {decoded['gender']}, {decoded['age_group']} demographic factors")
        elif top_view_idx == 2:  # Road Context
            narrative.append(f"  → {decoded['traffic_condition']} traffic, {decoded['num_lanes']} lanes")
        elif top_view_idx == 3:  # Environment
            narrative.append(f"  → {decoded['weather']} weather, {decoded['time_slot']} time")
        elif top_view_idx == 4:  # Site
            narrative.append(f"  → Intersection type: {decoded['intersection_type']}")
        else:
            narrative.append("  -> Cross-level individual-context interaction")

    # Summary
    narrative.append("\n[SUMMARY]")
    high_risk_tasks = [task for task, prob in predictions.items() if prob >= 0.5]

    if high_risk_tasks:
        narrative.append(f"High-risk violations: {', '.join([t.replace('_', ' ') for t in high_risk_tasks])}")
        narrative.append(f"Recommendation: Targeted enforcement and education")
    else:
        narrative.append("Low overall risk profile")
        narrative.append("Recommendation: Standard monitoring")

    narrative.append("\n" + "="*80 + "\n")

    return "\n".join(narrative)


def select_interesting_samples(df, test_loader, model, n_samples=10):
    """
    Select interesting samples for case studies:
    - High risk predictions (prob > 0.7)
    - Low risk predictions (prob < 0.3)
    - Misclassifications
    - Edge cases
    """

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for views, targets, masks in test_loader:
            preds, _, _ = model(views)
            probs = torch.stack(preds, dim=1).numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())

    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)

    selected = []

    # 1. High risk cases (any task > 0.7)
    high_risk_mask = (all_probs > 0.7).any(axis=1)
    high_risk_indices = np.where(high_risk_mask)[0]
    if len(high_risk_indices) > 0:
        selected.extend(np.random.choice(high_risk_indices, min(3, len(high_risk_indices)), replace=False))

    # 2. Low risk cases (all tasks < 0.3)
    low_risk_mask = (all_probs < 0.3).all(axis=1)
    low_risk_indices = np.where(low_risk_mask)[0]
    if len(low_risk_indices) > 0:
        selected.extend(np.random.choice(low_risk_indices, min(2, len(low_risk_indices)), replace=False))

    # 3. Misclassifications (predicted != actual)
    pred_labels = (all_probs >= 0.5).astype(int)
    misclass_mask = (pred_labels != all_targets).any(axis=1)
    misclass_indices = np.where(misclass_mask)[0]
    if len(misclass_indices) > 0:
        selected.extend(np.random.choice(misclass_indices, min(3, len(misclass_indices)), replace=False))

    # 4. Edge cases (prob around 0.5)
    edge_mask = ((all_probs > 0.4) & (all_probs < 0.6)).any(axis=1)
    edge_indices = np.where(edge_mask)[0]
    if len(edge_indices) > 0:
        selected.extend(np.random.choice(edge_indices, min(2, len(edge_indices)), replace=False))

    # Remove duplicates and limit to n_samples
    selected = list(set(selected))[:n_samples]

    return sorted(selected)


def main():
    print("="*80)
    print("CASE STUDIES GENERATOR")
    print("="*80)

    # Load data and model
    train_df, val_df, test_df, encoders, vocab = load_data()
    _, _, test_loader = get_loaders(train_df, val_df, test_df, vocab)
    model, loss_fn = load_best_model(vocab)

    print(f"\nLoaded model from: {cfg.CHECKPOINT_PATH}")
    print(f"Test set size: {len(test_df)}")

    # Select interesting samples
    print("\nSelecting interesting samples...")
    selected_indices = select_interesting_samples(test_df, test_loader, model, n_samples=10)
    print(f"Selected {len(selected_indices)} samples for case studies")

    # Generate case studies
    case_studies = []
    for idx in selected_indices:
        print(f"Generating case study for sample #{idx + 1}...")
        study = generate_case_study(idx, test_df, encoders, vocab, model, test_loader)
        if study:
            case_studies.append(study)

    # Save to file
    output_path = "outputs/case_studies.md"
    os.makedirs("outputs", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Case Studies: TG-MVMT-GFNet v2 Predictions\n\n")
        f.write("**Generated:** 2026-05-10\n")
        f.write(f"**Model:** TG-MVMT-GFNet v2 (14,513 parameters)\n")
        f.write(f"**Test Set:** {len(test_df)} samples\n\n")
        f.write("---\n\n")

        for study in case_studies:
            f.write(study)
            f.write("\n")

    print(f"\n[OK] Case studies saved to: {output_path}")
    print(f"Total case studies: {len(case_studies)}")

    # Print first case study as preview
    if case_studies:
        print("\n" + "="*80)
        print("PREVIEW: First Case Study")
        print("="*80)
        print(case_studies[0])


if __name__ == "__main__":
    main()
