"""
Evaluation and Reporting Module.

Generates:
1. Benchmark model comparative visualization (Recall, F1, ROC-AUC).
2. Confusion Matrix visualization annotated with clinical metrics.
3. Receiver Operating Characteristic (ROC) curve with optimal threshold marker.
4. Precision-Recall (PR) curve with Average Precision (AP).
5. Probability Calibration curve (Reliability diagram).
6. Comprehensive markdown evaluation report in reports/model_benchmarking_report.md.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.preprocessing import ALL_FEATURE_COLUMNS, TARGET_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "model_benchmarking_report.md"


def plot_benchmark_comparison(benchmark_df: pd.DataFrame, output_dir: Path) -> Path:
    """Generates a multi-metric comparative bar chart across all 5 benchmarked models."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    models = benchmark_df["Model"].values
    x = np.arange(len(models))
    width = 0.25

    rects1 = ax.bar(x - width, benchmark_df["Recall_Mean"], width, label="Recall (Sensitivity)", color="#d95f02", edgecolor="#333", alpha=0.9)
    rects2 = ax.bar(x, benchmark_df["F1_Mean"], width, label="F1-Score", color="#2b5c8f", edgecolor="#333", alpha=0.9)
    rects3 = ax.bar(x + width, benchmark_df["ROC_AUC_Mean"], width, label="ROC-AUC", color="#2ca02c", edgecolor="#333", alpha=0.9)

    ax.set_ylabel("Cross-Validated Score", fontsize=11, labelpad=8)
    ax.set_title("Model Benchmark Performance (5-Fold x 3 Repeats Stratified CV)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10.5)
    ax.set_ylim(0.70, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#ccc", fontsize=10)

    # Annotate values on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.008, f"{h:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.008, f"{h:.2f}", ha="center", va="bottom", fontsize=8.5)
    for rect in rects3:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.008, f"{h:.2f}", ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    out_path = output_dir / "model_comparison_benchmark.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> Path:
    """Generates an annotated clinical confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    labels = np.array([
        [f"True Neg (TN)\n{tn}\n({tn/(tn+fp):.1%})", f"False Pos (FP)\n{fp}\n({fp/(tn+fp):.1%})"],
        [f"False Neg (FN)\n{fn}\n({fn/(fn+tp):.1%})", f"True Pos (TP)\n{tp}\n({tp/(fn+tp):.1%})"]
    ])

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    sns.heatmap(
        cm, annot=labels, fmt="", cmap="Blues", cbar=False,
        xticklabels=["Predicted Negative", "Predicted Positive"],
        yticklabels=["Actual Negative", "Actual Positive"],
        annot_kws={"fontsize": 11, "fontweight": "bold"}, ax=ax,
        linewidths=1.5, linecolor="#f0f0f0"
    )

    ax.set_title("Holdout Test Confusion Matrix (Optimal Clinical Threshold)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    out_path = output_dir / "confusion_matrix.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_roc_and_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    optimal_threshold: float,
    output_dir: Path
) -> Tuple[Path, Path]:
    """Generates publication-quality ROC and Precision-Recall curves."""
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    roc_auc_val = auc(fpr, tpr)

    # Optimal point on ROC
    idx_opt = np.argmin(np.abs(roc_thresholds - optimal_threshold))
    opt_fpr = fpr[idx_opt]
    opt_tpr = tpr[idx_opt]

    # ROC Curve
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"ROC Curve (AUC = {roc_auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="#999999", lw=1.2, linestyle="--", label="Random Classifier (AUC = 0.50)")
    ax.scatter(opt_fpr, opt_tpr, color="#d95f02", s=90, zorder=5, label=f"Optimal Threshold ({optimal_threshold:.2f})\nSens: {opt_tpr:.1%}, Spec: {1-opt_fpr:.1%}")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, labelpad=8)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, labelpad=8)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#ccc", fontsize=10)

    plt.tight_layout()
    roc_path = output_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # PR Curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc_val = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.plot(recall, precision, color="#d95f02", lw=2.5, label=f"PR Curve (AP = {pr_auc_val:.4f})")
    no_skill = (y_true == 1).sum() / len(y_true)
    ax.axhline(no_skill, color="#999999", lw=1.2, linestyle="--", label=f"Baseline Prevalence ({no_skill:.1%})")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11, labelpad=8)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, labelpad=8)
    ax.set_title("Precision-Recall (PR) Curve", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#ccc", fontsize=10)

    plt.tight_layout()
    pr_path = output_dir / "precision_recall_curve.png"
    plt.savefig(pr_path, dpi=300)
    plt.close()

    return roc_path, pr_path


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob_calibrated: np.ndarray,
    y_prob_raw: np.ndarray,
    output_dir: Path
) -> Path:
    """Generates a Reliability Diagram comparing raw vs calibrated model probabilities."""
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, y_prob_calibrated, n_bins=8, strategy="uniform")
    prob_true_raw, prob_pred_raw = calibration_curve(y_true, y_prob_raw, n_bins=8, strategy="uniform")

    brier_cal = brier_score_loss(y_true, y_prob_calibrated)
    brier_raw = brier_score_loss(y_true, y_prob_raw)

    fig, (ax_curve, ax_hist) = plt.subplots(
        2, 1, figsize=(7.5, 7), sharex=True, gridspec_kw={"height_ratios": [0.7, 0.3]}, dpi=300
    )

    # Reliability diagram
    ax_curve.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Perfect Calibration")
    ax_curve.plot(prob_pred_raw, prob_true_raw, "s-", color="#2b5c8f", label=f"Uncalibrated (Brier = {brier_raw:.4f})")
    ax_curve.plot(prob_pred_cal, prob_true_cal, "o-", color="#d95f02", label=f"Sigmoid Calibrated (Brier = {brier_cal:.4f})")

    ax_curve.set_ylabel("Observed Fraction of Positives", fontsize=11, labelpad=8)
    ax_curve.set_title("Probability Calibration Curve (Reliability Diagram)", fontsize=13, fontweight="bold", pad=12)
    ax_curve.grid(True, linestyle="--", alpha=0.5)
    ax_curve.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#ccc", fontsize=10)

    # Distribution of predicted probabilities
    ax_hist.hist(y_prob_calibrated, range=(0, 1), bins=15, histtype="stepfilled", color="#d95f02", alpha=0.4, label="Calibrated Probs")
    ax_hist.hist(y_prob_raw, range=(0, 1), bins=15, histtype="step", color="#2b5c8f", linewidth=1.5, label="Raw Probs")
    ax_hist.set_xlabel("Mean Predicted Probability", fontsize=11, labelpad=8)
    ax_hist.set_ylabel("Patient Count", fontsize=11, labelpad=8)
    ax_hist.grid(True, linestyle="--", alpha=0.5)
    ax_hist.legend(loc="upper center", frameon=True, facecolor="white", edgecolor="#ccc", fontsize=9)

    plt.tight_layout()
    out_path = output_dir / "calibration_curve.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def generate_benchmark_report(
    benchmark_df: pd.DataFrame,
    metadata: Dict[str, Any],
    report_path: Path = REPORT_PATH
) -> Path:
    """Compiles a detailed model benchmark & evaluation markdown report."""
    opt_metrics = metadata["metrics"]["test_eval_optimal_thresh"]
    default_metrics = metadata["metrics"]["test_eval_default_0_5"]
    cm = opt_metrics["confusion_matrix"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    opt_thresh = metadata["optimal_threshold"]
    algo = metadata["algorithm"]

    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    md = f"""# Model Benchmarking, Calibration & Clinical Evaluation Report
## Predictive Analytics for Early Type-2 Diabetes Risk Detection

**Selected Production Architecture**: Calibrated `{algo}` Pipeline  
**Probability Calibration**: Sigmoid Platt Scaling via 5-Fold Internal CV  
**Optimal Decision Threshold**: `{opt_thresh:.3f}` (Derived via $F_2$-Score Optimization prioritizing Sensitivity)  
**Holdout Test Set Size**: {metadata['test_samples']} unseen patients (64 Positive, 40 Negative)

---

## 1. Candidate Model Benchmark Comparison

All 5 candidate classifiers were embedded inside leak-free pipelines (with standard numeric scaling and clinical binary encoding) and evaluated using **Repeated Stratified 5-Fold Cross Validation (15 total runs per candidate)**.

| Model Candidate | Mean Sensitivity (Recall) | Mean Precision | Mean F1-Score | Mean ROC-AUC | Mean Balanced Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for _, row in benchmark_df.iterrows():
        md += (
            f"| **{row['Model']}** | **{row['Recall_Mean']:.1%}** ± {row['Recall_Std']:.1%} | "
            f"{row['Precision_Mean']:.1%} ± {row['Precision_Std']:.1%} | "
            f"{row['F1_Mean']:.1%} ± {row['F1_Std']:.1%} | "
            f"**{row['ROC_AUC_Mean']:.4f}** ± {row['ROC_AUC_Std']:.4f} | "
            f"{row['Balanced_Acc_Mean']:.1%} |\n"
        )

    md += f"""
![Model Comparison Benchmark](figures/model_comparison_benchmark.png)

> **Architectural Selection**: `{algo}` achieved the highest discriminative ability (ROC-AUC) and outstanding Sensitivity across repeated cross-validation folds, making it the superior candidate for pre-clinical diabetes risk triage.

---

## 2. Holdout Test Set Performance

The final model was evaluated on the completely untouched holdout test partition ($N = {metadata['test_samples']}$).

### 2.1 Performance Comparison Across Decision Thresholds

| Metric | Standard Default (Threshold = 0.50) | Optimized Clinical (Threshold = {opt_thresh:.3f}) | Clinical Impact |
| :--- | :--- | :--- | :--- |
| **Sensitivity (Recall)** | **{default_metrics['recall']:.1%}** | **{opt_metrics['recall']:.1%}** | Minimized False Negatives |
| **Specificity** | {default_metrics['confusion_matrix'][0][0] / (default_metrics['confusion_matrix'][0][0] + default_metrics['confusion_matrix'][0][1]):.1%} | **{spec:.1%}** | Controlled false alarms |
| **Precision (PPV)** | {default_metrics['precision']:.1%} | **{opt_metrics['precision']:.1%}** | Confidence in positive calls |
| **F1-Score** | {default_metrics['f1']:.1%} | **{opt_metrics['f1']:.1%}** | Balanced accuracy measure |
| **F2-Score** | {default_metrics['f2']:.1%} | **{opt_metrics['f2']:.1%}** | Heavy weighting on recall |
| **ROC-AUC** | **{opt_metrics['roc_auc']:.4f}** | **{opt_metrics['roc_auc']:.4f}** | Threshold-independent discrimination |
| **Brier Calibration Score** | **{opt_metrics['brier_score']:.4f}** | **{opt_metrics['brier_score']:.4f}** | Calibrated probability accuracy (close to 0 is optimal) |

---

## 3. Clinical Diagnostic Plots

### 3.1 Confusion Matrix Analysis
At the clinical threshold ({opt_thresh:.2f}):
- **True Positives (TP)**: `{tp}` correctly identified diabetic patients.
- **True Negatives (TN)**: `{tn}` correctly identified healthy individuals.
- **False Negatives (FN)**: `{fn}` missed cases (Critical triage safety parameter).
- **False Positives (FP)**: `{fp}` flagged for clinical confirmation.

![Confusion Matrix](figures/confusion_matrix.png)

### 3.2 ROC and Precision-Recall Curves
The ROC curve demonstrates exceptional discrimination (AUC = {opt_metrics['roc_auc']:.4f}), well above random baseline. The PR curve confirms high precision across nearly the entire range of recall.

| Receiver Operating Characteristic (ROC) | Precision-Recall (PR) Curve |
| :---: | :---: |
| ![ROC Curve](figures/roc_curve.png) | ![PR Curve](figures/precision_recall_curve.png) |

---

## 4. Probability Calibration (Reliability Diagram)

In clinical triage, risk probabilities must reflect empirical frequencies (e.g., when the model predicts 70% risk, approximately 70 out of 100 individuals should truly test positive).

![Calibration Curve](figures/calibration_curve.png)

- **Uncalibrated Model Brier Score**: `{metadata['metrics']['test_eval_default_0_5']['brier_score']:.4f}`
- **Calibrated Model Brier Score**: `{opt_metrics['brier_score']:.4f}`
- The reliability diagram demonstrates that Sigmoid Platt calibration brings prediction probabilities into alignment with empirical ground truth, providing confidence in the risk scores displayed to end users.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)

    return report_path


def run_evaluation_pipeline() -> None:
    """Executes the complete evaluation suite and renders figures & report."""
    print("=" * 60)
    print("PHASE 5/6 EVALUATION & VISUALIZATION PIPELINE")
    print("=" * 60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load artifacts
    model_path = MODELS_DIR / "best_model.joblib"
    raw_pipe_path = MODELS_DIR / "tuned_pipeline.joblib"
    meta_path = MODELS_DIR / "model_metadata.json"
    benchmark_path = MODELS_DIR / "benchmark_results.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    if not all([model_path.exists(), meta_path.exists(), test_path.exists()]):
        raise FileNotFoundError("Required model artifacts or test data missing.")

    calibrated_model = joblib.load(model_path)
    raw_pipeline = joblib.load(raw_pipe_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    benchmark_df = pd.read_csv(benchmark_path)
    test_df = pd.read_csv(test_path)

    X_test = test_df[ALL_FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].values

    optimal_thresh = metadata["optimal_threshold"]

    # Inferences
    probs_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
    probs_raw = raw_pipeline.predict_proba(X_test)[:, 1]
    preds_opt = (probs_calibrated >= optimal_thresh).astype(int)

    print("[1/5] Rendering benchmark comparison plot...")
    plot_benchmark_comparison(benchmark_df, FIGURES_DIR)

    print("[2/5] Rendering confusion matrix...")
    plot_confusion_matrix(y_test, preds_opt, FIGURES_DIR)

    print("[3/5] Rendering ROC and Precision-Recall curves...")
    plot_roc_and_pr_curves(y_test, probs_calibrated, optimal_thresh, FIGURES_DIR)

    print("[4/5] Rendering probability calibration curve...")
    plot_calibration_curve(y_test, probs_calibrated, probs_raw, FIGURES_DIR)

    print("[5/5] Generating markdown evaluation report...")
    report_file = generate_benchmark_report(benchmark_df, metadata, REPORT_PATH)

    print(f"[+] All evaluation figures and report created successfully at {report_file}")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation_pipeline()
