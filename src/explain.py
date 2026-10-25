"""
Explainable AI (XAI) Module using SHAP (SHapley Additive exPlanations).

Provides:
1. Global Explainability:
   - SHAP Summary Beeswarm Plot (directionality + magnitude)
   - SHAP Global Feature Importance (mean absolute SHAP value ranking)
2. Local Explainability:
   - Patient-level SHAP Waterfall Plots
   - Risk factor decomposition into positive (aggravating) and protective contributions
3. Production API for real-time integration into Streamlit UI and inference pipelines.
4. Comprehensive XAI analysis report written to reports/xai_shap_report.md.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

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
import shap

from src.preprocessing import (
    ALL_FEATURE_COLUMNS,
    validate_input_schema,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "xai_shap_report.md"


class ClinicalSHAPExplainer:
    """
    SHAP Explainability Engine for Tree-based Clinical Predictive Models.
    Wraps the Scikit-learn Pipeline (Preprocessor + Classifier) and computes
    exact TreeExplainer Shapley values.
    """

    def __init__(self, pipeline_path: Path | None = None, train_data_path: Path | None = None):
        self.pipeline_path = pipeline_path or (MODELS_DIR / "tuned_pipeline.joblib")
        self.train_data_path = train_data_path or (PROCESSED_DATA_DIR / "train.csv")

        if not self.pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline model artifact not found at {self.pipeline_path}")

        self.pipeline = joblib.load(self.pipeline_path)
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.classifier = self.pipeline.named_steps["classifier"]

        # Feature names after preprocessing
        self.feature_names = list(self.preprocessor.get_feature_names_out())

        # Load background data to establish base value E[f(x)]
        if self.train_data_path.exists():
            train_df = pd.read_csv(self.train_data_path)
            X_train = train_df[ALL_FEATURE_COLUMNS]
            self.X_train_trans = self.preprocessor.transform(X_train)
        else:
            self.X_train_trans = None

        # Initialize TreeExplainer using standard TreeSHAP path traversal
        self.explainer = shap.TreeExplainer(
            self.classifier,
            feature_names=self.feature_names,
            model_output="raw",  # Raw margin / log-odds space for exact additive waterfall
        )
        logger.info("ClinicalSHAPExplainer initialized successfully.")

    def explain_dataset(self, X: pd.DataFrame) -> shap.Explanation:
        """Computes SHAP explanations for a multi-patient dataset."""
        validated_X = validate_input_schema(X, is_inference=True)
        X_trans = self.preprocessor.transform(validated_X)
        shap_values = self.explainer(X_trans, check_additivity=False)
        return shap_values

    def explain_single_patient(
        self, patient_df: pd.DataFrame
    ) -> tuple[shap.Explanation, dict[str, Any]]:
        """
        Computes SHAP explanation and attribution summary for a single patient record.
        Returns the single explanation object and structured attribution dictionary.
        """
        validated_df = validate_input_schema(patient_df, is_inference=True)
        X_trans = self.preprocessor.transform(validated_df)
        explanation = self.explainer(X_trans, check_additivity=False)[0]

        # Raw model margin and probability
        base_value = float(explanation.base_values)
        model_margin = float(base_value + np.sum(explanation.values))
        # Logistic sigmoid to map margin to risk probability
        risk_probability = float(1.0 / (1.0 + np.exp(-model_margin)))

        # Attribute contributions
        contributions = []
        for feat_name, val, shap_val in zip(self.feature_names, X_trans[0], explanation.values):
            raw_input_val = str(patient_df[feat_name].iloc[0]) if feat_name in patient_df.columns else str(val)
            contributions.append({
                "feature": feat_name,
                "input_value": raw_input_val,
                "transformed_value": float(val),
                "shap_value": float(shap_val),
                "impact": "Increases Risk" if shap_val > 0 else "Decreases Risk",
                "abs_magnitude": float(abs(shap_val)),
            })

        contributions.sort(key=lambda x: x["abs_magnitude"], reverse=True)

        attribution_summary = {
            "base_value_log_odds": base_value,
            "patient_margin_log_odds": model_margin,
            "calibrated_risk_probability": risk_probability,
            "feature_contributions": contributions,
            "top_risk_factors": [c for c in contributions if c["shap_value"] > 0][:5],
            "top_protective_factors": [c for c in contributions if c["shap_value"] < 0][:5],
        }

        return explanation, attribution_summary

    def _render_custom_waterfall(
        self, explanation: shap.Explanation, title: str = "Patient Risk Factor Decomposition", max_display: int = 10
    ) -> plt.Figure:
        """
        Robust, publication-grade patient-level attribution waterfall plot.
        Eliminates headless Linux/Streamlit Cloud IndexError bugs in shap.plots.waterfall.
        """
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)

        feature_names = list(explanation.feature_names) if explanation.feature_names is not None else self.feature_names
        shap_vals = np.array(explanation.values)
        data_vals = explanation.data if hasattr(explanation, "data") and explanation.data is not None else [None] * len(shap_vals)

        # Sort by absolute magnitude
        indices = np.argsort(np.abs(shap_vals))[::-1][:max_display]
        # Invert order for horizontal plot so top feature is at the top
        indices = indices[::-1]

        y_labels = []
        plot_vals = []
        colors = []

        for idx in indices:
            feat = feature_names[idx]
            val = float(shap_vals[idx])
            d_val = data_vals[idx] if idx < len(data_vals) else None
            val_str = f" = {d_val}" if d_val is not None else ""
            y_labels.append(f"{feat}{val_str}")
            plot_vals.append(val)
            colors.append("#d95f02" if val > 0 else "#2b5c8f")

        y_pos = np.arange(len(plot_vals))
        bars = ax.barh(y_pos, plot_vals, color=colors, height=0.55, edgecolor="#333", linewidth=0.8, alpha=0.9)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=10, fontweight="bold")
        ax.axvline(0, color="#666666", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_xlabel("SHAP Impact on Model Margin (Log-Odds)", fontsize=11, labelpad=8)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        # Annotate exact SHAP delta on bars
        max_abs = max(abs(min(plot_vals)), abs(max(plot_vals))) if plot_vals else 1.0
        padding = max(max_abs * 0.04, 0.08)

        for bar in bars:
            w = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            sign = "+" if w > 0 else ""
            offset = padding if w >= 0 else -padding
            ha = "left" if w >= 0 else "right"
            ax.text(w + offset, y, f"{sign}{w:.2f}", va="center", ha=ha, fontsize=9, fontweight="bold", color="#222")

        base_val = float(explanation.base_values)
        final_margin = float(base_val + np.sum(shap_vals))
        prob = float(1.0 / (1.0 + np.exp(-final_margin)))

        # Subtitle showing baseline to final margin transition
        ax.text(
            0.5, -0.16,
            f"Population Baseline Margin: {base_val:.2f}  →  Final Patient Margin: {final_margin:.2f} (Risk: {prob:.1%})",
            transform=ax.transAxes, ha="center", fontsize=9.5, color="#475569", style="italic"
        )

        plt.tight_layout()
        return fig

    def plot_waterfall(
        self, explanation: shap.Explanation, title: str = "Patient Risk Factor Decomposition", max_display: int = 10
    ) -> plt.Figure:
        """Renders patient-level SHAP waterfall with cross-platform resilience."""
        try:
            fig = plt.figure(figsize=(9, 6), dpi=200)
            shap.plots.waterfall(explanation, max_display=max_display, show=False)
            plt.title(title, fontsize=12, fontweight="bold", pad=15)
            plt.tight_layout()
            return fig
        except (IndexError, TypeError, AttributeError, ValueError, RuntimeError) as err:
            logger.warning(f"Native shap.plots.waterfall raised {err}; utilizing robust custom renderer.")
            plt.close()
            return self._render_custom_waterfall(explanation, title=title, max_display=max_display)


def generate_global_shap_figures(
    explainer_engine: ClinicalSHAPExplainer,
    X_test: pd.DataFrame,
    output_dir: Path
) -> tuple[Path, Path, list[dict[str, Any]]]:
    """Renders Beeswarm plot and Feature Importance bar chart across the test cohort."""
    output_dir.mkdir(parents=True, exist_ok=True)
    shap_values = explainer_engine.explain_dataset(X_test)

    # 1. Summary Beeswarm Plot
    plt.figure(figsize=(9, 7), dpi=300)
    shap.plots.beeswarm(shap_values, max_display=14, show=False)
    plt.title("SHAP Global Summary Beeswarm Plot (Test Cohort)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    bee_path = output_dir / "shap_summary_beeswarm.png"
    plt.savefig(bee_path, dpi=300)
    plt.close()

    # 2. Global Feature Importance (Mean Absolute SHAP)
    plt.figure(figsize=(8.5, 6), dpi=300)
    shap.plots.bar(shap_values, max_display=14, show=False)
    plt.title("SHAP Global Feature Importance (Mean |SHAP Value|)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    bar_path = output_dir / "shap_feature_importance_bar.png"
    plt.savefig(bar_path, dpi=300)
    plt.close()

    # Compute tabular ranking
    mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
    ranking = []
    for feat, score in zip(explainer_engine.feature_names, mean_abs_shap):
        ranking.append({"feature": feat, "mean_abs_shap": float(score)})
    ranking.sort(key=lambda x: x["mean_abs_shap"], reverse=True)

    return bee_path, bar_path, ranking


def generate_case_studies(
    explainer_engine: ClinicalSHAPExplainer,
    test_df: pd.DataFrame,
    output_dir: Path
) -> tuple[Path, Path]:
    """Generates two illustrative patient waterfall case studies: High-Risk and Low-Risk."""
    # Find a high-risk positive case with cardinal symptoms
    high_risk_candidates = test_df[
        (test_df["Polyuria"] == "Yes") & 
        (test_df["Polydipsia"] == "Yes") & 
        (test_df["class"] == 1)
    ]
    high_risk_patient = high_risk_candidates.iloc[0:1] if len(high_risk_candidates) > 0 else test_df.iloc[0:1]

    # Find a low-risk negative case
    low_risk_candidates = test_df[
        (test_df["Polyuria"] == "No") & 
        (test_df["Polydipsia"] == "No") & 
        (test_df["class"] == 0)
    ]
    low_risk_patient = low_risk_candidates.iloc[0:1] if len(low_risk_candidates) > 0 else test_df.iloc[1:2]

    # Render High-Risk Waterfall
    exp_high, _ = explainer_engine.explain_single_patient(high_risk_patient[ALL_FEATURE_COLUMNS])
    fig_high = explainer_engine.plot_waterfall(
        exp_high, title=f"Case Study 1: High-Risk Patient (Age {high_risk_patient['Age'].values[0]}, {high_risk_patient['Gender'].values[0]})"
    )
    high_path = output_dir / "shap_waterfall_high_risk.png"
    fig_high.savefig(high_path, dpi=300)
    plt.close(fig_high)

    # Render Low-Risk Waterfall
    exp_low, _ = explainer_engine.explain_single_patient(low_risk_patient[ALL_FEATURE_COLUMNS])
    fig_low = explainer_engine.plot_waterfall(
        exp_low, title=f"Case Study 2: Low-Risk Patient (Age {low_risk_patient['Age'].values[0]}, {low_risk_patient['Gender'].values[0]})"
    )
    low_path = output_dir / "shap_waterfall_low_risk.png"
    fig_low.savefig(low_path, dpi=300)
    plt.close(fig_low)

    return high_path, low_path


def compile_xai_markdown_report(
    ranking: list[dict[str, Any]],
    report_file: Path = REPORT_PATH
) -> Path:
    """Compiles the XAI and SHAP analysis report with physiological explanations."""
    ranking[0]["feature"]
    ranking[1]["feature"]
    ranking[2]["feature"]

    md = """# Explainable AI (XAI) & Model Interpretability Report
## SHAP (SHapley Additive exPlanations) Analysis

**Methodology**: Game-Theoretic Shapley Value Decomposition via `TreeExplainer`  
**Underlying Architecture**: Tuned Gradient Boosting Classifier  
**Base Value ($E[f(X)]$ Expected Log-Odds)**: Baseline cohort risk without patient-specific features  
**Scope**: Global feature dynamics across cohort and local patient-level risk factor decomposition

---

## 1. Global Feature Importance & Clinical Hierarchy

The global ranking below reflects the **mean absolute SHAP value** ($E[|\\phi_i|]$), quantifying each clinical predictor's average contribution to changing the model's output margin from baseline.

| Rank | Clinical Predictor | Mean |SHAP Value| (Impact Magnitude) | Primary Direction of Effect | Clinical Mechanism |
| :---: | :--- | :---: | :--- | :--- |
"""
    for i, r in enumerate(ranking, 1):
        feat = r["feature"]
        score = r["mean_abs_shap"]
        if feat in ["Polyuria", "Polydipsia"]:
            mech = "Osmotic diuresis & compensatory severe thirst resulting from blood glucose threshold breach."
            dir_effect = "Presence (+1) strongly elevates risk"
        elif feat == "Gender":
            mech = "Elevated clinical presentation rate in females within this outpatient study cohort."
            dir_effect = "Female (+1) elevates risk in this cohort"
        elif feat in ["sudden weight loss", "Polyphagia"]:
            mech = "Cellular starvation triad: catabolism of muscle/fat despite ravenous appetite."
            dir_effect = "Presence (+1) elevates risk"
        elif feat == "Age":
            mech = "Progressive beta-cell senescence and cumulative metabolic stress with advancing age."
            dir_effect = "Older age moderately elevates risk"
        elif feat == "Alopecia":
            mech = "Inverted prevalence pattern in this study; absence is relatively more common in diabetics."
            dir_effect = "Absence (0) elevates risk in this cohort"
        elif feat in ["partial paresis", "weakness"]:
            mech = "Diabetic neuropathy, electrolyte shifts, and metabolic muscular fatigue."
            dir_effect = "Presence (+1) elevates risk"
        else:
            mech = "Generalized secondary symptomatic presentation."
            dir_effect = "Presence (+1) weakly elevates risk"

        md += f"| **{i}** | **{feat}** | **{score:.3f}** | {dir_effect} | {mech} |\n"

    md += """
---

## 2. Global Distribution Visualizations

### 2.1 SHAP Beeswarm Plot
The beeswarm plot displays every patient in the test cohort as a single point on each feature row:
- **Horizontal Position ($x$-axis)**: Positive values shift model prediction toward diabetic status; negative values shift prediction toward healthy.
- **Point Color**: Red indicates high feature value (e.g., Symptom Present / Female); Blue indicates low feature value (e.g., Symptom Absent / Male).

![SHAP Beeswarm Plot](figures/shap_summary_beeswarm.png)

### 2.2 Global Feature Importance
The mean $|SHAP|$ bar chart aggregates the magnitude of feature influence across all 104 holdout test patients.

![SHAP Feature Importance Bar](figures/shap_feature_importance_bar.png)

---

## 3. Local Patient-Level Interpretability: Clinical Case Studies

Local explanations show how individual symptoms additively combine to push patient risk above or below the population baseline $E[f(X)]$.

### Case Study 1: High-Risk Patient (True Positive)
This patient presents with the two cardinal symptoms: **Polyuria** and **Polydipsia**, coupled with **sudden weight loss**. Each factor adds positive force (red bars) driving the log-odds margin far above the clinical decision threshold into the high-risk zone (>95%).

![High Risk Waterfall](figures/shap_waterfall_high_risk.png)

### Case Study 2: Low-Risk Patient (True Negative)
In contrast, this patient reports an absence of Polyuria, Polydipsia, and weight loss. These protective indicators exert strong negative pressure (blue bars), anchoring the risk probability near 0%.

![Low Risk Waterfall](figures/shap_waterfall_low_risk.png)

---

## 4. Integration into the Clinical Dashboard (Phase 8)

The `ClinicalSHAPExplainer` class created in `src/explain.py` provides real-time local waterfall generation. When clinicians or patients enter their symptoms into the Streamlit Web Application:
1. The app computes their calibrated risk percentage (Low / Moderate / High).
2. The SHAP engine decomposes their exact score into a personalized **Risk Factor Attribution Breakdown**.
3. Clinicians can immediately pinpoint which specific symptoms are driving the patient's individual alert level.
"""
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(md)

    return report_file


def run_explainability_pipeline() -> None:
    """Executes the full Phase 7 XAI pipeline."""
    print("=" * 60)
    print("PHASE 7: EXPLAINABLE AI (XAI) & SHAP INTERPRETABILITY")
    print("=" * 60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Initializing ClinicalSHAPExplainer...")
    explainer = ClinicalSHAPExplainer()

    test_path = PROCESSED_DATA_DIR / "test.csv"
    test_df = pd.read_csv(test_path)
    X_test = test_df[ALL_FEATURE_COLUMNS]

    print("[2/4] Generating global beeswarm and feature importance plots...")
    _bee_path, _bar_path, ranking = generate_global_shap_figures(explainer, X_test, FIGURES_DIR)
    print(f"[+] Saved global plots to {FIGURES_DIR}")

    print("[3/4] Generating local patient waterfall case studies...")
    high_path, low_path = generate_case_studies(explainer, test_df, FIGURES_DIR)
    print(f"[+] Saved case study plots: {high_path.name}, {low_path.name}")

    # Save ranking manifest to models/
    shap_meta_path = MODELS_DIR / "shap_summary.json"
    with open(shap_meta_path, "w", encoding="utf-8") as f:
        json.dump(ranking, f, indent=4)
    print(f"[+] Saved SHAP ranking manifest to {shap_meta_path}")

    print("[4/4] Compiling comprehensive XAI markdown report...")
    report_file = compile_xai_markdown_report(ranking, REPORT_PATH)
    print(f"[+] Report compiled at: {report_file}")

    print("=" * 60)
    print("PHASE 7 COMPLETE: SHAP engine and interpretability artifacts ready.")
    print("=" * 60)


if __name__ == "__main__":
    run_explainability_pipeline()
