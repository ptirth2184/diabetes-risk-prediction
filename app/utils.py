"""
UI Utilities, Styling & Cached Artifact Loaders for Streamlit Clinical Application.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import streamlit as st

from src.explain import ClinicalSHAPExplainer

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


@st.cache_resource(show_spinner="Loading predictive engine and explainability models...")
def load_all_artifacts() -> Tuple[Any, Dict[str, Any], ClinicalSHAPExplainer, list]:
    """Loads and caches the calibrated model, metadata, SHAP explainer, and feature hierarchy."""
    model_path = MODELS_DIR / "best_model.joblib"
    meta_path = MODELS_DIR / "model_metadata.json"
    shap_meta_path = MODELS_DIR / "shap_summary.json"

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Model artifacts not found. Please ensure Phase 5/6 has run.")

    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    explainer = ClinicalSHAPExplainer()

    ranking = []
    if shap_meta_path.exists():
        with open(shap_meta_path, "r", encoding="utf-8") as f:
            ranking = json.load(f)

    return model, metadata, explainer, ranking


@st.cache_data
def load_benchmark_and_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads the benchmark comparison CSV and holdout test set for exploration."""
    benchmark_path = MODELS_DIR / "benchmark_results.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    benchmark_df = pd.read_csv(benchmark_path) if benchmark_path.exists() else pd.DataFrame()
    test_df = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()

    return benchmark_df, test_df


def categorize_risk(prob: float, threshold: float = 0.320) -> Tuple[str, str, str, str]:
    """
    Categorizes calibrated risk probability into clinical triage tiers:
    Returns (Category, ColorHex, BadgeClass, ClinicalRecommendation)
    """
    if prob < threshold:
        return (
            "Low Risk",
            "#2e7d32",
            "badge-low",
            "Routine annual screening recommended. No acute osmotic signs detected. Maintain healthy diet and active lifestyle."
        )
    elif prob < 0.70:
        return (
            "Moderate Risk",
            "#ed6c02",
            "badge-moderate",
            "Elevated risk indicators observed. A confirmatory Fasting Plasma Glucose (FPG) or Glycated Hemoglobin (HbA1c) test with a physician is strongly advised."
        )
    else:
        return (
            "High Risk (Critical Alert)",
            "#d32f2f",
            "badge-high",
            "Multiple cardinal diabetic symptoms present. Immediate formal medical consultation and comprehensive laboratory diagnostic panel are urgently recommended."
        )


def get_custom_css() -> str:
    """Returns refined modern CSS for a premium medical application theme."""
    return """
    <style>
    /* Global Container Adjustments */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }
    
    /* Hero Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0d3b66 0%, #1d4e89 50%, #0077b6 100%);
        border-radius: 12px;
        padding: 24px 30px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(13, 59, 102, 0.15);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.9;
        margin-top: 6px;
        font-weight: 400;
    }

    /* Risk Result Card */
    .risk-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 22px;
        border: 1px solid #e0e6ed;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    /* Badges */
    .badge-low {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #c8e6c9;
    }
    .badge-moderate {
        background-color: #fff3e0;
        color: #e65100;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #ffe0b2;
    }
    .badge-high {
        background-color: #ffebee;
        color: #c62828;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #ffcdd2;
    }

    /* Metric Cards */
    .metric-card {
        background-color: #f8fafc;
        border-left: 4px solid #1d4e89;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0f172a;
    }

    /* Medical Disclaimer Banner */
    .disclaimer-box {
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-left: 4px solid #475569;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #334155;
        margin-top: 25px;
    }
    </style>
    """
