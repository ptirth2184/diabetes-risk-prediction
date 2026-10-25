"""
Unit and integration tests for application helper functions, risk categorization, and edge cases.
"""

from pathlib import Path
import joblib
import pandas as pd
import pytest

from app.utils import categorize_risk
from src.preprocessing import ALL_FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"


@pytest.fixture(scope="module")
def loaded_pipeline():
    """Loads the calibrated production model pipeline."""
    return joblib.load(MODEL_PATH)


def test_categorize_risk_boundaries():
    """Verifies that categorize_risk accurately classifies across clinical thresholds."""
    threshold = 0.320

    # Low risk
    cat_low, col_low, badge_low, _ = categorize_risk(0.15, threshold=threshold)
    assert cat_low == "Low Risk"
    assert "low" in badge_low

    # Moderate risk
    cat_mod, col_mod, badge_mod, _ = categorize_risk(0.45, threshold=threshold)
    assert cat_mod == "Moderate Risk"
    assert "moderate" in badge_mod

    # High risk
    cat_high, col_high, badge_high, _ = categorize_risk(0.85, threshold=threshold)
    assert "High Risk" in cat_high
    assert "high" in badge_high


def test_all_negative_symptoms_evaluates_low_risk(loaded_pipeline):
    """Verifies that a patient reporting zero symptoms receives a low risk score."""
    patient = {
        "Age": 28,
        "Gender": "Male",
        "Polyuria": "No",
        "Polydipsia": "No",
        "sudden weight loss": "No",
        "weakness": "No",
        "Polyphagia": "No",
        "Genital thrush": "No",
        "visual blurring": "No",
        "Itching": "No",
        "Irritability": "No",
        "delayed healing": "No",
        "partial paresis": "No",
        "muscle stiffness": "No",
        "Alopecia": "No",
        "Obesity": "No",
    }
    df = pd.DataFrame([patient])
    prob = loaded_pipeline.predict_proba(df)[0, 1]
    assert prob < 0.20, f"Expected low risk < 0.20 for asymptomatic patient, got {prob:.3f}"


def test_all_positive_symptoms_evaluates_high_risk(loaded_pipeline):
    """Verifies that a patient reporting all symptoms receives high risk."""
    patient = {
        "Age": 62,
        "Gender": "Female",
        "Polyuria": "Yes",
        "Polydipsia": "Yes",
        "sudden weight loss": "Yes",
        "weakness": "Yes",
        "Polyphagia": "Yes",
        "Genital thrush": "Yes",
        "visual blurring": "Yes",
        "Itching": "Yes",
        "Irritability": "Yes",
        "delayed healing": "Yes",
        "partial paresis": "Yes",
        "muscle stiffness": "Yes",
        "Alopecia": "Yes",
        "Obesity": "Yes",
    }
    df = pd.DataFrame([patient])
    prob = loaded_pipeline.predict_proba(df)[0, 1]
    assert prob > 0.85, f"Expected high risk > 0.85 for pan-symptomatic patient, got {prob:.3f}"


def test_monotonicity_of_cardinal_symptom(loaded_pipeline):
    """Verifies that adding the cardinal symptom (Polyuria) strictly increases predicted risk."""
    base_patient = {
        "Age": 45,
        "Gender": "Male",
        "Polyuria": "No",
        "Polydipsia": "No",
        "sudden weight loss": "No",
        "weakness": "No",
        "Polyphagia": "No",
        "Genital thrush": "No",
        "visual blurring": "No",
        "Itching": "No",
        "Irritability": "No",
        "delayed healing": "No",
        "partial paresis": "No",
        "muscle stiffness": "No",
        "Alopecia": "No",
        "Obesity": "No",
    }
    aggravated_patient = base_patient.copy()
    aggravated_patient["Polyuria"] = "Yes"

    prob_base = loaded_pipeline.predict_proba(pd.DataFrame([base_patient]))[0, 1]
    prob_agg = loaded_pipeline.predict_proba(pd.DataFrame([aggravated_patient]))[0, 1]

    assert prob_agg > prob_base, f"Expected Polyuria to increase risk: {prob_base:.3f} vs {prob_agg:.3f}"


def test_age_extreme_boundaries_inference(loaded_pipeline):
    """Verifies pipeline does not crash on extreme age boundaries (16 and 110)."""
    p_young = {col: "No" for col in ALL_FEATURE_COLUMNS}
    p_young["Age"] = 16
    p_young["Gender"] = "Male"

    p_elder = {col: "No" for col in ALL_FEATURE_COLUMNS}
    p_elder["Age"] = 110
    p_elder["Gender"] = "Female"

    df_young = pd.DataFrame([p_young])
    df_elder = pd.DataFrame([p_elder])

    prob_y = loaded_pipeline.predict_proba(df_young)[0, 1]
    prob_e = loaded_pipeline.predict_proba(df_elder)[0, 1]

    assert 0.0 <= prob_y <= 1.0
    assert 0.0 <= prob_e <= 1.0
