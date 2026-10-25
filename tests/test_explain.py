"""
Unit tests for Explainable AI (XAI) and SHAP explainability module.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import shap

from src.explain import ClinicalSHAPExplainer
from src.preprocessing import ALL_FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"


@pytest.fixture(scope="module")
def shap_explainer():
    """Initializes the ClinicalSHAPExplainer instance."""
    return ClinicalSHAPExplainer()


@pytest.fixture
def sample_patient() -> pd.DataFrame:
    """Creates a sample patient DataFrame for local explanation testing."""
    return pd.DataFrame([{
        "Age": 52,
        "Gender": "Male",
        "Polyuria": "Yes",
        "Polydipsia": "Yes",
        "sudden weight loss": "No",
        "weakness": "Yes",
        "Polyphagia": "No",
        "Genital thrush": "No",
        "visual blurring": "No",
        "Itching": "Yes",
        "Irritability": "No",
        "delayed healing": "Yes",
        "partial paresis": "No",
        "muscle stiffness": "No",
        "Alopecia": "No",
        "Obesity": "No",
    }])


def test_explainer_initialization(shap_explainer):
    """Verifies that the SHAP explainer initializes and loads 16 feature names."""
    assert shap_explainer is not None
    assert len(shap_explainer.feature_names) == 16
    assert "Polyuria" in shap_explainer.feature_names
    assert "Polydipsia" in shap_explainer.feature_names


def test_explain_single_patient_structure(shap_explainer, sample_patient):
    """Verifies that single patient explanation outputs structured attribution details."""
    exp, summary = shap_explainer.explain_single_patient(sample_patient)

    assert isinstance(exp, shap.Explanation)
    assert len(exp.values) == 16
    assert "base_value_log_odds" in summary
    assert "calibrated_risk_probability" in summary
    assert 0.0 <= summary["calibrated_risk_probability"] <= 1.0
    assert len(summary["feature_contributions"]) == 16
    assert len(summary["top_risk_factors"]) > 0


def test_single_patient_additivity(shap_explainer, sample_patient):
    """Verifies local game-theoretic additivity: base_value + sum(phi_i) == patient margin."""
    exp, summary = shap_explainer.explain_single_patient(sample_patient)
    computed_margin = summary["base_value_log_odds"] + np.sum(exp.values)
    np.testing.assert_almost_equal(computed_margin, summary["patient_margin_log_odds"], decimal=4)


def test_waterfall_plot_rendering(shap_explainer, sample_patient):
    """Verifies that waterfall plot returns an active Matplotlib figure."""
    exp, _ = shap_explainer.explain_single_patient(sample_patient)
    fig = shap_explainer.plot_waterfall(exp, title="Unit Test Waterfall")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_dataset_explanations_shape(shap_explainer):
    """Verifies that explain_dataset computes SHAP values for all rows and 16 features."""
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test = test_df[ALL_FEATURE_COLUMNS]
    shap_vals = shap_explainer.explain_dataset(X_test)
    assert shap_vals.shape == (len(test_df), 16)
