"""
Unit tests for model inference, calibration boundaries, thresholding, and pipeline loading.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import ALL_FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"


@pytest.fixture(scope="module")
def loaded_model():
    """Loads the serialized calibrated model pipeline."""
    assert MODEL_PATH.exists(), f"Model file not found at {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def model_metadata():
    """Loads the model metadata."""
    assert METADATA_PATH.exists(), f"Metadata file not found at {METADATA_PATH}"
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def single_patient_sample() -> pd.DataFrame:
    """Creates a sample patient data frame with typical high-risk symptoms."""
    return pd.DataFrame([{
        "Age": 55,
        "Gender": "Male",
        "Polyuria": "Yes",
        "Polydipsia": "Yes",
        "sudden weight loss": "Yes",
        "weakness": "Yes",
        "Polyphagia": "Yes",
        "Genital thrush": "No",
        "visual blurring": "Yes",
        "Itching": "No",
        "Irritability": "Yes",
        "delayed healing": "No",
        "partial paresis": "Yes",
        "muscle stiffness": "No",
        "Alopecia": "No",
        "Obesity": "Yes",
    }])


def test_model_loading_and_metadata_integrity(loaded_model, model_metadata):
    """Verifies that the serialized model loads and metadata is valid."""
    assert loaded_model is not None
    assert "algorithm" in model_metadata
    assert "optimal_threshold" in model_metadata
    assert 0.0 < model_metadata["optimal_threshold"] < 1.0


def test_predict_proba_range_and_shape(loaded_model, single_patient_sample):
    """Verifies that predict_proba outputs valid probabilities between 0 and 1."""
    probs = loaded_model.predict_proba(single_patient_sample)
    assert probs.shape == (1, 2)
    assert 0.0 <= probs[0, 0] <= 1.0
    assert 0.0 <= probs[0, 1] <= 1.0
    np.testing.assert_almost_equal(probs[0, 0] + probs[0, 1], 1.0, decimal=5)


def test_high_risk_patient_classified_positive(loaded_model, model_metadata, single_patient_sample):
    """Verifies that a patient with cardinal symptoms (Polyuria + Polydipsia) receives high risk."""
    prob_pos = loaded_model.predict_proba(single_patient_sample)[0, 1]
    optimal_thresh = model_metadata["optimal_threshold"]
    assert prob_pos > optimal_thresh, f"Expected high risk > {optimal_thresh}, got {prob_pos}"


def test_holdout_test_set_evaluation_bounds(loaded_model, model_metadata):
    """Verifies performance metrics on the test dataset remain above safety thresholds."""
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test = test_df[ALL_FEATURE_COLUMNS]
    y_test = test_df["class"].values

    probs = loaded_model.predict_proba(X_test)[:, 1]
    thresh = model_metadata["optimal_threshold"]
    preds = (probs >= thresh).astype(int)

    # Clinical Safety: Recall must exceed 95%
    tp = np.sum((preds == 1) & (y_test == 1))
    fn = np.sum((preds == 0) & (y_test == 1))
    recall = tp / (tp + fn)
    assert recall >= 0.95, f"Clinical Sensitivity dropped below 95%: {recall:.2%}"
