"""
Unit tests for data preprocessing, schema validation, and leak-free transformations.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.preprocessing import (
    ALL_FEATURE_COLUMNS,
    BINARY_VALUE_MAPPING,
    ClinicalBinaryEncoder,
    build_preprocessing_pipeline,
    prepare_data_splits,
    validate_input_schema,
)


@pytest.fixture
def sample_raw_dataframe() -> pd.DataFrame:
    """Creates a synthetic miniature DataFrame conforming to UCI dataset schema."""
    data = {
        "Age": [40, 55, 30, 65, 50, 45, 60, 35, 70, 25],
        "Gender": ["Male", "Female", "Male", "Female", "Male", "Male", "Female", "Male", "Female", "Male"],
        "Polyuria": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
        "Polydipsia": ["Yes", "No", "Yes", "No", "No", "No", "Yes", "No", "Yes", "No"],
        "sudden weight loss": ["No", "Yes", "No", "No", "Yes", "No", "Yes", "No", "No", "No"],
        "weakness": ["Yes", "Yes", "No", "Yes", "Yes", "No", "Yes", "No", "Yes", "No"],
        "Polyphagia": ["No", "Yes", "Yes", "No", "Yes", "No", "Yes", "No", "No", "No"],
        "Genital thrush": ["No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No"],
        "visual blurring": ["No", "Yes", "No", "Yes", "No", "No", "Yes", "No", "No", "No"],
        "Itching": ["Yes", "No", "Yes", "No", "Yes", "Yes", "No", "No", "Yes", "No"],
        "Irritability": ["No", "No", "Yes", "No", "Yes", "No", "No", "No", "Yes", "No"],
        "delayed healing": ["Yes", "Yes", "No", "No", "Yes", "No", "Yes", "No", "No", "No"],
        "partial paresis": ["No", "Yes", "No", "No", "Yes", "No", "Yes", "No", "No", "No"],
        "muscle stiffness": ["Yes", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No"],
        "Alopecia": ["Yes", "No", "Yes", "No", "No", "Yes", "No", "Yes", "No", "Yes"],
        "Obesity": ["Yes", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No"],
        "class": [
            "Positive", "Negative", "Positive", "Negative", "Positive",
            "Negative", "Positive", "Negative", "Positive", "Negative"
        ],
    }
    return pd.DataFrame(data)


def test_binary_encoder_casing_and_whitespace():
    """Verifies that ClinicalBinaryEncoder trims whitespace and handles case insensitivity."""
    encoder = ClinicalBinaryEncoder(mapping=BINARY_VALUE_MAPPING)
    raw_inputs = pd.DataFrame({
        "Polyuria": [" Yes ", "no", "YES", "No ", "1", "0"],
        "Polydipsia": ["no", " YES", " No", "yes", "0", "1"],
    })
    transformed = encoder.fit_transform(raw_inputs)
    expected = np.array([
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1],
    ], dtype=np.int32)
    np.testing.assert_array_equal(transformed, expected)


def test_binary_encoder_invalid_category_raises_error():
    """Verifies that unrecognized values raise an informative ValueError."""
    encoder = ClinicalBinaryEncoder(mapping=BINARY_VALUE_MAPPING)
    bad_data = pd.DataFrame({"Polyuria": ["Yes", "Maybe", "No"]})
    with pytest.raises(ValueError, match="unrecognized binary categories"):
        encoder.fit_transform(bad_data)


def test_schema_validation_missing_columns(sample_raw_dataframe):
    """Verifies schema validation catches missing features."""
    incomplete_df = sample_raw_dataframe.drop(columns=["Polyuria"])
    with pytest.raises(ValueError, match="missing required feature columns"):
        validate_input_schema(incomplete_df, is_inference=False)


def test_schema_validation_invalid_age(sample_raw_dataframe):
    """Verifies schema validation rejects physiologically impossible age values."""
    bad_age_df = sample_raw_dataframe.copy()
    bad_age_df.loc[0, "Age"] = -5
    with pytest.raises(ValueError, match="Invalid values in 'Age'"):
        validate_input_schema(bad_age_df, is_inference=False)


def test_stratified_split_preserves_ratios(sample_raw_dataframe):
    """Verifies that prepare_data_splits perfectly balances classes across partitions."""
    X_train, X_test, y_train, y_test = prepare_data_splits(
        sample_raw_dataframe, test_size=0.3, random_state=42
    )
    assert len(X_train) == 7
    assert len(X_test) == 3
    # Check stratification
    train_pos_ratio = y_train.mean()
    test_pos_ratio = y_test.mean()
    # In 10 samples (5 pos, 5 neg): split of 7 and 3 gives 4/7 or 3/7 vs 1/3 or 2/3
    assert abs(train_pos_ratio - 0.5) < 0.15
    assert abs(test_pos_ratio - 0.5) < 0.20


def test_zero_data_leakage_in_preprocessing_pipeline(sample_raw_dataframe):
    """Verifies that the ColumnTransformer calculates scaler parameters strictly on X_train."""
    X_train, X_test, _y_train, _y_test = prepare_data_splits(
        sample_raw_dataframe, test_size=0.4, random_state=42
    )

    pipeline = build_preprocessing_pipeline(scale_numeric=True)
    X_train_trans = pipeline.fit_transform(X_train)
    X_test_trans = pipeline.transform(X_test)

    # Extract the fitted numeric scaler
    fitted_scaler = pipeline.named_transformers_["numeric"]
    assert isinstance(fitted_scaler, StandardScaler)

    # Scaler mean must match X_train["Age"].mean() exactly, NOT whole dataset mean
    np.testing.assert_almost_equal(fitted_scaler.mean_[0], X_train["Age"].mean(), decimal=5)
    assert X_train_trans.shape == (len(X_train), len(ALL_FEATURE_COLUMNS))
    assert X_test_trans.shape == (len(X_test), len(ALL_FEATURE_COLUMNS))
