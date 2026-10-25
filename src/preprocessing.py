"""
Preprocessing and Data Ingestion Pipeline Module.

Architectural Guarantees:
1. Strict Data Leakage Prevention: Transformers fit exclusively on training splits.
2. Production Modularity: Full Scikit-learn BaseEstimator & TransformerMixin compliance.
3. Explicit Schema & Type Enforcement: Handles missing values, case sensitivity, and whitespace.
4. Reproducible Stratification: Stratified 80/20 train/test split with deterministic seed.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Canonical Schema Definitions
NUMERIC_FEATURES: list[str] = ["Age"]

BINARY_SYMPTOM_FEATURES: list[str] = [
    "Polyuria",
    "Polydipsia",
    "sudden weight loss",
    "weakness",
    "Polyphagia",
    "Genital thrush",
    "visual blurring",
    "Itching",
    "Irritability",
    "delayed healing",
    "partial paresis",
    "muscle stiffness",
    "Alopecia",
    "Obesity",
]

DEMOGRAPHIC_CATEGORICAL_FEATURES: list[str] = ["Gender"]

ALL_FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + DEMOGRAPHIC_CATEGORICAL_FEATURES + BINARY_SYMPTOM_FEATURES
TARGET_COLUMN: str = "class"

TARGET_MAPPING: dict[str, int] = {"Negative": 0, "Positive": 1}
INVERSE_TARGET_MAPPING: dict[int, str] = {0: "Negative", 1: "Positive"}

BINARY_VALUE_MAPPING: dict[str, int] = {
    "no": 0,
    "yes": 1,
    "n": 0,
    "y": 1,
    "false": 0,
    "true": 1,
    "0": 0,
    "1": 1,
}

GENDER_VALUE_MAPPING: dict[str, int] = {
    "male": 0,
    "female": 1,
    "m": 0,
    "f": 1,
}


class ClinicalBinaryEncoder(BaseEstimator, TransformerMixin):
    """
    Robust scikit-learn transformer for clinical binary questionnaire responses.
    Handles case-insensitivity, whitespace trimming, and string/numeric types.
    """

    def __init__(self, mapping: dict[str, int] | None = None):
        self.mapping = mapping if mapping is not None else BINARY_VALUE_MAPPING
        self.feature_names_in_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ClinicalBinaryEncoder":
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            X_df = X.copy()

        transformed = pd.DataFrame(index=X_df.index)
        for col in X_df.columns:
            series = X_df[col].astype(str).str.strip().str.lower()
            mapped = series.map(self.mapping)
            if mapped.isnull().any():
                invalid_vals = series[mapped.isnull()].unique()
                raise ValueError(
                    f"Column '{col}' contains unrecognized binary categories: {invalid_vals}. "
                    f"Expected values matching mapping keys: {list(self.mapping.keys())}"
                )
            transformed[col] = mapped.astype(np.int32)

        return transformed.to_numpy()

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        if input_features is not None:
            return np.array(input_features, dtype=object)
        if self.feature_names_in_ is not None:
            return np.array(self.feature_names_in_, dtype=object)
        return np.array([], dtype=object)


def build_preprocessing_pipeline(
    scale_numeric: bool = True,
    use_robust_scaler: bool = False
) -> ColumnTransformer:
    """
    Builds an isolated Scikit-learn ColumnTransformer.
    
    Ensures zero data leakage: fitting must only occur on training folds.
    """
    numeric_scaler = RobustScaler() if use_robust_scaler else StandardScaler()
    
    transformers = [
        (
            "numeric",
            numeric_scaler if scale_numeric else "passthrough",
            NUMERIC_FEATURES,
        ),
        (
            "gender",
            ClinicalBinaryEncoder(mapping=GENDER_VALUE_MAPPING),
            DEMOGRAPHIC_CATEGORICAL_FEATURES,
        ),
        (
            "symptoms",
            ClinicalBinaryEncoder(mapping=BINARY_VALUE_MAPPING),
            BINARY_SYMPTOM_FEATURES,
        ),
    ]

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def validate_input_schema(df: pd.DataFrame, is_inference: bool = False) -> pd.DataFrame:
    """
    Validates input data schema against canonical feature definitions.
    Strips whitespace from column names and verifies required fields exist.
    """
    cleaned_df = df.copy()
    cleaned_df.columns = cleaned_df.columns.str.strip()

    missing_features = [col for col in ALL_FEATURE_COLUMNS if col not in cleaned_df.columns]
    if missing_features:
        raise ValueError(f"Input data is missing required feature columns: {missing_features}")

    if not is_inference and TARGET_COLUMN not in cleaned_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is required for training data.")

    # Validate Age boundaries
    if (cleaned_df["Age"] <= 0).any() or (cleaned_df["Age"] > 120).any():
        raise ValueError("Invalid values in 'Age'. Age must be strictly between 1 and 120.")

    return cleaned_df


def prepare_data_splits(
    raw_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs stratified train/test split.
    Guarantees consistent target class ratios in both partitions.
    """
    validated_df = validate_input_schema(raw_df, is_inference=False)
    
    X = validated_df[ALL_FEATURE_COLUMNS].copy()
    # Map target strings to integer binary: Positive -> 1, Negative -> 0
    y_raw = validated_df[TARGET_COLUMN].astype(str).str.strip()
    y = y_raw.map(TARGET_MAPPING)

    if y.isnull().any():
        unmapped = y_raw[y.isnull()].unique()
        raise ValueError(f"Target column contains invalid classes: {unmapped}. Expected: {list(TARGET_MAPPING.keys())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
        shuffle=True,
    )

    logger.info(
        f"Data split complete: Train={len(X_train)} samples ({y_train.mean():.1%} positive), "
        f"Test={len(X_test)} samples ({y_test.mean():.1%} positive)"
    )
    return X_train, X_test, y_train, y_test


def export_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path
) -> dict[str, Any]:
    """
    Persists train/test splits and dataset metadata into data/processed/.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train.values

    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test.values

    train_csv_path = output_dir / "train.csv"
    test_csv_path = output_dir / "test.csv"
    manifest_path = output_dir / "dataset_manifest.json"

    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    manifest = {
        "random_state": 42,
        "test_size": 0.2,
        "stratified": True,
        "features": {
            "numeric": NUMERIC_FEATURES,
            "demographic": DEMOGRAPHIC_CATEGORICAL_FEATURES,
            "symptoms": BINARY_SYMPTOM_FEATURES,
            "total_count": len(ALL_FEATURE_COLUMNS),
        },
        "target": {
            "name": TARGET_COLUMN,
            "mapping": TARGET_MAPPING,
        },
        "dataset_splits": {
            "train": {
                "total_samples": len(train_df),
                "positive_count": int((y_train == 1).sum()),
                "negative_count": int((y_train == 0).sum()),
                "positive_ratio": float(y_train.mean()),
            },
            "test": {
                "total_samples": len(test_df),
                "positive_count": int((y_test == 1).sum()),
                "negative_count": int((y_test == 0).sum()),
                "positive_ratio": float(y_test.mean()),
            },
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    logger.info(f"Saved processed datasets to: {output_dir}")
    return manifest


def run_pipeline() -> None:
    """Executes the Phase 4 preprocessing pipeline end-to-end."""
    project_root = Path(__file__).resolve().parent.parent
    raw_path = project_root / "data" / "raw" / "diabetes_data_upload.csv"
    processed_dir = project_root / "data" / "processed"

    print("=" * 60)
    print("PHASE 4: PREPROCESSING PIPELINE & LEAK-FREE SPLITTING")
    print("=" * 60)

    raw_df = pd.read_csv(raw_path)
    X_train, X_test, y_train, y_test = prepare_data_splits(raw_df, test_size=0.2, random_state=42)

    # Verify pipeline fit on training split only
    preprocessor = build_preprocessing_pipeline(scale_numeric=True)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    print(f"[+] Fitted ColumnTransformer on train split only ({X_train_trans.shape[0]}x{X_train_trans.shape[1]}).")
    print(f"[+] Transformed test split without leakage ({X_test_trans.shape[0]}x{X_test_trans.shape[1]}).")
    print(f"[+] Transformed feature names: {list(feature_names)}")

    export_processed_data(X_train, X_test, y_train, y_test, processed_dir)
    print(f"[+] Processed splits and manifest saved to {processed_dir}.")
    print("=" * 60)
    print("PHASE 4 COMPLETE: Preprocessing module validated.")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
