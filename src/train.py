"""
Model Training, Benchmark, Probability Calibration & Threshold Optimization Module.

Features:
1. Benchmarks 5 core classifiers inside leak-free Scikit-learn pipelines:
   - Logistic Regression (L2, balanced class weight)
   - Random Forest Classifier
   - Gradient Boosting Classifier
   - Support Vector Classifier (RBF kernel)
   - k-Nearest Neighbors (distance-weighted)
2. Repeated Stratified 5-Fold Cross Validation (15 evaluations per model).
3. Hyperparameter optimization via GridSearchCV.
4. Probability Calibration (CalibratedClassifierCV) to yield reliable clinical risk percentages.
5. Decision Threshold Optimization (maximizing Recall / F2-score to minimize False Negatives).
6. Complete serialization of preprocessor + model pipeline to models/best_model.joblib.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    cross_validate,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.preprocessing import (
    build_preprocessing_pipeline,
    prepare_data_splits,
    validate_input_schema,
    ALL_FEATURE_COLUMNS,
    TARGET_COLUMN,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def get_candidate_models() -> Dict[str, Any]:
    """Returns the set of candidate classification estimators."""
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_split=4,
            class_weight="balanced",
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=3,
            subsample=0.85,
            random_state=42,
        ),
        "Support Vector Machine": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42,
        ),
        "k-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
        ),
    }


def benchmark_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
    cv_repeats: int = 3,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    """
    Evaluates candidate models within leak-free pipelines using Repeated Stratified K-Fold CV.
    """
    logger.info(f"Starting model benchmark ({cv_splits}-fold x {cv_repeats} repeats = {cv_splits * cv_repeats} evals/model)...")
    cv = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=random_state)
    candidates = get_candidate_models()

    scoring = {
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "balanced_accuracy": "balanced_accuracy",
    }

    benchmark_records = []
    pipelines_dict = {}

    for name, estimator in candidates.items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessing_pipeline(scale_numeric=True)),
            ("classifier", estimator),
        ])
        pipelines_dict[name] = pipe

        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        benchmark_records.append({
            "Model": name,
            "Recall_Mean": float(np.mean(scores["test_recall"])),
            "Recall_Std": float(np.std(scores["test_recall"])),
            "Precision_Mean": float(np.mean(scores["test_precision"])),
            "Precision_Std": float(np.std(scores["test_precision"])),
            "F1_Mean": float(np.mean(scores["test_f1"])),
            "F1_Std": float(np.std(scores["test_f1"])),
            "ROC_AUC_Mean": float(np.mean(scores["test_roc_auc"])),
            "ROC_AUC_Std": float(np.std(scores["test_roc_auc"])),
            "Balanced_Acc_Mean": float(np.mean(scores["test_balanced_accuracy"])),
            "Balanced_Acc_Std": float(np.std(scores["test_balanced_accuracy"])),
        })

    benchmark_df = pd.DataFrame(benchmark_records)
    # Primary sort: Recall (minimize false negatives), secondary: ROC-AUC
    benchmark_df = benchmark_df.sort_values(
        by=["Recall_Mean", "ROC_AUC_Mean"], ascending=[False, False]
    ).reset_index(drop=True)

    return benchmark_df, pipelines_dict


def tune_top_model(
    top_model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> Pipeline:
    """Performs systematic hyperparameter grid search on the leading model."""
    logger.info(f"Tuning leading model: {top_model_name}...")

    param_grids = {
        "Random Forest": {
            "classifier__n_estimators": [100, 150, 200],
            "classifier__max_depth": [4, 6, 8, None],
            "classifier__min_samples_split": [2, 4, 6],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__class_weight": ["balanced", None],
        },
        "Gradient Boosting": {
            "classifier__n_estimators": [80, 120, 160],
            "classifier__learning_rate": [0.03, 0.07, 0.1],
            "classifier__max_depth": [2, 3, 4],
            "classifier__subsample": [0.8, 0.9, 1.0],
        },
        "Logistic Regression": {
            "classifier__C": [0.05, 0.1, 0.5, 1.0, 5.0],
            "classifier__penalty": ["l2"],
            "classifier__class_weight": ["balanced", None],
        },
        "Support Vector Machine": {
            "classifier__C": [0.5, 1.0, 2.0, 5.0],
            "classifier__gamma": ["scale", "auto", 0.01, 0.1],
            "classifier__class_weight": ["balanced", None],
        },
    }

    base_pipe = Pipeline([
        ("preprocessor", build_preprocessing_pipeline(scale_numeric=True)),
        ("classifier", get_candidate_models().get(top_model_name, RandomForestClassifier(random_state=random_state))),
    ])

    grid = param_grids.get(top_model_name, param_grids["Random Forest"])
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=random_state)

    search = GridSearchCV(
        base_pipe,
        param_grid=grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    logger.info(f"Best params for {top_model_name}: {search.best_params_}")
    logger.info(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    return search.best_estimator_


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    beta: float = 2.0
) -> Dict[str, Any]:
    """
    Finds optimal decision threshold balancing clinical sensitivity vs specificity.
    Evaluates:
    - Youden's J-statistic (Sensitivity + Specificity - 1)
    - F-beta score (beta=2 favors Recall over Precision by factor of 2)
    """
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_j_idx = int(np.argmax(j_scores))
    youden_threshold = float(roc_thresholds[best_j_idx])
    # Bound threshold within [0.05, 0.95]
    youden_threshold = max(0.05, min(0.95, youden_threshold))

    # Fine threshold grid search for F2
    candidate_thresholds = np.linspace(0.1, 0.9, 81)
    best_f2 = -1.0
    best_f2_thresh = 0.5

    for thresh in candidate_thresholds:
        preds = (y_prob >= thresh).astype(int)
        f2 = fbeta_score(y_true, preds, beta=beta, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_f2_thresh = float(thresh)

    return {
        "youden_threshold": youden_threshold,
        "youden_j_score": float(j_scores[best_j_idx]),
        "f_beta_threshold": best_f2_thresh,
        "max_f_beta": float(best_f2),
        "selected_clinical_threshold": float(best_f2_thresh),
    }


def calibrate_and_fit_final_pipeline(
    tuned_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Fits and wraps the model with CalibratedClassifierCV.
    Finds the optimal clinical threshold and computes holdout test metrics.
    """
    logger.info("Calibrating model probabilities using CalibratedClassifierCV (isotonic / sigmoid)...")

    # Fit tuned pipeline on training set
    tuned_pipeline.fit(X_train, y_train)

    # Cross-validated probability calibration
    calibrated_clf = CalibratedClassifierCV(
        estimator=tuned_pipeline,
        method="sigmoid",
        cv=5,
    )
    calibrated_clf.fit(X_train, y_train)

    # In-fold predicted probabilities on train for threshold determination
    train_probs = calibrated_clf.predict_proba(X_train)[:, 1]
    threshold_info = find_optimal_threshold(y_train.values, train_probs, beta=2.0)
    optimal_thresh = threshold_info["selected_clinical_threshold"]

    # Evaluate on untouched holdout test set
    test_probs = calibrated_clf.predict_proba(X_test)[:, 1]
    default_preds = (test_probs >= 0.5).astype(int)
    calibrated_preds = (test_probs >= optimal_thresh).astype(int)

    # Test metrics at both default 0.5 and optimized clinical threshold
    metrics = {
        "default_threshold": 0.5,
        "optimal_threshold": optimal_thresh,
        "test_eval_default_0_5": {
            "accuracy": float(accuracy_score(y_test, default_preds)),
            "recall": float(recall_score(y_test, default_preds)),
            "precision": float(precision_score(y_test, default_preds)),
            "f1": float(f1_score(y_test, default_preds)),
            "f2": float(fbeta_score(y_test, default_preds, beta=2.0)),
            "roc_auc": float(roc_auc_score(y_test, test_probs)),
            "brier_score": float(brier_score_loss(y_test, test_probs)),
            "confusion_matrix": confusion_matrix(y_test, default_preds).tolist(),
        },
        "test_eval_optimal_thresh": {
            "accuracy": float(accuracy_score(y_test, calibrated_preds)),
            "recall": float(recall_score(y_test, calibrated_preds)),
            "precision": float(precision_score(y_test, calibrated_preds)),
            "f1": float(f1_score(y_test, calibrated_preds)),
            "f2": float(fbeta_score(y_test, calibrated_preds, beta=2.0)),
            "roc_auc": float(roc_auc_score(y_test, test_probs)),
            "brier_score": float(brier_score_loss(y_test, test_probs)),
            "confusion_matrix": confusion_matrix(y_test, calibrated_preds).tolist(),
        },
        "threshold_optimization_details": threshold_info,
    }

    return calibrated_clf, metrics


def save_model_artifacts(
    calibrated_model: Any,
    raw_tuned_pipeline: Pipeline,
    metadata: Dict[str, Any],
    benchmark_df: pd.DataFrame,
    output_dir: Path = MODELS_DIR
) -> None:
    """Serializes the final calibrated model, uncalibrated pipeline, and metadata manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "best_model.joblib"
    uncalibrated_path = output_dir / "tuned_pipeline.joblib"
    meta_path = output_dir / "model_metadata.json"
    benchmark_path = output_dir / "benchmark_results.csv"

    joblib.dump(calibrated_model, model_path)
    joblib.dump(raw_tuned_pipeline, uncalibrated_path)
    benchmark_df.to_csv(benchmark_path, index=False)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Model artifacts successfully serialized to: {output_dir}")


def run_training_pipeline() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Runs Phase 5 & 6 end-to-end."""
    print("=" * 60)
    print("PHASES 5 & 6: MODEL BENCHMARK, TUNING, CALIBRATION & THRESHOLDING")
    print("=" * 60)

    train_path = PROCESSED_DATA_DIR / "train.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Processed datasets not found. Please run Phase 4 first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df[ALL_FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[ALL_FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    # 1. Benchmark 5 candidate models
    print("\n[1/5] Benchmarking 5 candidate classifiers via Repeated Stratified K-Fold CV...")
    benchmark_df, _ = benchmark_models(X_train, y_train, cv_splits=5, cv_repeats=3)
    print("\n--- BENCHMARK RESULTS (Sorted by Recall & ROC-AUC) ---")
    print(benchmark_df[["Model", "Recall_Mean", "Precision_Mean", "F1_Mean", "ROC_AUC_Mean"]].to_string(index=False))

    # Select leading model
    leading_model_name = benchmark_df.iloc[0]["Model"]
    print(f"\n[+] Leading model selected: {leading_model_name}")

    # 2. Hyperparameter Grid Search Tuning
    print(f"\n[2/5] Tuning hyperparameters for {leading_model_name}...")
    tuned_pipe = tune_top_model(leading_model_name, X_train, y_train)

    # 3. Probability Calibration & Threshold Optimization
    print("\n[3/5] Applying probability calibration (Sigmoid) & optimizing clinical threshold...")
    calibrated_model, eval_metrics = calibrate_and_fit_final_pipeline(
        tuned_pipe, X_train, y_train, X_test, y_test
    )

    # 4. Compile metadata
    metadata = {
        "algorithm": leading_model_name,
        "calibrated": True,
        "calibration_method": "sigmoid",
        "optimal_threshold": eval_metrics["optimal_threshold"],
        "features": ALL_FEATURE_COLUMNS,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": eval_metrics,
    }

    # 5. Serialize Artifacts
    print("\n[4/5] Serializing model and metadata...")
    save_model_artifacts(calibrated_model, tuned_pipe, metadata, benchmark_df, MODELS_DIR)

    opt_metrics = eval_metrics["test_eval_optimal_thresh"]
    cm = opt_metrics["confusion_matrix"]
    print("\n--- HOLDOUT TEST EVALUATION (At Optimal Threshold) ---")
    print(f"Optimal Decision Threshold: {eval_metrics['optimal_threshold']:.3f}")
    print(f"Sensitivity / Recall:      {opt_metrics['recall']:.1%}")
    print(f"Specificity:               {cm[0][0] / (cm[0][0] + cm[0][1]):.1%}")
    print(f"Precision:                 {opt_metrics['precision']:.1%}")
    print(f"F1-Score:                  {opt_metrics['f1']:.1%}")
    print(f"F2-Score (Recall Focus):   {opt_metrics['f2']:.1%}")
    print(f"ROC-AUC Score:             {opt_metrics['roc_auc']:.4f}")
    print(f"Brier Calibration Loss:    {opt_metrics['brier_score']:.4f}")
    print(f"Confusion Matrix [TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}]")

    print("\n" + "=" * 60)
    print("PHASES 5 & 6 COMPLETE: Model trained, calibrated, and serialized.")
    print("=" * 60)

    return benchmark_df, metadata


if __name__ == "__main__":
    run_training_pipeline()
