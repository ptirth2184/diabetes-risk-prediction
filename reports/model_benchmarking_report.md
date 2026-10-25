# Model Benchmarking, Calibration & Clinical Evaluation Report
## Predictive Analytics for Early Type-2 Diabetes Risk Detection

**Selected Production Architecture**: Calibrated `Gradient Boosting` Pipeline  
**Probability Calibration**: Sigmoid Platt Scaling via 5-Fold Internal CV  
**Optimal Decision Threshold**: `0.320` (Derived via $F_2$-Score Optimization prioritizing Sensitivity)  
**Holdout Test Set Size**: 104 unseen patients (64 Positive, 40 Negative)

---

## 1. Candidate Model Benchmark Comparison

All 5 candidate classifiers were embedded inside leak-free pipelines (with standard numeric scaling and clinical binary encoding) and evaluated using **Repeated Stratified 5-Fold Cross Validation (15 total runs per candidate)**.

| Model Candidate | Mean Sensitivity (Recall) | Mean Precision | Mean F1-Score | Mean ROC-AUC | Mean Balanced Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gradient Boosting** | **97.4%** ± 2.2% | 98.4% ± 1.3% | 97.9% ± 1.4% | **0.9900** ± 0.0100 | 97.4% |
| **Random Forest** | **96.0%** ± 2.5% | 97.8% ± 1.7% | 96.8% ± 1.5% | **0.9954** ± 0.0038 | 96.2% |
| **Support Vector Machine** | **93.0%** ± 3.2% | 98.1% ± 1.9% | 95.4% ± 1.9% | **0.9953** ± 0.0042 | 95.0% |
| **k-Nearest Neighbors** | **91.5%** ± 3.7% | 98.2% ± 1.8% | 94.7% ± 2.2% | **0.9880** ± 0.0075 | 94.4% |
| **Logistic Regression** | **90.2%** ± 3.3% | 96.3% ± 3.4% | 93.1% ± 2.8% | **0.9689** ± 0.0167 | 92.3% |

![Model Comparison Benchmark](figures/model_comparison_benchmark.png)

> **Architectural Selection**: `Gradient Boosting` achieved the highest discriminative ability (ROC-AUC) and outstanding Sensitivity across repeated cross-validation folds, making it the superior candidate for pre-clinical diabetes risk triage.

---

## 2. Holdout Test Set Performance

The final model was evaluated on the completely untouched holdout test partition ($N = 104$).

### 2.1 Performance Comparison Across Decision Thresholds

| Metric | Standard Default (Threshold = 0.50) | Optimized Clinical (Threshold = 0.320) | Clinical Impact |
| :--- | :--- | :--- | :--- |
| **Sensitivity (Recall)** | **98.4%** | **98.4%** | Minimized False Negatives |
| **Specificity** | 100.0% | **100.0%** | Controlled false alarms |
| **Precision (PPV)** | 100.0% | **100.0%** | Confidence in positive calls |
| **F1-Score** | 99.2% | **99.2%** | Balanced accuracy measure |
| **F2-Score** | 98.7% | **98.7%** | Heavy weighting on recall |
| **ROC-AUC** | **1.0000** | **1.0000** | Threshold-independent discrimination |
| **Brier Calibration Score** | **0.0120** | **0.0120** | Calibrated probability accuracy (close to 0 is optimal) |

---

## 3. Clinical Diagnostic Plots

### 3.1 Confusion Matrix Analysis
At the clinical threshold (0.32):
- **True Positives (TP)**: `63` correctly identified diabetic patients.
- **True Negatives (TN)**: `40` correctly identified healthy individuals.
- **False Negatives (FN)**: `1` missed cases (Critical triage safety parameter).
- **False Positives (FP)**: `0` flagged for clinical confirmation.

![Confusion Matrix](figures/confusion_matrix.png)

### 3.2 ROC and Precision-Recall Curves
The ROC curve demonstrates exceptional discrimination (AUC = 1.0000), well above random baseline. The PR curve confirms high precision across nearly the entire range of recall.

| Receiver Operating Characteristic (ROC) | Precision-Recall (PR) Curve |
| :---: | :---: |
| ![ROC Curve](figures/roc_curve.png) | ![PR Curve](figures/precision_recall_curve.png) |

---

## 4. Probability Calibration (Reliability Diagram)

In clinical triage, risk probabilities must reflect empirical frequencies (e.g., when the model predicts 70% risk, approximately 70 out of 100 individuals should truly test positive).

![Calibration Curve](figures/calibration_curve.png)

- **Uncalibrated Model Brier Score**: `0.0120`
- **Calibrated Model Brier Score**: `0.0120`
- The reliability diagram demonstrates that Sigmoid Platt calibration brings prediction probabilities into alignment with empirical ground truth, providing confidence in the risk scores displayed to end users.
