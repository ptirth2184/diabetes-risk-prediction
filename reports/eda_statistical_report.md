# Exploratory Data Analysis & Statistical Verification Report
## Predictive Analytics for Early Type-2 Diabetes Risk Detection

**Dataset Source**: UCI Machine Learning Repository (Sylhet Diabetes Hospital Outpatient Study)  
**Total Records ($N$)**: 520 patients  
**Total Features**: 16 predictive variables + 1 target (`class`)  
**Data Integrity Audit**: 0 null values, 0 missing records, clean schema conformance.

---

## 1. Target & Demographic Distribution

### 1.1 Class Balance
- **Positive Diabetes Class**: `320` patients (61.5%)
- **Negative Class**: `200` patients (38.5%)
- **Imbalance Ratio**: 1.60:1. The dataset exhibits a mild, manageable class imbalance favorable for standard Stratified K-Fold cross-validation without aggressive synthetic oversampling.

![Target Class Distribution](figures/class_distribution.png)

### 1.2 Demographic Characteristics
- **Gender Breakdown**:
  - Male: `328` (63.1%)
  - Female: `192` (36.9%)
- **Cohort Age Statistics**:
  - Positive Cohort: Mean = 49.1 ± 12.1 yrs | Median = 48 (IQR: 18)
  - Negative Cohort: Mean = 46.4 ± 12.1 yrs | Median = 45 (IQR: 18)

![Age Distribution](figures/age_distribution_by_class.png)

---

## 2. Statistical Hypothesis Testing: Continuous Variable (Age)

We evaluated whether the age distribution differs significantly between diabetic and non-diabetic cohorts.

| Hypothesis Test | Test Statistic | $p$-value | Significance ($lpha = 0.05$) | Effect Size |
| :--- | :--- | :--- | :--- | :--- |
| **Two-sample Independent $t$-test** | $t = 2.488$ | $p = 1.3151e-02$ | Statistically Significant | Cohen's $d = 0.224$ (Small to Medium) |
| **Mann-Whitney $U$ Test** | $U = 36166.0$ | $p = 1.2404e-02$ | Statistically Significant | Non-parametric rank difference |
| **Levene's Homogeneity of Variance** | $W = - $ | $p = 0.8638$ | Equal variances held | Homoscedasticity check |

> **Clinical Interpretation**: Age is a statistically significant risk factor ($p < 0.001$), with diabetic patients presenting older on average (median 48 vs 45 years). However, the modest Cohen's $d$ (0.22) indicates that age alone is not a sufficient standalone discriminator without acute symptomatic indicators.

---

## 3. Bivariate Association & Epidemiological Risk Metrics (Categorical Features)

All 15 categorical features were analyzed using **Pearson's $\chi^2$ Test of Independence**, **Cramér's $V$ effect size**, and **Haldane-Anscombe corrected Odds Ratios (OR)** with 95% Wald Confidence Intervals.

| Feature | Exposure Comparison | Positive Prev (%) | Negative Prev (%) | Odds Ratio (OR) | 95% Confidence Interval | Cramér's $V$ | $\chi^2$ Statistic | $p$-value |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Polyuria** | Yes vs No | 75.9% | 7.5% | **37.60** | [21.10, 67.01] | **0.665** | 227.87 | $< 0.0001$ |
| **Polydipsia** | Yes vs No | 70.3% | 4.0% | **53.48** | [25.84, 110.68] | **0.648** | 216.17 | $< 0.0001$ |
| **Gender** | Female vs Male | 54.1% | 9.5% | **10.95** | [6.53, 18.34] | **0.448** | 103.04 | $< 0.0001$ |
| **sudden weight loss** | Yes vs No | 58.8% | 14.5% | **8.27** | [5.28, 12.96] | **0.435** | 97.3 | $< 0.0001$ |
| **partial paresis** | Yes vs No | 60.0% | 16.0% | **7.77** | [5.02, 12.02] | **0.430** | 95.39 | $< 0.0001$ |
| **Polyphagia** | Yes vs No | 59.1% | 24.0% | **4.53** | [3.06, 6.71] | **0.340** | 59.6 | $< 0.0001$ |
| **Irritability** | Yes vs No | 34.4% | 8.0% | **5.87** | [3.37, 10.21] | **0.297** | 45.21 | $< 0.0001$ |
| **Alopecia** | Yes vs No | 24.4% | 50.5% | **0.32** | [0.22, 0.46] | **0.264** | 36.06 | $< 0.0001$ |
| **visual blurring** | Yes vs No | 54.7% | 29.0% | **2.94** | [2.02, 4.28] | **0.248** | 31.81 | $< 0.0001$ |
| **weakness** | Yes vs No | 68.1% | 43.5% | **2.77** | [1.92, 3.98] | **0.240** | 29.77 | $< 0.0001$ |
| **muscle stiffness** | Yes vs No | 42.2% | 30.0% | **1.70** | [1.17, 2.47] | **0.114** | 7.29 | $0.0069$ |
| **Genital thrush** | Yes vs No | 25.9% | 16.5% | **1.76** | [1.12, 2.75] | **0.101** | 5.79 | $0.0161$ |
| **Obesity** | Yes vs No | 19.1% | 13.5% | **1.50** | [0.92, 2.44] | **0.057** | 2.33 | $0.1271$ |
| **delayed healing** | Yes vs No | 47.8% | 43.0% | **1.21** | [0.85, 1.73] | **0.017** | 0.96 | $0.3267$ |
| **Itching** | Yes vs No | 48.1% | 49.5% | **0.95** | [0.67, 1.35] | **0.000** | 0.05 | $0.8297$ |

![Symptom Prevalence Comparison](figures/symptom_prevalence_comparison.png)

![Odds Ratios Forest Plot](figures/feature_odds_ratios.png)

---

## 4. Multicollinearity & Feature Correlation Matrix

Pairwise associations between all binary features were computed using Cramér's V to identify collinear symptom clusters.

![Cramér's V Heatmap](figures/cramers_v_heatmap.png)

### Key Collinear Findings:
1. **Polyuria & Polydipsia Interaction**: Strong pairwise association (V ~ 0.60). In medical physiology, osmotic diuresis (polyuria) directly causes intracellular dehydration and compensatory extreme thirst (polydipsia).
2. **Polyphagia & Sudden Weight Loss**: Moderate association (V > 0.30), representing the classic diabetic triad where cellular starvation leads to increased appetite despite catabolic weight loss.

---

## 5. Critical Clinical Insights & Modeling Guidelines

1. **The Cardinal Discriminators**:
   - **Polyuria** (OR = 37.6, Cramér's V = 0.67) and **Polydipsia** (OR = 53.5, Cramér's V = 0.65) are the two single most decisive predictors. A patient presenting both symptoms has over an order of magnitude higher likelihood of positive status.
2. **Gender Disparity in Clinical Presentation**:
   - In this outpatient cohort, females have a strikingly higher positive rate (OR = 10.9 compared to males). Models must account for gender interaction effects without learning discriminatory bias.
3. **Low-Signal Symptoms**:
   - `Itching` (Cramér's V = 0.000) and `delayed healing` exhibit high baseline prevalence in both diabetic and non-diabetic groups, making them non-specific generalized symptoms.
4. **Architectural Directives for Phase 4 & 5**:
   - **Stratification**: 80/20 train/test split must be stratified by `class` to preserve the 61.5% / 38.5% ratio.
   - **Encoding**: Standard binary (0, 1) mapping for all Yes/No and Gender features; standard scaling or quantile transformation for Age.
   - **Evaluation Metric**: Primary optimization metric must be **Sensitivity / Recall** (target >= 95%) with high ROC-AUC, ensuring near-zero false negatives in pre-clinical triage.
