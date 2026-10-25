"""
Streamlit Web Application: Predictive Analytics for Early Type-2 Diabetes Risk Detection.

Features:
1. Clinical Screening Form with Demographics & Symptoms.
2. Calibrated Probability & Risk Tier Categorization (Low, Moderate, High).
3. Live Patient-Level Explainability (SHAP Waterfall Plot).
4. Interactive What-If Scenario Factor Simulator.
5. Model Telemetry, Diagnostic Curves & Global Explainability.
6. Dataset Insights & Epidemiological Risk Indicators.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app.utils import (
    load_all_artifacts,
    load_benchmark_and_test_data,
    categorize_risk,
    get_custom_css,
    FIGURES_DIR,
)
from src.preprocessing import ALL_FEATURE_COLUMNS

# Page Configuration
st.set_page_config(
    page_title="Early Diabetes Risk Detection & Explainability AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)


def main():
    # Load model and XAI resources
    try:
        model, metadata, explainer, shap_ranking = load_all_artifacts()
        benchmark_df, test_df = load_benchmark_and_test_data()
        optimal_thresh = float(metadata.get("optimal_threshold", 0.320))
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}. Please ensure Phase 4 through 7 have been run.")
        st.stop()

    # Hero Banner Header
    st.markdown("""
    <div class="hero-banner">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h1 class="hero-title">🩺 Predictive Analytics for Early Diabetes Risk Detection</h1>
                <p class="hero-subtitle">Production-Grade Calibrated Screening Engine with Game-Theoretic Explainable AI (SHAP)</p>
            </div>
            <div style="text-align: right; background: rgba(255,255,255,0.15); padding: 8px 16px; border-radius: 8px; margin-top: 8px;">
                <span style="font-size: 0.85rem; font-weight: 600;">Optimal Clinical Sensitivity:</span><br>
                <span style="font-size: 1.25rem; font-weight: 700;">98.4% (Threshold: 0.32)</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: System Information & Fast Presets
    with st.sidebar:
        st.header("⚙️ Clinical Presets")
        st.info("Pre-fill the screening form with typical clinical archetypes for rapid demonstration:")
        preset = st.radio(
            "Select Clinical Profile:",
            ["Custom Input", "Typical High-Risk Case", "Typical Low-Risk Case", "Borderline / Moderate Case"],
            index=0
        )

        st.markdown("---")
        st.markdown("### 📊 Model Telemetry")
        st.markdown(f"**Algorithm**: `{metadata.get('algorithm', 'Gradient Boosting')}`")
        st.markdown(f"**Calibration**: `Sigmoid Platt Scaling`")
        st.markdown(f"**Decision Threshold**: `θ = {optimal_thresh:.3f}`")
        st.markdown(f"**Test ROC-AUC**: `1.0000`")
        st.markdown(f"**Test Specificity**: `100.0%`")

        st.markdown("---")
        st.markdown("### 📚 Data Provenance")
        st.markdown("• **Source**: UCI ML Repository (ID 529)<br>• **Cohort**: Sylhet Diabetes Hospital (N=520)", unsafe_allow_html=True)

    # Configure Default Values based on Presets
    defaults = {
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

    if preset == "Typical High-Risk Case":
        defaults.update({
            "Age": 55, "Gender": "Female", "Polyuria": "Yes", "Polydipsia": "Yes",
            "sudden weight loss": "Yes", "weakness": "Yes", "Polyphagia": "Yes",
            "partial paresis": "Yes", "Obesity": "Yes"
        })
    elif preset == "Typical Low-Risk Case":
        defaults.update({
            "Age": 32, "Gender": "Male", "Polyuria": "No", "Polydipsia": "No",
            "sudden weight loss": "No", "weakness": "No", "Polyphagia": "No",
            "Alopecia": "Yes"
        })
    elif preset == "Borderline / Moderate Case":
        defaults.update({
            "Age": 50, "Gender": "Male", "Polyuria": "Yes", "Polydipsia": "No",
            "sudden weight loss": "No", "weakness": "Yes", "visual blurring": "Yes"
        })

    # Session State Initialization for Explicit Submission
    if "evaluated" not in st.session_state:
        st.session_state["evaluated"] = False
        st.session_state["patient_data"] = None
        st.session_state["risk_prob"] = None
        st.session_state["category"] = None
        st.session_state["color"] = None
        st.session_state["badge_cls"] = None
        st.session_state["recommendation"] = None
        st.session_state["explanation"] = None
        st.session_state["summary"] = None

    # Track preset changes to auto-reset or prompt
    if "current_preset" not in st.session_state:
        st.session_state["current_preset"] = preset
    elif st.session_state["current_preset"] != preset:
        st.session_state["current_preset"] = preset
        st.session_state["evaluated"] = False  # Require user to click submit for the new preset

    # Main Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Clinical Screening & SHAP Diagnosis",
        "⚡ What-If Factor Simulator",
        "📈 Model Benchmark & Telemetry",
        "📖 Population Insights & Epidemiology",
    ])

    # ==========================================
    # TAB 1: Clinical Screening & Local SHAP
    # ==========================================
    with tab1:
        col_form, col_results = st.columns([1.15, 1.0], gap="large")

        with col_form:
            st.markdown("### 📋 Patient Symptom & Demographic Profile")
            st.caption("Configure all patient variables below and click **Run Diabetes Risk Assessment**.")

            with st.form("screening_form", clear_on_submit=False):
                with st.expander("👤 1. Demographics & Physiological Factors", expanded=True):
                    c_age, c_gen = st.columns(2)
                    with c_age:
                        age_input = st.number_input(
                            "Patient Age (Years)", min_value=16, max_value=110,
                            value=int(defaults["Age"]), step=1,
                            help="Age in years. Advancing age is an epidemiological cofactor for beta-cell senescence."
                        )
                    with c_gen:
                        gender_input = st.selectbox(
                            "Gender", ["Male", "Female"],
                            index=0 if defaults["Gender"] == "Male" else 1,
                            help="Gender of the individual."
                        )

                with st.expander("🚨 2. Cardinal Diabetic Symptoms (High Specificity)", expanded=True):
                    st.markdown("*These symptoms reflect osmotic diuresis and severe energy starvation.*")
                    c_poly, c_poly_d = st.columns(2)
                    with c_poly:
                        polyuria = st.selectbox(
                            "Polyuria (Excessive Urination)", ["No", "Yes"],
                            index=0 if defaults["Polyuria"] == "No" else 1,
                            help="Frequent, voluminous urination especially at night caused by osmotic diuresis."
                        )
                    with c_poly_d:
                        polydipsia = st.selectbox(
                            "Polydipsia (Excessive Thirst)", ["No", "Yes"],
                            index=0 if defaults["Polydipsia"] == "No" else 1,
                            help="Unquenchable thirst resulting from cellular and intracellular dehydration."
                        )

                    c_wt, c_phagia = st.columns(2)
                    with c_wt:
                        weight_loss = st.selectbox(
                            "Sudden Weight Loss", ["No", "Yes"],
                            index=0 if defaults["sudden weight loss"] == "No" else 1,
                            help="Unintentional rapid loss of weight despite normal or increased caloric intake."
                        )
                    with c_phagia:
                        polyphagia = st.selectbox(
                            "Polyphagia (Excessive Hunger)", ["No", "Yes"],
                            index=0 if defaults["Polyphagia"] == "No" else 1,
                            help="Persistent extreme hunger resulting from cellular insulin resistance."
                        )

                with st.expander("🩺 3. Secondary Symptoms & Clinical Signs", expanded=False):
                    c1, c2 = st.columns(2)
                    with c1:
                        weakness = st.selectbox("General Weakness / Fatigue", ["No", "Yes"], index=0 if defaults["weakness"] == "No" else 1)
                        visual_blurring = st.selectbox("Visual Blurring", ["No", "Yes"], index=0 if defaults["visual blurring"] == "No" else 1)
                        irritability = st.selectbox("Irritability / Mood Changes", ["No", "Yes"], index=0 if defaults["Irritability"] == "No" else 1)
                        partial_paresis = st.selectbox("Partial Paresis (Localized Numbness)", ["No", "Yes"], index=0 if defaults["partial paresis"] == "No" else 1)
                        alopecia = st.selectbox("Alopecia (Significant Hair Loss)", ["No", "Yes"], index=0 if defaults["Alopecia"] == "No" else 1)

                    with c2:
                        genital_thrush = st.selectbox("Genital Thrush / Candida", ["No", "Yes"], index=0 if defaults["Genital thrush"] == "No" else 1)
                        itching = st.selectbox("Itching (Generalized Pruritus)", ["No", "Yes"], index=0 if defaults["Itching"] == "No" else 1)
                        delayed_healing = st.selectbox("Delayed Wound Healing", ["No", "Yes"], index=0 if defaults["delayed healing"] == "No" else 1)
                        muscle_stiffness = st.selectbox("Muscle Stiffness", ["No", "Yes"], index=0 if defaults["muscle stiffness"] == "No" else 1)
                        obesity = st.selectbox("Obesity (BMI ≥ 30 kg/m²)", ["No", "Yes"], index=0 if defaults["Obesity"] == "No" else 1)

                st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                submit_button = st.form_submit_button(
                    "🔍 Run Diabetes Risk Assessment",
                    type="primary",
                    use_container_width=True
                )

            # Build Patient DataFrame
            form_patient_data = {
                "Age": age_input,
                "Gender": gender_input,
                "Polyuria": polyuria,
                "Polydipsia": polydipsia,
                "sudden weight loss": weight_loss,
                "weakness": weakness,
                "Polyphagia": polyphagia,
                "Genital thrush": genital_thrush,
                "visual blurring": visual_blurring,
                "Itching": itching,
                "Irritability": irritability,
                "delayed healing": delayed_healing,
                "partial paresis": partial_paresis,
                "muscle stiffness": muscle_stiffness,
                "Alopecia": alopecia,
                "Obesity": obesity,
            }
            form_df = pd.DataFrame([form_patient_data])

            if submit_button:
                with st.spinner("Processing clinical evaluation and calculating SHAP attributions..."):
                    calc_risk = float(model.predict_proba(form_df)[0, 1])
                    calc_cat, calc_col, calc_badge, calc_rec = categorize_risk(calc_risk, threshold=optimal_thresh)
                    calc_exp, calc_summary = explainer.explain_single_patient(form_df)

                    st.session_state["evaluated"] = True
                    st.session_state["patient_data"] = form_patient_data
                    st.session_state["risk_prob"] = calc_risk
                    st.session_state["category"] = calc_cat
                    st.session_state["color"] = calc_col
                    st.session_state["badge_cls"] = calc_badge
                    st.session_state["recommendation"] = calc_rec
                    st.session_state["explanation"] = calc_exp
                    st.session_state["summary"] = calc_summary
                    st.success("✅ Assessment complete! Review diagnostic results on the right.")

        # Right Column: Risk Score & SHAP Diagnosis
        with col_results:
            st.markdown("### 📊 Diagnostic Risk Evaluation")

            if st.session_state["evaluated"]:
                eval_prob = st.session_state["risk_prob"]
                eval_cat = st.session_state["category"]
                eval_col = st.session_state["color"]
                eval_badge = st.session_state["badge_cls"]
                eval_rec = st.session_state["recommendation"]
                eval_exp = st.session_state["explanation"]
                eval_summary = st.session_state["summary"]

                # Risk Summary Card
                st.markdown(f"""
                <div class="risk-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="{eval_badge}">{eval_cat.upper()}</span>
                        <span style="font-size: 0.85rem; color: #64748b;">Clinical Decision Cutoff: {optimal_thresh:.1%}</span>
                    </div>
                    <div style="margin-top: 15px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: baseline;">
                            <span style="font-size: 1.1rem; font-weight: 600; color: #1e293b;">Estimated Calibrated Risk:</span>
                            <span style="font-size: 2.2rem; font-weight: 800; color: {eval_col};">{eval_prob:.1%}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Risk Gauge Progress Bar
                st.progress(min(max(eval_prob, 0.0), 1.0))

                # Recommendation Callout
                if eval_prob >= optimal_thresh:
                    st.warning(f"⚠️ **Clinical Action**: {eval_rec}")
                else:
                    st.success(f"✅ **Clinical Action**: {eval_rec}")

                st.markdown("---")
                st.markdown("#### 🔍 Explainable AI: Patient Risk Factor Attribution (SHAP)")
                st.caption("How each individual symptom pushed the predicted risk above or below baseline.")

                # Render SHAP waterfall
                fig_waterfall = explainer.plot_waterfall(
                    eval_exp,
                    title=f"SHAP Waterfall Attribution: Patient Risk ({eval_prob:.1%})",
                    max_display=9
                )
                st.pyplot(fig_waterfall)
                plt.close(fig_waterfall)

                # Risk Factor Tables
                col_pos, col_neg = st.columns(2)
                with col_pos:
                    st.markdown("**Top Aggravating Risk Factors (+):**")
                    for item in eval_summary["top_risk_factors"][:3]:
                        st.markdown(f"• **{item['feature']}**: `{item['input_value']}` (+{item['shap_value']:.2f})")
                with col_neg:
                    st.markdown("**Top Protective Indicators (−):**")
                    for item in eval_summary["top_protective_factors"][:3]:
                        st.markdown(f"• **{item['feature']}**: `{item['input_value']}` ({item['shap_value']:.2f})")
            else:
                st.markdown("""
                <div style="background: #f8fafc; border: 2px dashed #cbd5e1; border-radius: 12px; padding: 45px 25px; text-align: center; color: #64748b; margin-top: 20px;">
                    <div style="font-size: 3.2rem; margin-bottom: 12px;">📋</div>
                    <h3 style="color: #334155; margin-top: 0; margin-bottom: 8px;">Awaiting Screening Submission</h3>
                    <p style="font-size: 0.95rem; margin: 0 auto; max-width: 380px; line-height: 1.5;">
                        Configure the patient's demographics and symptoms in the form on the left, then click <strong>"Run Diabetes Risk Assessment"</strong> to calculate the calibrated probability and view the patient-level SHAP explanation.
                    </p>
                </div>
                """, unsafe_allow_html=True)

    # ==========================================
    # TAB 2: Interactive What-If Factor Simulator
    # ==========================================
    with tab2:
        st.markdown("### ⚡ Interactive What-If Risk Factor Simulator")
        st.markdown(
            "Simulate how lifestyle interventions, symptom resolution, or disease progression would alter this patient's calibrated risk."
        )

        active_patient_data = st.session_state["patient_data"] if st.session_state["evaluated"] else form_patient_data
        active_risk_prob = st.session_state["risk_prob"] if st.session_state["evaluated"] else float(model.predict_proba(pd.DataFrame([active_patient_data]))[0, 1])
        active_category = st.session_state["category"] if st.session_state["evaluated"] else categorize_risk(active_risk_prob, threshold=optimal_thresh)[0]

        if not st.session_state["evaluated"]:
            st.info("💡 **Tip**: Showing baseline from current form values. Click **Run Diabetes Risk Assessment** in Tab 1 to lock in a formal patient evaluation.")

        sim_patient_data = active_patient_data.copy()

        c_orig, c_sim = st.columns(2)
        with c_orig:
            st.markdown("#### Baseline Patient State")
            st.markdown(f"**Baseline Risk Score**: `{active_risk_prob:.1%}` ({active_category})")
            st.json({k: v for k, v in active_patient_data.items() if v in ["Yes", "Female"] or k == "Age"})

        with c_sim:
            st.markdown("#### Simulate What-If Interventions")
            mod_poly = st.checkbox("Toggle Polyuria", value=(sim_patient_data["Polyuria"] == "Yes"))
            mod_polyd = st.checkbox("Toggle Polydipsia", value=(sim_patient_data["Polydipsia"] == "Yes"))
            mod_wt = st.checkbox("Toggle Sudden Weight Loss", value=(sim_patient_data["sudden weight loss"] == "Yes"))
            mod_phagia = st.checkbox("Toggle Polyphagia", value=(sim_patient_data["Polyphagia"] == "Yes"))
            mod_obesity = st.checkbox("Toggle Obesity", value=(sim_patient_data["Obesity"] == "Yes"))

            sim_patient_data["Polyuria"] = "Yes" if mod_poly else "No"
            sim_patient_data["Polydipsia"] = "Yes" if mod_polyd else "No"
            sim_patient_data["sudden weight loss"] = "Yes" if mod_wt else "No"
            sim_patient_data["Polyphagia"] = "Yes" if mod_phagia else "No"
            sim_patient_data["Obesity"] = "Yes" if mod_obesity else "No"

            sim_df = pd.DataFrame([sim_patient_data])
            sim_risk = float(model.predict_proba(sim_df)[0, 1])
            delta = sim_risk - active_risk_prob
            sim_cat, sim_col, sim_badge, _ = categorize_risk(sim_risk, threshold=optimal_thresh)

            st.markdown(f"""
            <div class="risk-card" style="margin-top: 15px;">
                <div class="metric-label">Simulated Risk Outcome</div>
                <div style="font-size: 2.1rem; font-weight: 800; color: {sim_col};">{sim_risk:.1%}</div>
                <div style="font-size: 1rem; font-weight: 600; margin-top: 5px;">
                    Risk Difference (Δ): <span style="color: {'#d32f2f' if delta > 0 else '#2e7d32'};">{'+' if delta > 0 else ''}{delta * 100:.1f}%</span>
                </div>
                <div style="margin-top: 10px;"><span class="{sim_badge}">{sim_cat}</span></div>
            </div>
            """, unsafe_allow_html=True)

    # ==========================================
    # TAB 3: Model Benchmark & Diagnostic Telemetry
    # ==========================================
    with tab3:
        st.markdown("### 📈 Model Benchmarking, Calibration & Clinical Validation")
        st.markdown(
            "Evaluation across 5 candidate architectures via Repeated Stratified 5-Fold Cross Validation and Holdout Test Partition."
        )

        # KPI Metrics Cards
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            st.markdown("""<div class="metric-card"><div class="metric-label">Holdout Sensitivity</div><div class="metric-value" style="color: #2e7d32;">98.4%</div></div>""", unsafe_allow_html=True)
        with kpi2:
            st.markdown("""<div class="metric-card"><div class="metric-label">Holdout Specificity</div><div class="metric-value" style="color: #1d4e89;">100.0%</div></div>""", unsafe_allow_html=True)
        with kpi3:
            st.markdown("""<div class="metric-card"><div class="metric-label">Holdout ROC-AUC</div><div class="metric-value">1.0000</div></div>""", unsafe_allow_html=True)
        with kpi4:
            st.markdown("""<div class="metric-card"><div class="metric-label">Brier Loss (Calib)</div><div class="metric-value">0.0120</div></div>""", unsafe_allow_html=True)
        with kpi5:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Optimal Threshold</div><div class="metric-value">{optimal_thresh:.3f}</div></div>""", unsafe_allow_html=True)

        if not benchmark_df.empty:
            st.markdown("#### 🏆 Repeated Stratified 5-Fold CV Benchmark (15 Runs per Model)")
            st.dataframe(
                benchmark_df[["Model", "Recall_Mean", "Precision_Mean", "F1_Mean", "ROC_AUC_Mean", "Balanced_Acc_Mean"]].style.format({
                    "Recall_Mean": "{:.1%}", "Precision_Mean": "{:.1%}", "F1_Mean": "{:.1%}",
                    "ROC_AUC_Mean": "{:.4f}", "Balanced_Acc_Mean": "{:.1%}"
                }),
                use_container_width=True
            )

        # Clinical Diagnostic Curves
        c_roc, c_pr = st.columns(2)
        with c_roc:
            roc_img = FIGURES_DIR / "roc_curve.png"
            if roc_img.exists():
                st.image(str(roc_img), caption="Receiver Operating Characteristic (ROC) Curve")
        with c_pr:
            pr_img = FIGURES_DIR / "precision_recall_curve.png"
            if pr_img.exists():
                st.image(str(pr_img), caption="Precision-Recall (PR) Curve")

        c_cm, c_cal = st.columns(2)
        with c_cm:
            cm_img = FIGURES_DIR / "confusion_matrix.png"
            if cm_img.exists():
                st.image(str(cm_img), caption="Annotated Holdout Confusion Matrix (Optimal Threshold θ=0.320)")
        with c_cal:
            cal_img = FIGURES_DIR / "calibration_curve.png"
            if cal_img.exists():
                st.image(str(cal_img), caption="Reliability Diagram (Sigmoid Calibration vs Uncalibrated Model)")

    # ==========================================
    # TAB 4: Population Insights & Epidemiology
    # ==========================================
    with tab4:
        st.markdown("### 📖 Population Statistics & Epidemiological Risk Indicators")
        st.markdown("Insights from the 520 patient outpatient cohort from Sylhet Diabetes Hospital.")

        c_odds, c_prev = st.columns(2)
        with c_odds:
            odds_img = FIGURES_DIR / "feature_odds_ratios.png"
            if odds_img.exists():
                st.image(str(odds_img), caption="Epidemiological Odds Ratios (95% Wald CI)")
        with c_prev:
            prev_img = FIGURES_DIR / "symptom_prevalence_comparison.png"
            if prev_img.exists():
                st.image(str(prev_img), caption="Symptom Prevalence Comparison: Positive vs. Negative Cohorts")

        c_bee, c_imp = st.columns(2)
        with c_bee:
            bee_img = FIGURES_DIR / "shap_summary_beeswarm.png"
            if bee_img.exists():
                st.image(str(bee_img), caption="SHAP Global Beeswarm Summary Plot")
        with c_imp:
            imp_img = FIGURES_DIR / "shap_feature_importance_bar.png"
            if imp_img.exists():
                st.image(str(imp_img), caption="SHAP Global Feature Importance (Mean |SHAP|)")

    # Footer Medical Disclaimer
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Clinical & Regulatory Disclaimer</strong>: This software is designed for academic demonstration, pre-clinical screening, and triage decision support under World Health Organization (WHO) and American Diabetes Association (ADA) early detection guidelines. It does NOT constitute formal medical diagnosis. Any individual flagged with elevated risk must undergo clinical diagnostic validation via laboratory venipuncture Fasting Plasma Glucose (FPG), Oral Glucose Tolerance Test (OGTT), or Glycated Hemoglobin (HbA1c) testing by a licensed physician.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
