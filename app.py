import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(
    page_title="CMM Risk Prediction Calculator",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

def load_model():
    with open("model/svm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model/features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, scaler, features

model, scaler, FEATURES = load_model()
import os
st.write(os.path.getmtime("model/svm_model.pkl"))


# ── Title ────────────────────────────────────────────────────
st.markdown("""
    <h2 style='text-align:center; color:#2c3e50;'>
        🫀 CMM Risk Prediction Calculator for Elderly Patients with Sarcopenia
    </h2>
    <p style='text-align:center; color:gray; font-size:13px;'>
        ⚠️ For clinical reference only. Not a substitute for professional medical judgment.
    </p>
""", unsafe_allow_html=True)
st.divider()

# ── Input ────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🩺 Physical Examination & Biomarkers")
    systo  = st.number_input("Systolic BP (mmHg)",   60,   260,  130,  step=1)
    weight = st.number_input("Body Weight (kg)",     30.0, 150.0, 60.0, step=0.1)
    bmi    = st.number_input("BMI (kg/m²)",          10.0,  50.0, 23.0, step=0.1)
    grip   = st.number_input("Grip Strength (kg)",    0.0,  80.0, 28.0, step=0.1)
    hba1c  = st.number_input("HbA1c (%)",             3.0,  20.0,  6.1, step=0.1)
    cysc   = st.number_input("Cystatin C (mg/L)",     0.3,   5.0, 0.92, step=0.01)

with col2:
    st.markdown("#### 📋 General Information & Health Status")
    gender    = st.radio("Sex", ["Male", "Female"], horizontal=True)
    cesd      = st.number_input("Depression Score (CESD-10, 0–30)", 0, 30, 8)
    cognition = st.number_input("Cognitive Score (0–30)",           0, 30, 10)
    digest    = st.radio("Digestive System Disease", ["No", "Yes"], horizontal=True)

    st.markdown("""
    <div style='background:#f0f4f8; padding:12px; border-radius:8px;
                font-size:12px; margin-top:16px;'>
    <b>📌 Normal Reference Ranges</b><br>
    Systolic BP: 90–140 mmHg<br>
    BMI: 18.5–23.9 kg/m²<br>
    HbA1c: 4.0–6.0 %<br>
    Cystatin C: 0.51–1.09 mg/L<br>
    Grip Strength (Male): ≥28 kg &nbsp; (Female): ≥18 kg
    </div>
    """, unsafe_allow_html=True)

# ── Predict Button ───────────────────────────────────────────
st.divider()
if st.button("🔍 Predict", type="primary", use_container_width=True):

    input_dict = {
        'systo':           systo,
        'mweight':         weight,
        'bl_hbalc':        hba1c,
        'BMI':             bmi,
        'cesd10':          cesd,
        'total_cognition': cognition,
        'MS':              grip,
        'gender':          0 if gender == "Male" else 1,
        '消化系统疾病':     1 if digest == "Yes" else 0,
        'bl_cysc':         cysc,
    }

    input_df     = pd.DataFrame([input_dict])[FEATURES]
    input_scaled = scaler.transform(input_df)
    prob         = model.predict_proba(input_scaled)[0][1]
    pred         = model.predict(input_scaled)[0]

    # Risk stratification
    if prob < 0.3:
        level, color, bg, advice = (
            "Low Risk", "#27ae60", "#eafaf1",
            "Regular follow-up is recommended. Maintain a healthy lifestyle and annual check-ups."
        )
    elif prob < 0.6:
        level, color, bg, advice = (
            "Moderate Risk", "#e67e22", "#fef9e7",
            "Enhanced monitoring is advised. Actively manage modifiable risk factors such as blood pressure, blood glucose, and body weight."
        )
    else:
        level, color, bg, advice = (
            "High Risk", "#e74c3c", "#fdedec",
            "Prompt referral to cardiology or geriatrics is recommended. Develop an individualized intervention plan with close follow-up."
        )

    # Result card
    st.markdown(f"""
    <div style='background:{bg}; border-left:5px solid {color};
                padding:20px; border-radius:8px; margin-bottom:16px;'>
        <h3 style='color:{color}; margin:0;'>Prediction Result: {level}</h3>
        <p style='font-size:32px; font-weight:bold; margin:8px 0; color:{color};'>
            CMM Risk Probability: {prob*100:.1f}%
        </p>
        <p style='margin:0; color:#555;'>🩺 {advice}</p>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Risk Probability", f"{prob*100:.1f}%")
    m2.metric("Prediction",       "CMM" if pred == 1 else "No CMM")
    m3.metric("Risk Level",       level)

    # Risk gauge
    fig, ax = plt.subplots(figsize=(8, 1.8))
    for i in range(100):
        c = '#2ecc71' if i < 30 else '#f39c12' if i < 60 else '#e74c3c'
        ax.barh(0, 1, left=i, color=c, height=0.6, alpha=0.7)
    ax.axvline(prob * 100, color='#2c3e50', linewidth=3, zorder=5)
    ax.plot(prob * 100, 0, 'v', color='#2c3e50', markersize=12, zorder=6)
    ax.text(prob * 100, 0.38, f'{prob*100:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold', color='#2c3e50')
    ax.text(15,  -0.42, 'Low Risk\n(0–30%)',       ha='center', fontsize=9, color='#27ae60')
    ax.text(45,  -0.42, 'Moderate Risk\n(30–60%)', ha='center', fontsize=9, color='#e67e22')
    ax.text(80,  -0.42, 'High Risk\n(60–100%)',    ha='center', fontsize=9, color='#e74c3c')
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 0.6)
    ax.axis('off')
    ax.set_title("CMM Risk Gauge", fontsize=13, pad=10)
    st.pyplot(fig)
    st.caption("⚠️ This tool is intended for Chinese elderly patients (aged ≥60) with confirmed sarcopenia, based on CHARLS 2015 data. It is not applicable to other populations.")
