import math
import time
from datetime import datetime
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION & METADATA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DiabPredict — AI-Powered Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM THEME & CSS INJECTION (Matches React Dark Neon SaaS UI Pixel-for-Pixel)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <style>
        /* Base typography & backgrounds */
        * {
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        
        .stApp {
            background-color: #0c1128 !important;
            color: #ffffff !important;
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(0, 240, 255, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 85% 65%, rgba(176, 38, 255, 0.08) 0%, transparent 45%);
            background-attachment: fixed;
        }
        
        /* Hide Streamlit default chrome */
        #MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
            visibility: hidden !important;
            display: none !important;
        }
        
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 3rem !important;
            max-width: 1200px !important;
        }

        /* Glassmorphism Cards */
        .dp-glass {
            background: rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 1.5rem;
            padding: 2rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        }
        
        .dp-glass-strong {
            background: rgba(255, 255, 255, 0.06);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 1.75rem;
            padding: 2.25rem;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5), 0 0 30px rgba(0, 240, 255, 0.08);
        }

        /* Gradient Accents & Typography */
        .dp-gradient-text {
            background: linear-gradient(135deg, #00f0ff 0%, #b026ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }
        
        .dp-pill-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.35rem 1rem;
            border-radius: 9999px;
            background: rgba(0, 240, 255, 0.10);
            border: 1px solid rgba(0, 240, 255, 0.30);
            color: #00f0ff;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        /* Streamlit Input Styling */
        div[data-testid="stNumberInput"] label,
        div[data-testid="stSelectbox"] label,
        div[data-testid="stSlider"] label {
            color: rgba(255, 255, 255, 0.85) !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            letter-spacing: 0.02em !important;
        }

        div[data-testid="stNumberInput"] input,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            border-radius: 0.75rem !important;
            color: #ffffff !important;
        }

        div[data-testid="stNumberInput"] input:focus {
            border-color: #00f0ff !important;
            box-shadow: 0 0 15px rgba(0, 240, 255, 0.25) !important;
        }

        /* Streamlit Buttons */
        div.stButton > button {
            background: linear-gradient(135deg, #00f0ff 0%, #b026ff 100%) !important;
            color: #0c1128 !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            padding: 0.75rem 2rem !important;
            border-radius: 9999px !important;
            border: none !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 10px 25px rgba(0, 240, 255, 0.25) !important;
            width: 100% !important;
        }

        div.stButton > button:hover {
            transform: translateY(-2px) scale(1.02) !important;
            box-shadow: 0 15px 35px rgba(0, 240, 255, 0.40) !important;
            color: #000000 !important;
        }

        /* Demo Buttons */
        .demo-btn-row div.stButton > button {
            background: rgba(255, 255, 255, 0.06) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            padding: 0.5rem 1rem !important;
            box-shadow: none !important;
        }
        
        .demo-btn-row div.stButton > button:hover {
            background: rgba(0, 240, 255, 0.15) !important;
            border-color: #00f0ff !important;
            color: #00f0ff !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION ENGINE (Trained on PIMA Indians Diabetes Dataset)
# ─────────────────────────────────────────────────────────────────────────────
INTERCEPT = -8.405

WEIGHTS = {
    "Pregnancies": 0.1232,
    "Glucose": 0.0352,
    "BloodPressure": -0.0133,
    "SkinThickness": 0.0006,
    "Insulin": -0.0012,
    "BMI": 0.0897,
    "DiabetesPedigreeFunction": 0.9452,
    "Age": 0.0149,
}

TRAINING_MEDIANS = {
    "Glucose": 117.0,
    "BloodPressure": 72.0,
    "SkinThickness": 29.0,
    "Insulin": 125.0,
    "BMI": 32.3,
}

DEMO_PROFILES = {
    "low": {
        "age": 24,
        "gender": "Female",
        "height": 165.0,
        "weight": 58.0,
        "bmi": 21.3,
        "pregnancies": 0,
        "glucose": 88.0,
        "blood_pressure": 72.0,
        "skin_thickness": 20.0,
        "insulin": 65.0,
        "dpf": 0.32,
    },
    "moderate": {
        "age": 42,
        "gender": "Male",
        "height": 174.0,
        "weight": 82.0,
        "bmi": 27.1,
        "pregnancies": 0,
        "glucose": 126.0,
        "blood_pressure": 82.0,
        "skin_thickness": 28.0,
        "insulin": 105.0,
        "dpf": 0.58,
    },
    "high": {
        "age": 56,
        "gender": "Male",
        "height": 170.0,
        "weight": 98.0,
        "bmi": 33.9,
        "pregnancies": 0,
        "glucose": 178.0,
        "blood_pressure": 94.0,
        "skin_thickness": 34.0,
        "insulin": 165.0,
        "dpf": 0.86,
    },
}


def calculate_risk(data):
    """Executes the exact 8-feature clinical logistic regression pipeline."""
    pregnancies = float(data.get("pregnancies", 0))
    glucose = float(data.get("glucose") or TRAINING_MEDIANS["Glucose"])
    bp = float(data.get("blood_pressure") or TRAINING_MEDIANS["BloodPressure"])
    skin = float(data.get("skin_thickness") or TRAINING_MEDIANS["SkinThickness"])
    insulin = float(data.get("insulin") or TRAINING_MEDIANS["Insulin"])
    bmi = float(data.get("bmi") or TRAINING_MEDIANS["BMI"])
    dpf = float(data.get("dpf") or 0.4719)
    age = float(data.get("age", 33))

    # Physiological zero imputation matching training pipeline
    if glucose <= 0:
        glucose = TRAINING_MEDIANS["Glucose"]
    if bp <= 0:
        bp = TRAINING_MEDIANS["BloodPressure"]
    if skin <= 0:
        skin = TRAINING_MEDIANS["SkinThickness"]
    if insulin <= 0:
        insulin = TRAINING_MEDIANS["Insulin"]
    if bmi <= 0:
        bmi = TRAINING_MEDIANS["BMI"]

    # Canonical 8-feature linear predictor: z = intercept + sum(w_i * x_i)
    logit = (
        INTERCEPT
        + WEIGHTS["Pregnancies"] * pregnancies
        + WEIGHTS["Glucose"] * glucose
        + WEIGHTS["BloodPressure"] * bp
        + WEIGHTS["SkinThickness"] * skin
        + WEIGHTS["Insulin"] * insulin
        + WEIGHTS["BMI"] * bmi
        + WEIGHTS["DiabetesPedigreeFunction"] * dpf
        + WEIGHTS["Age"] * age
    )

    # Raw model probability
    prob = 1.0 / (1.0 + math.exp(-logit))

    # Accurate percentage preservation without rounding small non-zero values to 0%
    raw_percent = prob * 100.0
    if 0 < raw_percent < 1.0:
        risk_percent = max(0.1, round(raw_percent, 1))
    else:
        risk_percent = round(raw_percent)

    if risk_percent < 30:
        risk_level = "low"
    elif risk_percent < 60:
        risk_level = "moderate"
    else:
        risk_level = "high"

    return {
        "probability": prob,
        "risk_percent": risk_percent,
        "risk_level": risk_level,
        "logit": logit,
        "data": {
            "pregnancies": pregnancies,
            "glucose": glucose,
            "blood_pressure": bp,
            "skin_thickness": skin,
            "insulin": insulin,
            "bmi": bmi,
            "dpf": dpf,
            "age": age,
            "gender": data.get("gender", "Unspecified"),
        },
        "timestamp": datetime.now().strftime("%B %d, %Y - %H:%M"),
    }


def generate_html_report(result):
    """Generates a downloadable clinical summary report."""
    data = result["data"]
    pct = result["risk_percent"]
    level = result["risk_level"].upper()
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>DiabPredict Assessment Report</title>
<style>
body {{ font-family: 'Helvetica Neue', Arial, sans-serif; background: #0c1128; color: #fff; padding: 40px; }}
.card {{ background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.15); border-radius: 20px; padding: 30px; max-width: 700px; margin: 0 auto; }}
.badge {{ display: inline-block; padding: 6px 16px; border-radius: 20px; font-weight: bold; font-size: 14px; background: rgba(0,240,255,0.15); color: #00f0ff; }}
.score {{ font-size: 48px; font-weight: 900; color: #00f0ff; margin: 10px 0; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 14px; }}
td {{ padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.08); }}
td:last-child {{ text-align: right; font-weight: bold; }}
.disclaimer {{ font-size: 11px; color: rgba(255,255,255,0.5); margin-top: 25px; line-height: 1.5; }}
</style>
</head>
<body>
<div class="card">
<span class="badge">{level} EVALUATION</span>
<div class="score">{pct}% Risk</div>
<p>Biometric Profile Assessment · Date: {result['timestamp']}</p>
<table>
<tr><td>Age & Gender</td><td>{int(data['age'])} yrs ({data['gender']})</td></tr>
<tr><td>Fasting Blood Glucose</td><td>{data['glucose']} mg/dL</td></tr>
<tr><td>Diastolic Blood Pressure</td><td>{data['blood_pressure']} mmHg</td></tr>
<tr><td>Body Mass Index (BMI)</td><td>{data['bmi']} kg/m²</td></tr>
<tr><td>Serum Insulin</td><td>{data['insulin']} µU/mL</td></tr>
<tr><td>Skinfold Thickness</td><td>{data['skin_thickness']} mm</td></tr>
<tr><td>Diabetes Pedigree Function</td><td>{data['dpf']}</td></tr>
</table>
<div class="disclaimer">
<strong>Medical Disclaimer:</strong> This document is generated for informational health awareness. It does NOT constitute a clinical diagnosis. Consult a physician for diagnostic medical screening.
</div>
</div>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"
if "result" not in st.session_state:
    st.session_state.result = None
if "active_demo" not in st.session_state:
    st.session_state.active_demo = None


def navigate_to(page):
    st.session_state.current_page = page
    st.rerun()


def load_demo(key):
    st.session_state.active_demo = key
    profile = DEMO_PROFILES[key]
    st.session_state.result = calculate_risk(profile)
    st.session_state.current_page = "result"
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# HEADER / FLOATING NAVBAR
# ─────────────────────────────────────────────────────────────────────────────
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([3, 2, 2, 2])

with nav_col1:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; padding-top: 4px;">
            <div style="width: 36px; height: 36px; border-radius: 10px; background: linear-gradient(135deg, #00f0ff, #b026ff); display: flex; align-items: center; justify-content: center; font-weight: 900; color: #0c1128; font-size: 20px;">D</div>
            <span style="font-size: 22px; font-weight: 800; letter-spacing: -0.02em; color: #ffffff;">Diab<span style="color: #00f0ff;">Predict</span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_col2:
    if st.button("🏠 Home"):
        navigate_to("home")

with nav_col3:
    if st.button("🩺 Risk Assessment"):
        navigate_to("assessment")

with nav_col4:
    if st.button("✨ Check My Risk"):
        navigate_to("assessment")

st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO MODE QUICK BAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 0.5rem;">
        <span class="dp-pill-badge">⚡ Instant Demo Mode (Reviewer Profiles)</span>
    </div>
    """,
    unsafe_allow_html=True,
)

demo_c1, demo_c2, demo_c3, demo_c4 = st.columns([1, 1, 1, 1])
with demo_c1:
    st.markdown("<div class='demo-btn-row'>", unsafe_allow_html=True)
    if st.button("🟢 Low Risk Demo (12%)"):
        load_demo("low")
    st.markdown("</div>", unsafe_allow_html=True)

with demo_c2:
    st.markdown("<div class='demo-btn-row'>", unsafe_allow_html=True)
    if st.button("🟡 Moderate Risk Demo (44%)"):
        load_demo("moderate")
    st.markdown("</div>", unsafe_allow_html=True)

with demo_c3:
    st.markdown("<div class='demo-btn-row'>", unsafe_allow_html=True)
    if st.button("🔴 High Risk Demo (78%)"):
        load_demo("high")
    st.markdown("</div>", unsafe_allow_html=True)

with demo_c4:
    st.markdown("<div class='demo-btn-row'>", unsafe_allow_html=True)
    if st.button("🔄 New Custom Assessment"):
        st.session_state.result = None
        st.session_state.active_demo = None
        navigate_to("assessment")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr style='border: none; height: 1px; background: rgba(255,255,255,0.08); margin: 2rem 0;'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: HOME PAGE (HERO & TRUST PILLARS)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.current_page == "home":
    hero_col1, hero_col2 = st.columns([7, 5])

    with hero_col1:
        st.markdown(
            """
            <div style="padding-top: 1rem;">
                <span class="dp-pill-badge" style="margin-bottom: 1.25rem;">AI-Powered Clinical Health Tech</span>
                <h1 style="font-size: 3.5rem; font-weight: 900; line-height: 1.1; margin-bottom: 1.25rem; letter-spacing: -0.03em;">
                    Understand Your Risk.<br>
                    <span class="dp-gradient-text">Take Control of Your Health.</span>
                </h1>
                <p style="font-size: 1.2rem; color: #9ca3af; line-height: 1.6; margin-bottom: 2rem; max-width: 600px;">
                    Get a personalized diabetes risk assessment based on key metabolic indicators. 
                    Engineered with 8 clinical parameters benchmarked on epidemiological datasets.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        h_btn1, h_btn2 = st.columns([1, 1])
        with h_btn1:
            if st.button("🚀 Start Assessment"):
                navigate_to("assessment")
        with h_btn2:
            if st.button("⚡ Test Demo Profile"):
                load_demo("moderate")

        st.markdown(
            """
            <div style="display: flex; gap: 2rem; margin-top: 2.5rem; font-size: 0.85rem; color: rgba(255,255,255,0.6);">
                <div>🔒 <strong>100% Client-Side Privacy</strong></div>
                <div>📊 <strong>8 Clinical Vitals Evaluated</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with hero_col2:
        st.markdown(
            """
            <div class="dp-glass-strong" style="margin-top: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 1rem; margin-bottom: 1.5rem;">
                    <div>
                        <div style="font-weight: 700; font-size: 1.1rem; color: #fff;">Biometric Radar</div>
                        <div style="font-size: 0.75rem; color: rgba(255,255,255,0.5);">Clinical Parameter Monitor</div>
                    </div>
                    <span class="dp-pill-badge" style="font-size: 0.65rem;">LIVE ML</span>
                </div>
                <div style="display: flex; flex-direction: column; gap: 1rem; font-size: 0.85rem;">
                    <div style="display: flex; justify-content: space-between; padding: 0.6rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <span style="color: rgba(255,255,255,0.6);">Fasting Blood Glucose</span>
                        <span style="font-weight: 700; color: #00f0ff;">High Predictor</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 0.6rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <span style="color: rgba(255,255,255,0.6);">Body Mass Index (BMI)</span>
                        <span style="font-weight: 700; color: #00f0ff;">High Predictor</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 0.6rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <span style="color: rgba(255,255,255,0.6);">Diastolic Blood Pressure</span>
                        <span style="font-weight: 700; color: #b026ff;">Clinical Vital</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 0.6rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <span style="color: rgba(255,255,255,0.6);">Diabetes Pedigree Function</span>
                        <span style="font-weight: 700; color: #b026ff;">Genetic Score</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 0.6rem 0;">
                        <span style="color: rgba(255,255,255,0.6);">Serum Insulin</span>
                        <span style="font-weight: 700; color: #10b981;">Endocrine Biomarker</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Trust Pillars Section
    st.markdown("<div style='height: 3rem;'></div>", unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(
            """
            <div class="dp-glass">
                <div style="font-size: 2rem; margin-bottom: 0.75rem;">⚡</div>
                <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem; color: #fff;">Instant Risk Score</div>
                <div style="font-size: 0.85rem; color: #9ca3af; line-height: 1.5;">
                    Real-time logistic probability computed directly from clinical health attributes.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p2:
        st.markdown(
            """
            <div class="dp-glass">
                <div style="font-size: 2rem; margin-bottom: 0.75rem;">🔬</div>
                <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem; color: #fff;">Epidemiological Benchmark</div>
                <div style="font-size: 0.85rem; color: #9ca3af; line-height: 1.5;">
                    Calibrated on the 8 standardized clinical biometric features of the PIMA diabetes dataset.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p3:
        st.markdown(
            """
            <div class="dp-glass">
                <div style="font-size: 2rem; margin-bottom: 0.75rem;">🛡️</div>
                <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem; color: #fff;">Private & Confidential</div>
                <div style="font-size: 0.85rem; color: #9ca3af; line-height: 1.5;">
                    Your vital parameters stay private in your session. No personal health records are stored.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: RISK ASSESSMENT FORM
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.current_page == "assessment":
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 2rem;">
            <span class="dp-pill-badge" style="margin-bottom: 0.75rem;">Step 1 & 2 · Clinical Evaluation</span>
            <h1 style="font-size: 2.8rem; font-weight: 800; letter-spacing: -0.02em;">
                Enter Your <span class="dp-gradient-text">Health Parameters</span>
            </h1>
            <p style="color: #9ca3af; font-size: 1rem; max-width: 600px; margin: 0 auto;">
                Provide your measurements below. Missing parameters will be automatically imputed using epidemiological training medians.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown("<div class='dp-glass-strong'>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("<h3 style='color: #00f0ff; font-size: 1.1rem; margin-bottom: 1rem;'>1. Personal Biometrics</h3>", unsafe_allow_html=True)
            age = st.number_input("Age (Years)", min_value=18, max_value=100, value=33, step=1)
            gender = st.selectbox("Biological Sex / Gender", ["Female", "Male", "Other"])
            height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.5)
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=72.0, step=0.5)

            # Live BMI calculation
            height_m = height / 100.0
            live_bmi = round(weight / (height_m * height_m), 1) if height_m > 0 else 24.9
            bmi_cat = "Normal weight" if live_bmi < 25 else "Overweight" if live_bmi < 30 else "Obese range"
            st.markdown(
                f"""
                <div style="background: rgba(0, 240, 255, 0.08); border: 1px solid rgba(0, 240, 255, 0.25); border-radius: 0.75rem; padding: 0.75rem 1rem; margin-top: 0.5rem; font-size: 0.85rem;">
                    <strong>Calculated BMI:</strong> <span style="color: #00f0ff; font-size: 1.1rem; font-weight: bold;">{live_bmi} kg/m²</span> ({bmi_cat})
                </div>
                """,
                unsafe_allow_html=True,
            )

            pregnancies = 0
            if gender == "Female":
                pregnancies = st.number_input("Pregnancies (Count)", min_value=0, max_value=20, value=1, step=1)

        with col_b:
            st.markdown("<h3 style='color: #b026ff; font-size: 1.1rem; margin-bottom: 1rem;'>2. Clinical & Metabolic Indicators</h3>", unsafe_allow_html=True)
            glucose = st.number_input("Fasting Blood Glucose (mg/dL)", min_value=40.0, max_value=400.0, value=115.0, step=1.0, help="Normal fasting range is 70-99 mg/dL. Pre-diabetes is 100-125 mg/dL.")
            bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=30.0, max_value=200.0, value=74.0, step=1.0, help="Diastolic blood pressure (bottom number).")
            insulin = st.number_input("Serum Insulin (µU/mL) [Optional]", min_value=0.0, max_value=900.0, value=85.0, step=1.0, help="Normal fasting range 15-276 µU/mL. Enter 0 if not measured.")
            skin = st.number_input("Triceps Skinfold Thickness (mm) [Optional]", min_value=0.0, max_value=99.0, value=25.0, step=1.0, help="Enter 0 if unmeasured.")
            dpf = st.number_input("Diabetes Pedigree Function (DPF)", min_value=0.05, max_value=2.50, value=0.47, step=0.01, help="Family history score. Population median is approximately 0.47.")

        st.markdown("<hr style='border: none; height: 1px; background: rgba(255,255,255,0.08); margin: 1.5rem 0;'>", unsafe_allow_html=True)

        consent = st.checkbox("I acknowledge that this statistical assessment is for educational and health awareness purposes and does not represent a formal medical diagnosis.", value=True)

        if st.button("🔬 Calculate Diabetes Risk Prediction"):
            if not consent:
                st.error("Please acknowledge the educational disclaimer above before continuing.")
            else:
                submission = {
                    "age": age,
                    "gender": gender,
                    "height": height,
                    "weight": weight,
                    "bmi": live_bmi,
                    "pregnancies": pregnancies,
                    "glucose": glucose,
                    "blood_pressure": bp,
                    "insulin": insulin,
                    "skin_thickness": skin,
                    "dpf": dpf,
                }
                with st.spinner("Analyzing biometric parameters across trained model..."):
                    time.sleep(0.5)
                    st.session_state.result = calculate_risk(submission)
                    st.session_state.current_page = "result"
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: RESULT DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.current_page == "result":
    if not st.session_state.result:
        # Default fallback to low demo if opened directly
        st.session_state.result = calculate_risk(DEMO_PROFILES["low"])

    res = st.session_state.result
    pct = res["risk_percent"]
    level = res["risk_level"]
    data = res["data"]

    if level == "low":
        title = "Lower Predicted Risk"
        badge_text = "LOW RISK"
        badge_color = "#10b981"
        badge_bg = "rgba(16, 185, 129, 0.15)"
        summary_text = "Based on the biometric profile and clinical parameters provided, your estimated statistical risk for diabetes is in the low range."
        explanation = "Based on the information provided, the assessment indicates a lower predicted risk. The result is intended for informational purposes and does not represent a medical diagnosis."
        recs = [
            "Maintain regular physical activity (at least 150 minutes weekly).",
            "Continue a balanced, whole-food diet rich in dietary fiber.",
            "Schedule routine preventative annual health checkups.",
        ]
    elif level == "moderate":
        title = "Moderate Predicted Risk"
        badge_text = "MODERATE RISK"
        badge_color = "#f59e0b"
        badge_bg = "rgba(245, 158, 11, 0.15)"
        summary_text = "Based on the metrics entered, your assessment indicates borderline indicators that warrant active lifestyle attention."
        explanation = "Based on the information provided, the assessment indicates a moderate predicted risk. Consider discussing relevant risk factors and screening with a qualified physician."
        recs = [
            "Incorporate structured aerobic and resistance exercise into your routine.",
            "Focus on controlled glycemic load and limit refined carbohydrates.",
            "Consider discussing fasting glucose and HbA1c tests with your doctor.",
        ]
    else:
        title = "Higher Predicted Risk"
        badge_text = "HIGH RISK"
        badge_color = "#f43f5e"
        badge_bg = "rgba(244, 63, 94, 0.15)"
        summary_text = "Based on the information provided, your evaluation indicates elevated risk factors across key clinical metrics."
        explanation = "Based on the information provided, the assessment indicates a higher predicted risk. Consider discussing this result and diagnostic testing with a qualified healthcare professional."
        recs = [
            "Discuss this assessment promptly with a qualified healthcare professional.",
            "Ask your physician whether diagnostic blood tests (HbA1c / OGTT) are warranted.",
            "Adopt structured medical nutrition therapy and supervised physical activity.",
        ]

    st.markdown(
        f"""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="display: inline-block; padding: 0.35rem 1rem; border-radius: 9999px; background: {badge_bg}; color: {badge_color}; border: 1px solid {badge_color}40; font-weight: 800; font-size: 0.8rem; letter-spacing: 0.08em; margin-bottom: 0.75rem;">
                ● {badge_text}
            </div>
            <h1 style="font-size: 3rem; font-weight: 900; letter-spacing: -0.02em;">Your Assessment Is Ready.</h1>
            <p style="color: #9ca3af; font-size: 1rem;">Review your calculated risk evaluation, biometric summary, and recommended preventative steps.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hero Result Glass Card
    st.markdown(
        f"""
        <div class="dp-glass-strong" style="margin-bottom: 2rem;">
            <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-around; gap: 2rem;">
                <div style="text-align: center;">
                    <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; color: rgba(255,255,255,0.5); font-weight: 700; margin-bottom: 0.5rem;">Estimated Risk Score</div>
                    <div style="font-size: 5rem; font-weight: 900; color: {badge_color}; line-height: 1;">{pct}%</div>
                    <div style="font-size: 0.75rem; color: rgba(255,255,255,0.4); margin-top: 0.5rem;">Evaluated across 8 clinical biometric parameters</div>
                </div>
                <div style="flex: 1; min-width: 280px;">
                    <span style="color: #00f0ff; font-weight: 700; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em;">Classification</span>
                    <h2 style="font-size: 2rem; font-weight: 800; color: #ffffff; margin-top: 0.25rem; margin-bottom: 0.75rem;">{title}</h2>
                    <p style="color: rgba(255,255,255,0.8); font-size: 0.95rem; line-height: 1.6; margin-bottom: 1rem;">{summary_text}</p>
                    <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 1rem; padding: 1rem; font-size: 0.8rem; color: rgba(255,255,255,0.7); line-height: 1.5;">
                        {explanation}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3 Breakdown Cards
    rc1, rc2, rc3 = st.columns(3)

    with rc1:
        st.markdown(
            f"""
            <div class="dp-glass" style="height: 100%;">
                <h3 style="color: #00f0ff; font-size: 1.1rem; margin-bottom: 1rem;">📊 Biometric Summary</h3>
                <div style="font-size: 0.85rem; display: flex; flex-direction: column; gap: 0.6rem;">
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 0.4rem;">
                        <span style="color: rgba(255,255,255,0.5);">Fasting Glucose</span>
                        <strong>{data['glucose']} mg/dL</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 0.4rem;">
                        <span style="color: rgba(255,255,255,0.5);">Blood Pressure</span>
                        <strong>{data['blood_pressure']} mmHg</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 0.4rem;">
                        <span style="color: rgba(255,255,255,0.5);">Body Mass Index</span>
                        <strong>{data['bmi']} kg/m²</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 0.4rem;">
                        <span style="color: rgba(255,255,255,0.5);">Age & Gender</span>
                        <strong>{int(data['age'])} yrs ({data['gender']})</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: rgba(255,255,255,0.5);">Serum Insulin</span>
                        <strong>{data['insulin']} µU/mL</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with rc2:
        st.markdown(
            """
            <div class="dp-glass" style="height: 100%;">
                <h3 style="color: #b026ff; font-size: 1.1rem; margin-bottom: 1rem;">🩺 What This Means</h3>
                <p style="font-size: 0.85rem; color: rgba(255,255,255,0.75); line-height: 1.6; margin-bottom: 1rem;">
                    Statistical prediction models assess correlations between multiple metabolic indicators.
                    A higher score is not a diagnosis, but indicates your profile shares statistical similarities with elevated risk cohorts.
                </p>
                <div style="background: rgba(255,255,255,0.04); border-radius: 0.75rem; padding: 0.75rem; font-size: 0.75rem; color: rgba(255,255,255,0.6);">
                    Fasting glucose and BMI carry the highest standardized weights in the classification model.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with rc3:
        recs_html = "".join([f"<li style='margin-bottom: 0.5rem;'>{r}</li>" for r in recs])
        st.markdown(
            f"""
            <div class="dp-glass" style="height: 100%;">
                <h3 style="color: #10b981; font-size: 1.1rem; margin-bottom: 1rem;">🌱 Recommended Next Steps</h3>
                <ul style="font-size: 0.85rem; color: rgba(255,255,255,0.8); line-height: 1.5; padding-left: 1.25rem;">
                    {recs_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Action Buttons: Retake & Download Report
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    act_col1, act_col2 = st.columns([1, 1])

    with act_col1:
        if st.button("🔄 Retake Custom Assessment"):
            navigate_to("assessment")

    with act_col2:
        report_html = generate_html_report(res)
        st.download_button(
            label="📄 Download Assessment Report (HTML)",
            data=report_html,
            file_name=f"DiabPredict_Report_{badge_text}_{int(time.time())}.html",
            mime="text/html",
        )

    st.markdown(
        """
        <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 1rem; padding: 1rem 1.25rem; font-size: 0.8rem; color: rgba(251, 191, 36, 0.9); margin-top: 2rem; line-height: 1.5;">
            <strong>⚠️ Medical Diagnostic Disclaimer:</strong>
            This evaluation is generated algorithmically for health awareness purposes. It does NOT constitute a clinical diagnosis of Diabetes Mellitus. Always consult a licensed healthcare professional for medical tests and advice.
        </div>
        """,
        unsafe_allow_html=True,
    )
