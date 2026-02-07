import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Space Debris Decay Predictor",
    page_icon="🛰️",
    layout="centered"
)

# -------------------------------------------------
# Custom CSS (Dark UI + Buttons + Cards)
# -------------------------------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #0e1117;
    }

    div.stButton > button {
        background-color: #2563eb;
        color: white;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        font-size: 16px;
        font-weight: 600;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #1e40af;
        color: white;
    }

    .card {
        padding: 18px;
        border-radius: 12px;
        background-color: #161b22;
        border: 1px solid #30363d;
        color: white;
    }

    .risk-card {
        padding: 18px;
        border-radius: 12px;
        background-color: #0d1117;
        border: 1px solid #30363d;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">🛰️ Space Debris Decay Predictor</h1>
    <p style="text-align:center; color:#9ca3af;">
    Predict the orbital decay time of space debris using machine learning
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("spacedata.csv")

df = load_data()

# -------------------------------------------------
# Preprocessing
# -------------------------------------------------
le = LabelEncoder()
df["object_type"] = le.fit_transform(df["object_type"])
df["solar_activity"] = le.fit_transform(df["solar_activity"])

X = df.drop("decay_time_days", axis=1)
y = df["decay_time_days"]

# -------------------------------------------------
# Train Model
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# -------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------
st.sidebar.markdown("## 🔧 Input Parameters")

object_type = st.sidebar.selectbox("Object Type", ["Debris", "Active"])
altitude = st.sidebar.slider("Orbital Altitude (km)", 200, 1200, 500)
inclination = st.sidebar.slider("Inclination (degrees)", 0.0, 98.0, 45.0)
eccentricity = st.sidebar.slider("Eccentricity", 0.0001, 0.05, 0.01)
mass = st.sidebar.slider("Mass (kg)", 1.0, 2000.0, 500.0)
area = st.sidebar.slider("Cross-sectional Area (m²)", 0.01, 20.0, 2.0)
drag_coeff = st.sidebar.slider("Drag Coefficient", 1.8, 2.5, 2.2)
mean_motion = st.sidebar.slider("Mean Motion", 11.0, 16.0, 14.0)
solar_activity = st.sidebar.selectbox("Solar Activity", ["Low", "Medium", "High"])

# -------------------------------------------------
# Encode Inputs
# -------------------------------------------------
object_type_enc = 1 if object_type == "Debris" else 0
solar_activity_enc = {"Low": 0, "Medium": 1, "High": 2}[solar_activity]

input_data = np.array([[
    object_type_enc,
    altitude,
    inclination,
    eccentricity,
    mass,
    area,
    drag_coeff,
    mean_motion,
    solar_activity_enc
]])

# -------------------------------------------------
# Prediction Section
# -------------------------------------------------
st.markdown("## 📊 Prediction")

col1, col2 = st.columns(2)

with col1:
    predict_btn = st.button("🚀 Predict Decay Time", use_container_width=True)

with col2:
    st.markdown(
        """
        <div class="card">
        <b>Model:</b> Random Forest Regressor<br>
        <b>Task:</b> Regression<br>
        <b>Target:</b> Decay Time (days)
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------
# Prediction Output
# -------------------------------------------------
if predict_btn:
    prediction = model.predict(input_data)[0]

    st.markdown("---")
    st.markdown("## 🧠 Prediction Result")

    st.markdown(
        f"""
        <div class="card">
        <h2>{prediction:.1f} days</h2>
        <p style="color:#9ca3af;">Estimated Orbital Decay Time</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------
    # FINAL Risk Logic (Balanced)
    # -----------------------------
    if prediction <= 180:
        risk = "High Risk"
        emoji = "🔴"
        note = "Re-entry expected within 6 months. Immediate monitoring required."
    elif prediction <= 730:
        risk = "Medium Risk"
        emoji = "🟠"
        note = "Re-entry possible within 6–24 months. Regular tracking advised."
    else:
        risk = "Low Risk"
        emoji = "🟢"
        note = "Re-entry not expected soon. Low operational concern."

    st.markdown(
        f"""
        <div class="risk-card">
        <h4>⚠️ Risk Assessment</h4>
        <p style="font-size:18px;"><b>{emoji} {risk}</b></p>
        <p style="color:#9ca3af; font-size:14px;">{note}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "This prediction is generated using a machine learning model trained on "
        "physics-inspired orbital decay data. Results are indicative and "
        "intended for analytical purposes only."
    )

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption(
    "🛰️ Space Debris Decay Predictor | Machine Learning & Streamlit Project"
)
