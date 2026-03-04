import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# --------------------------------
# Page Configuration
# --------------------------------
st.set_page_config(
    page_title="CKD Risk Monitoring System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------
# Authentication
# --------------------------------
def login():
    st.title("🔐 CKD Risk Monitoring Login")
    st.markdown("Please enter your credentials to continue.")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "sam" and password == "sam1234":
            st.session_state.logged_in = True
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# --------------------------------
# Load Model
# --------------------------------
MODEL_PATH = "outputs_ckd/ckd_rf_model.pkl"
FEATURES_PATH = "outputs_ckd/feature_names.pkl"

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

# --------------------------------
# Initialize Session Storage
# --------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "feedback" not in st.session_state:
    st.session_state.feedback = []

# --------------------------------
# Sidebar Navigation
# --------------------------------
st.sidebar.title("🧭 Navigation")

page = st.sidebar.radio(
    "",
    ["Home", "Dashboard", "History", "Feedback", "About"]
)

st.sidebar.markdown("---")

st.sidebar.metric("Total Predictions", len(st.session_state.history))

if st.session_state.history:
    avg_risk = np.mean([h["probability"] for h in st.session_state.history])
    st.sidebar.metric("Average CKD Risk", f"{avg_risk:.2f}")

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ==========================================================
# HOME PAGE
# ==========================================================
if page == "Home":

    st.title("🏠 CKD Risk Monitoring System")
    st.markdown("""
    ### 🩺 What is Chronic Kidney Disease (CKD)?

    Chronic Kidney Disease (CKD) is a long-term condition where 
    the kidneys gradually lose their ability to filter waste 
    and excess fluids from the blood.

    Early detection can prevent:
    - Kidney failure
    - Cardiovascular disease
    - Metabolic complications
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⚠ Major Risk Factors")
        st.markdown("""
        - High blood pressure  
        - Diabetes  
        - Elevated creatinine  
        - Obesity  
        - Long-term NSAID use  
        """)

    with col2:
        st.subheader("🧪 Clinical Indicators Used")
        st.markdown("""
        - Age  
        - Serum Creatinine  
        - Blood Pressure  
        - Laboratory dataset features  
        """)

    st.info("👉 Navigate to the Dashboard to start predicting CKD risk.")

# ==========================================================
# DASHBOARD PAGE
# ==========================================================
elif page == "Dashboard":

    st.title("📊 CKD Risk Prediction Dashboard")
    st.caption("Clinical Decision Support powered by Random Forest")

    st.subheader("Patient Clinical Details")

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age (years)", 1, 100, 45)

    with c2:
        creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.1, 20.0, 2.1)

    with c3:
        bp = st.number_input("Blood Pressure (mmHg)", 50, 200, 120)

    drug = st.selectbox(
        "Prescribed Drug",
        ["None", "Ibuprofen", "Aspirin", "Paracetamol", "Metformin"]
    )

    if st.button("🔍 Predict CKD Risk"):

        input_dict = {f: 0 for f in feature_names}

        for f in input_dict:
            lf = f.lower()
            if "age" in lf:
                input_dict[f] = age
            elif "creatinine" in lf:
                input_dict[f] = creatinine
            elif "bp" in lf or "blood" in lf:
                input_dict[f] = bp

        input_df = pd.DataFrame([input_dict])

        prob = model.predict_proba(input_df)[0][1]
        label = "High CKD Risk" if prob >= 0.65 else "Low CKD Risk"

        st.subheader("📌 Prediction Result")

        colA, colB = st.columns(2)
        colA.metric("Risk Level", label)
        colB.metric("Probability", f"{prob:.3f}")

        # Risk Interpretation
        if prob >= 0.80:
            st.error("⚠ Very High Risk – Immediate nephrology consultation recommended.")
        elif prob >= 0.65:
            st.warning("⚠ Moderate Risk – Clinical monitoring advised.")
        else:
            st.success("✅ Low Risk – Maintain healthy lifestyle and routine monitoring.")

        # Drug Safety
        st.subheader("💊 Prescription Safety")

        if drug.lower() == "ibuprofen" and prob >= 0.65:
            st.error("⚠ NSAIDs may worsen kidney function in high-risk patients.")
        else:
            st.success("✅ Selected drug appears safe for current CKD risk level.")

        # Save to history
        st.session_state.history.append({
            "time": datetime.now(),
            "age": age,
            "creatinine": creatinine,
            "bp": bp,
            "probability": prob,
            "risk": label
        })

    # Live Monitoring
    if st.session_state.history:
        st.subheader("📈 Live Risk Monitoring")

        hist_df = pd.DataFrame(st.session_state.history)

        col1, col2 = st.columns(2)

        with col1:
            st.line_chart(hist_df.set_index("time")["probability"])

        with col2:
            st.bar_chart(hist_df[["age", "creatinine", "bp"]])

# ==========================================================
# HISTORY PAGE
# ==========================================================
elif page == "History":

    st.title("📜 Prediction History")

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)

        st.dataframe(hist_df)

        st.download_button(
            "⬇ Download CSV Report",
            hist_df.to_csv(index=False),
            "ckd_history_report.csv",
            "text/csv"
        )
    else:
        st.info("No prediction history available yet.")

# ==========================================================
# FEEDBACK PAGE
# ==========================================================
elif page == "Feedback":

    st.title("📝 System Feedback")

    name = st.text_input("Your Name")
    rating = st.slider("Rate the system (1-5)", 1, 5, 3)
    comments = st.text_area("Your Feedback")

    if st.button("Submit Feedback"):

        st.session_state.feedback.append({
            "name": name,
            "rating": rating,
            "comments": comments,
            "time": datetime.now()
        })

        st.success("Thank you for your feedback!")

    if st.session_state.feedback:
        st.subheader("Previous Feedback")
        fb_df = pd.DataFrame(st.session_state.feedback)
        st.dataframe(fb_df)

# ==========================================================
# ABOUT PAGE
# ==========================================================
elif page == "About":

    st.title("ℹ About This System")

    st.markdown("""
    ### 🔬 Technical Details
    - Algorithm: Random Forest Classifier  
    - Class Imbalance Handling: SMOTE  
    - Threshold: 0.65  
    - Evaluation: Accuracy, Precision, Recall, F1  

    ### 🎯 Purpose
    This system assists healthcare professionals in:
    - Early CKD detection  
    - Monitoring patient risk trends  
    - Avoiding nephrotoxic prescriptions  
    - Supporting clinical decision-making  

    ### ⚠ Disclaimer
    This tool is for research and educational purposes only.
    It does not replace professional medical diagnosis.
    """)