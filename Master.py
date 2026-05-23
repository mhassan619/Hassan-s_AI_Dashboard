import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Hassan AI Suite", layout="wide", initial_sidebar_state="collapsed")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    label { color: #ffffff !important; font-weight: bold !important; font-size: 1.1rem !important; }
    .main-header {
        font-size: 45px; font-weight: 800; text-align: center;
        background: -webkit-linear-gradient(#00c6ff, #0072ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #161b22; color: white; border: 1px solid #30363d;
        padding: 30px; border-radius: 15px; font-size: 18px; font-weight: bold; width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# --- HOME ---
if st.session_state.page == 'Home':
    st.markdown('<div class="main-header">MACHINE LEARNING DASHBOARD OF HASSAN🚀</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🎓 Student Performance"): st.session_state.page = 'Student'; st.rerun()
    with col2:
        if st.button("🏡 House Price"): st.session_state.page = 'House'; st.rerun()
    with col3:
        if st.button("📉 Customer Churn"): st.session_state.page = 'Churn'; st.rerun()

# --- 1. STUDENT (Delta Generator Fix) ---
elif st.session_state.page == 'Student':
    st.button("⬅️ Back", on_click=lambda: st.session_state.update({"page": "Home"}))
    st.header("🎓 Student Performance Predictor")
    c1, c2 = st.columns(2)
    with c1:
        hrs, att = st.number_input("Study Hours", value=6), st.number_input("Attendance (%)", value=80)
    with c2:
        prev, parent = st.number_input("Previous Scores", value=70), st.selectbox("Parent Education", [0, 1, 2, 3])
    if st.button("Predict Result"):
        try:
            model, scaler = pickle.load(open('student_model.pkl', 'rb')), pickle.load(open('scaler.pkl', 'rb'))
            full_feats = np.array([[0, 0, parent, 0, 0, hrs, att, prev]]) 
            res = model.predict(scaler.transform(full_feats))[0]
            if res == 1: st.success("Likely to Pass ✅")
            else: st.error("At Risk of Failing ⚠️")
        except Exception as e: st.error(f"Error: {e}")

# --- 2. HOUSE (No Changes) ---
elif st.session_state.page == 'House':
    st.button("⬅️ Back", on_click=lambda: st.session_state.update({"page": "Home"}))
    st.header("🏡 House Price Estimator")
    c1, c2, c3 = st.columns(3)
    with c1:
        bed, sq_liv, floors = st.number_input("Bedrooms", value=3), st.number_input("Sqft Living", value=1800), st.number_input("Floors", value=1)
    with c2:
        bath, sq_lot, water = st.number_input("Bathrooms", value=2), st.number_input("Sqft Lot", value=5000), st.selectbox("Waterfront", [0, 1])
    with c3:
        grade, yr, cond = st.slider("Grade", 1, 13, 7), st.number_input("Year Built", value=1990), st.slider("Condition", 1, 5, 3)
    if st.button("Estimate Price"):
        try:
            model = pickle.load(open('house_rf_model.pkl', 'rb'))
            feats = [[bed, bath, sq_liv, sq_lot, floors, water, 0, cond, grade, sq_liv, 0, yr, 0, 98178, 47, -122, 1500, 5000]]
            st.metric("Price", f"${model.predict(feats)[0]:,.2f}")
        except Exception as e: st.error(f"Error: {e}")

# --- 3. CHURN (Dataframe Fix for XGBoost) ---
elif st.session_state.page == 'Churn':
    st.button("⬅️ Back", on_click=lambda: st.session_state.update({"page": "Home"}))
    st.header("📉 Business Churn Analytics")
    c1, c2 = st.columns(2)
    with c1:
        tenure = st.number_input("Tenure (Months)", value=12)
        monthly = st.number_input("Monthly Charges ($)", value=70)
    with c2:
        total = st.number_input("Total Charges ($)", value=840)
        contract = st.selectbox("Contract Type", [0, 1, 2])
    
    if st.button("Analyze Risk"):
        try:
            model = pickle.load(open('churn_xgb_model.pkl', 'rb'))
            # XGBoost needs DataFrame if it was trained with one
            cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
            # Creating dummy data for 19 columns
            data = np.zeros((1, 19))
            data[0, 4] = tenure
            data[0, 14] = contract
            data[0, 17] = monthly
            data[0, 18] = total
            df_input = pd.DataFrame(data, columns=cols)
            
            prediction = model.predict(df_input)
            if prediction[0] == 1: st.error("High Risk: Likely to Churn! ⚠️")
            else: st.success("Low Risk: Loyal Customer ✅")
        except Exception as e: st.error(f"Logic Error: {e}")