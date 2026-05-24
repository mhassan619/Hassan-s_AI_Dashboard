<div align="center">

# 🧠 INTEGRATED MACHINE LEARNING DASHBOARD
### 🚀 Predictive Analytics Engine, Core Classification & Regression Systems

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/XGBoost-111111?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
</p>

---

[🌐 Launch Live ML Dashboard](https://hassan-saidashboard-awujmtpzvvsekbjzfgb6y5.streamlit.app/) • [💼 LinkedIn Profile](https://www.linkedin.com/in/muhammad-hassan-python) • [💻 Main Portfolio](https://github.com/mhassan619)

</div>

---

## 📋 Project Overview

This production-ready repository contains the end-to-end machine learning lifecycle infrastructure—spanning **Exploratory Data Analysis (EDA), Feature Engineering, Model Training, Evaluation Benchmarks, and Serialization**. 

All models are integrated into a single, interactive **Streamlit Multi-Page Portal**, allowing non-technical recruiters and stakeholders to interact with complex predictive pipelines via an intuitive web interface.

---

## 🛠️ Machine Learning Models & Pipeline Architecture

The suite consists of three distinct enterprise-grade predictive models:

### 1. 📂 Student Performance Predictor (Classification)
* **Objective:** Predict student academic outcomes and passing probability based on behavioral metrics.
* **Features:** Study hours, historical scores, attendance rates, and parental education profiles.
* **Pipeline:** Localized feature scaling, categorical encoding, and evaluation via Confusion Matrix and F1-Score optimization.
* <a href="https://github.com/mhassan619/Hassan-s_AI_Dashboard/tree/main/Student%20Performance%20Project" target="_blank"><img src="https://img.shields.io/badge/View%20Model%20Pipeline-3478F6?style=flat-square" alt="Pipeline" /></a>

### 2. 📂 House Price Estimator (Regression)
* **Objective:** Estimate real estate valuation through non-linear multi-variable mapping.
* **Core Algorithm:** Random Forest Regressor.
* **Pipeline:** Handles multi-collinearity, implements automated feature ranking, and balances living spaces with regional economic coefficients to prevent illogical prediction drops.
* <a href="https://github.com/mhassan619/Hassan-s_AI_Dashboard/tree/main/House%20Price%20Prediction" target="_blank"><img src="https://img.shields.io/badge/View%20Model%20Pipeline-3478F6?style=flat-square" alt="Pipeline" /></a>

### 3. 📂 Customer Churn Analytics (Advanced Classification)
* **Objective:** Identify high-risk corporate or retail accounts likely to churn.
* **Core Algorithm:** XGBoost (Extreme Gradient Boosting).
* **Pipeline:** Handles imbalanced datasets, processes usage matrices, and ranks high-entropy features to provide actionable retention alerts for business logic.
* <a href="https://github.com/mhassan619/Hassan-s_AI_Dashboard/tree/main/Customer%20Churn%20Prediction" target="_blank"><img src="https://img.shields.io/badge/View%20Model%20Pipeline-3478F6?style=flat-square" alt="Pipeline" /></a>

---

## 📂 Modular System Architecture

To ensure clean code paradigms, model serialization (.pkl matrices) and UI layout rendering are cleanly abstracted:

```text
ML_Dashboard_Repository/
│
├── Master.py                # Dashboard Main UI Router & Layout Hub
├── requirements.txt         # Matrix Math & Model Pipeline dependencies
│
├── Student Performance/     # Classification Architecture
│   ├── student_model.pkl    # Serialized trained scikit-learn pipeline
│   └── app.py               # Localized inference mapping logic
│
├── House Price Prediction/  # Regression Architecture
│   ├── house_model.pkl      # Random Forest weights & mathematical arrays
│   └── app.py               # Inference UI & numerical inputs
│
└── Customer Churn/          # Gradient Boosting Architecture
    ├── churn_model.pkl      # Serialized XGBoost booster matrices
    └── app.py               # Churn probability analytics rendering

```
## 📊 Core Technical Stack
 * **Mathematical Computing:** NumPy & Pandas (High-density matrix operations & vector manipulation)
 * **Statistical Modeling:** Scikit-Learn (Pipelines, Feature scaling, Random Forest implementations)
 * **Gradient Boosting:** XGBoost Engine (Optimized gradient boosting frameworks)
 * **Visualization:** Matplotlib & Seaborn (Feature importance histograms & distribution graphs)
 * **App State Serialization:** Pickle (Binary object saving for lightning-fast model load times)
## 💻 Local Deployment & Setup
Execute the complete ML inference engine on your local workstation:
 1. **Clone the Infrastructure:**
   ```bash
   git clone [https://github.com/mhassan619/Hassan-s_AI_Dashboard.git](https://github.com/mhassan619/Hassan-s_AI_Dashboard.git)
   cd Hassan-s_AI_Dashboard
   
   ```
 2. **Environment & Dependency Loading:**
   ```bash
   python -m venv ml_env
   source ml_env/bin/activate  # Windows: ml_env\Scripts\activate
   pip install -r requirements.txt
   
   ```
 3. **Boot the Analytics Interface:**
   ```bash
   streamlit run Master.py
   
   ```
## ⚡ Engineering Integrity & Standards
 * **No-Crash Architecture:** User input thresholds are constrained natively using Streamlit UI bounds to eliminate math overflows or model array dimension exceptions.
 * **Optimized Inference:** Binary .pkl files load into RAM once at runtime initialization, ensuring prediction sub-second latencies during user input cycles.
<div align="center">
### 🌱 "Consistency beats motivation, every single day."
📩 Query via Email • 🌐 Let's Connect on LinkedIn
</div>
