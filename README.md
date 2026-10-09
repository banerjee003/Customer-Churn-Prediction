<div align="center">
  <img src="assets/logo.png" alt="ChurnShield Logo" width="130" style="border-radius: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.35);" />
  <h1>🛡️ ChurnShield — AI-Powered Customer Churn Prediction System</h1>
</div>

<p align="center">
  <a href="https://customer-churn-prediction-churnshield.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/Live_App-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit Cloud" />
  </a>
  &nbsp;&nbsp;
  <a href="https://churnshield-mwt7.onrender.com/" target="_blank">
    <img src="https://img.shields.io/badge/Live_App-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white" alt="Render Web Service" />
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/Streamlit-1.51-FF4B4B?logo=streamlit" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Scikit--Learn-1.7-F7931E?logo=scikit-learn" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/XAI-SHAP-brightgreen" alt="SHAP" />
  <img src="https://img.shields.io/badge/Database-SQL_Server-CC292B?logo=microsoft-sql-server" alt="SQL Server" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License" />
</p>

> 🌐 **Live Cloud Applications:**
> - **Streamlit Community Cloud:** **[https://customer-churn-prediction-churnshield.streamlit.app/](https://customer-churn-prediction-churnshield.streamlit.app/)**
> - **Render Web Service:** **[https://churnshield-mwt7.onrender.com/](https://churnshield-mwt7.onrender.com/)**

An enterprise-grade, end-to-end Machine Learning solution designed to predict customer churn in the telecommunications sector. **ChurnShield** combines production-ready Scikit-Learn pipelines, hyperparameter-tuned ensemble models, Explainable AI (SHAP), and an interactive dark-themed Streamlit dashboard delivering real-time churn risk scores and data-backed retention strategies.

---

## 📌 Project Highlights & Key Features

- **End-to-End Data Lifecycle**: SQL Server schema design $\rightarrow$ Automated ETL $\rightarrow$ Exploratory Data Analysis $\rightarrow$ Pipeline Serialization $\rightarrow$ Web App Deployment.
- **Leakage-Free Preprocessing**: Bundled `ColumnTransformer` with median imputation, standard scaling, and one-hot encoding (`handle_unknown='ignore'`).
- **Class Imbalance Mitigation**: Addressed the ~26% natural churn imbalance using stratified splits (`stratify=y`) and balanced class weighting (`class_weight='balanced'`).
- **Model Diversity**: Direct comparison between interpretable linear baselines (**Logistic Regression**) and non-linear ensembles (**Random Forest** tuned with `GridSearchCV`).
- **Explainable AI (XAI)**:
  - **Local Interpretability**: Customer-level SHAP Waterfall plots revealing specific push/pull factors.
  - **Global Interpretability**: Population-level SHAP Beeswarm distributions and Mean absolute $|SHAP|$ rankings.
- **Modern Executive Dashboard**: Built with Streamlit featuring responsive glassmorphic cards, custom vector icons, risk assessment tiers, and actionable retention playbooks.

---

## 🔄 End-to-End Workflow Architecture

```mermaid
flowchart TD
    subgraph Data_Tier["1. Data Layer"]
        A[("SQL Server: customerChurnDB")] -->|pyodbc / SQLAlchemy| B["Telco Dataset (7,043 Records)"]
        A2["Telco_Customer_Churn.csv"] -.->|Fallback / Local| B
    end

    subgraph Modeling_Tier["2. Machine Learning Pipeline"]
        B --> C["EDA & Feature Stratification"]
        C --> D["sklearn ColumnTransformer\n(Imputer + Scaler + OneHotEncoder)"]
        D --> E["Model Training & Cross-Validation"]
        E --> F1["Logistic Regression (Balanced)"]
        E --> F2["Random Forest (GridSearchCV Tuned)"]
        F1 --> G["Serialized Artifacts\n(logistic_model.pkl & rf_model.pkl)"]
        F2 --> G
    end

    subgraph XAI_Tier["3. Explainable AI (SHAP)"]
        G --> H1["SHAP TreeExplainer"]
        H1 --> H2["Local Waterfall Plot\n(Individual Customer Risk)"]
        H1 --> H3["Global Beeswarm Summary\n(Feature Distributions)"]
    end

    subgraph App_Tier["4. Streamlit Web Application (app.py)"]
        G --> I["Interactive Dashboard"]
        H2 --> I
        H3 --> I
        I --> J1["Tab 1: Real-Time Prediction Engine"]
        I --> J2["Tab 2: Model Intelligence & Strategic Insights"]
    end
```

---

## 📊 Model Performance Comparison

Evaluated on an independent, stratified 30% test holdout:

| Metric | Logistic Regression (Baseline) | Random Forest (Best Tuned) | Key Takeaway |
| :--- | :---: | :---: | :--- |
| **ROC AUC** | **0.86** | **0.85** | High discriminatory power across classification thresholds |
| **Recall (Churn)** | **0.84** | 0.78 | Catches 84% of all at-risk customers (minimizes false negatives) |
| **Precision** | 0.52 | **0.56** | Random Forest reduces false alarms |
| **F1 Score** | 0.64 | **0.65** | Balanced harmonic mean between precision and recall |

---

## 💡 Key Business Drivers & Strategic Recommendations

Based on SHAP and feature importance analyses, the primary churn drivers are:

1. **Contract Type**: Month-to-month subscribers exhibit the highest churn probability. Long-term (1-year and 2-year) contracts dramatically increase retention.
   - *Action*: Offer tiered introductory discounts or fee waivers for upgrading to annual contracts.
2. **Customer Tenure**: High attrition occurs during the initial 1–6 months of customer onboarding.
   - *Action*: Trigger automated proactive onboarding check-ins and customer success support during month 1 and 3.
3. **Monthly Charges & Fiber Optic**: Higher monthly spending combined with Fiber Optic service correlates with elevated churn if service expectations aren't met.
   - *Action*: Bundle value-added services (Online Security, Device Protection) to heighten perceived value.

---

## 📁 Repository Directory Structure

```plaintext
Customer Churn Prediction/
├── dataset/
│   └── Telco_Customer_Churn.csv    # Source telecommunications dataset (7,043 rows)
├── assets/
│   └── logo.png                    # Project visual branding logo
├── schema.sql                      # Production SQL Server DDL schema with indexes & constraints
├── .env.example                    # Environment variable template for database credentials
├── .env                            # Local configuration (ignored by Git)
├── .gitignore                      # Git exclusion rules for caches, secrets, and environments
├── requirements.txt                # Pinned and tested Python dependencies
├── work.ipynb                      # Complete data science notebook (EDA, training, tuning, XAI)
├── logistic_model.pkl              # Serialized Logistic Regression pipeline
├── rf_model.pkl                    # Serialized Random Forest pipeline with preprocessor
├── app.py                          # Main Streamlit web application
└── README.md                       # Comprehensive documentation
```

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- **Python 3.10+**
- **Microsoft ODBC Driver 18 for SQL Server** (if connecting to a live SQL Server instance)

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/your-username/Customer-Churn-Prediction.git
cd "Customer Churn Prediction"

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and set your SQL Server instance name:
```bash
cp .env.example .env
```
Inside `.env`:
```env
SQL_SERVER=YOUR_SQL_SERVER_NAME
SQL_DATABASE=customerChurnDB
SQL_DRIVER=ODBC Driver 18 for SQL Server
```

### 5. (Optional) Initialize SQL Server Database
Run [schema.sql](file:///d:/Data%20Science/Customer%20Churn%20Prediction/schema.sql) in SQL Server Management Studio (SSMS) or Azure Data Studio to create `customerChurnDB` and `dbo.CustomerChurn` table.

### 6. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`, or test the live cloud instances on **[Streamlit Community Cloud](https://customer-churn-prediction-churnshield.streamlit.app/)** or **[Render](https://churnshield-mwt7.onrender.com/)**.

---

## 🖥️ Application Interface Overview

- **Tab 1: Prediction Engine**:
  - Input custom customer parameters across 5 profile cards (Demographics, Tenure, Phone, Internet, Billing).
  - Select between **Logistic Regression** and **Random Forest**.
  - Receive an instant churn probability gauge, categorized risk tier (*Low*, *Medium*, *High Risk*), and an individual **SHAP Waterfall Explanation**.
- **Tab 2: Model Intelligence & Insights**:
  - Gini-based feature importance ranking.
  - Population-wide **SHAP Beeswarm Distribution** computed on representative customer samples.
  - Mean $|SHAP|$ global feature impact.
  - Head-to-head model performance metrics table and executive business strategies.

---

## 🛠️ Technology Stack

- **Modeling & Analytics**: `scikit-learn`, `joblib`, `shap`, `numpy`, `pandas`
- **Visualization**: `matplotlib`, `seaborn`
- **Application Delivery**: `streamlit`
- **Database Connectivity**: `SQLAlchemy`, `pyodbc`, `python-dotenv`
- **Database Engine**: Microsoft SQL Server (T-SQL)

---

## 📜 License
Distributed under the MIT License. Feel free to use and adapt this project for educational or commercial purposes.
