# Fraud Detection Project

## Project Overview
This project focuses on detecting fraudulent activities in **e-commerce transactions** and **bank credit transactions**. The main goal is to build accurate and explainable machine learning models that identify fraud while balancing security and user experience.

---

## Project Structure

fraud-detection/
├── data/raw/ # Original datasets (Fraud_Data.csv, creditcard.csv)
├── data/processed/ # Cleaned and feature-engineered data
├── notebooks/ # Jupyter notebooks for analysis
├── src/ # Reusable scripts and functions
│ ├── preprocessing.py
│ └── eda.py
├── tests/ # Unit tests
├── models/ # Saved model artifacts
├── scripts/ # Helper scripts
├── requirements.txt # Python dependencies
├── README.md
└── .gitignore

yaml
Copy code

---

## Datasets

### Fraud_Data.csv
- E-commerce transaction data  
- Columns: `user_id`, `signup_time`, `purchase_time`, `purchase_value`, `device_id`, `source`, `browser`, `sex`, `age`, `ip_address`, `class`  
- Target: `class` (1 = fraudulent, 0 = legitimate)

### creditcard.csv
- Bank credit card transactions  
- Features: `Time`, `V1`–`V28` (PCA anonymized), `Amount`  
- Target: `Class` (1 = fraudulent, 0 = legitimate)

---

## Data Preprocessing & Feature Engineering

- **Missing Values:** Checked and imputed/dropped as needed  
- **Duplicates:** Removed  
- **Feature Engineering:**
  - `time_since_signup` (Fraud_Data)  
  - Transaction velocity and frequency  
  - `hour_of_day`, `day_of_week` features  
  - IP-to-country mapping  
- **Categorical Encoding:** One-hot for browser, source, sex, country  
- **Scaling:** StandardScaler for numeric features  
- **Class Imbalance:** Handled using SMOTE; visualized before and after

---

## Exploratory Data Analysis (EDA)

- **Visualizations included:**
  - Class distribution before and after SMOTE  
  - Fraud rate by country  
  - Purchase amount distribution  
  - Transaction amount distribution (credit card dataset)  

- **Key Insights:**
  - Fraud is concentrated in specific countries and high-value transactions  
  - Imbalanced datasets require careful model evaluation (F1, AUC-PR, recall)

---

## Reusable Code

- Functions for cleaning, scaling, and feature separation are in `src/preprocessing.py`  
- EDA visualizations are in `src/eda.py`  
- Error handling ensures robustness and reproducibility

---

## Next Steps / Modeling Plan

- **Model candidates:** Logistic Regression, Random Forest, XGBoost/LightGBM  
- **Evaluation metrics:** F1-score, Recall, AUC-PR  
- **Hyperparameter tuning:** GridSearchCV / RandomizedSearchCV  
- **Challenges & Mitigations:**
  - Extreme class imbalance → mitigated with SMOTE  
  - Overfitting → mitigated with cross-validation and feature selection  
  - Real-time prediction requirements → optimize model inference time

---

## Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/blienmoges/kifiya-week5.git
cd fraud-detection