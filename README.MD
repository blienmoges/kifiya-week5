# Fraud Detection Project

## Project Overview
This project detects fraudulent activity in:
1) E-commerce transactions (`Fraud_Data.csv`)
2) Bank credit card transactions (`creditcard.csv`)

The goal is to build accurate and explainable ML models that detect fraud while balancing security and user experience.

---

## Project Structure
fraud-detection/
├── data/
│   ├── raw/              # Original datasets (Fraud_Data.csv, creditcard.csv)
│   └── processed/        # Cleaned and feature-engineered data
├── notebooks/            # Jupyter notebooks for analysis and experiments
├── src/
│   ├── preprocessing.py  # Cleaning / scaling / utilities
│   └── eda.py            # EDA visualizations
├── scripts/              # Train/evaluate helper scripts
├── tests/                # Unit tests
├── models/               # Saved model artifacts
├── reports/              # CV summaries, plots, tables (generated)
├── requirements.txt
├── README.md
└── .gitignore

---

## Datasets

### 1) Fraud_Data.csv (E-commerce)
Columns: `user_id, signup_time, purchase_time, purchase_value, device_id, source, browser, sex, age, ip_address, class`  
Target: `class` (1 = fraud, 0 = legitimate)

### 2) creditcard.csv (Bank credit card)
Features: `Time, V1–V28, Amount`  
Target: `Class` (1 = fraud, 0 = legitimate)

---

## Data Preprocessing & Feature Engineering
- Missing values: handled during cleaning (median for numeric; "missing" for categorical)
- Duplicates: removed
- Feature engineering (Fraud_Data):
  - `time_since_signup`
  - transaction velocity/frequency features
  - `hour_of_day`, `day_of_week`
  - IP-to-country mapping
- Encoding: One-hot encoding for `browser, source, sex, country`
- Scaling: StandardScaler for numeric features

### Class Imbalance Handling
- SMOTE is used **within cross-validation folds** to avoid data leakage.
- Metrics emphasize imbalanced learning: **AUC-PR (primary), Recall, F1**.

---

## Exploratory Data Analysis (EDA)
Included visualizations:
- Class distribution (before/after SMOTE)
- Fraud rate by country
- Purchase value distribution
- Credit card transaction amount distribution

Key insights:
- Fraud tends to concentrate in specific countries and higher-value transactions.
- Imbalanced datasets require evaluation using AUC-PR, recall, and F1 (accuracy is misleading).

---

## Modeling
Candidate models:
- Logistic Regression (baseline, interpretable)
- Random Forest
- XGBoost / LightGBM

Evaluation metrics:
- **AUC-PR (primary)**
- Recall
- F1-score

### Cross-Validation Reporting (Required Outputs)
We report:
- CV mean ± std for AUC-PR / Recall / F1
- Fold-by-fold scores
- Optional: out-of-fold (OOF) predictions for reproducibility

Generated artifacts (in `reports/`):
- `reports/cv_summary.csv`
- `reports/cv_folds.csv`

---

## Model Comparison
A single leaderboard table summarizes baseline, tuned, and ensemble models:

| Model | Resampling | CV AUC-PR (mean±std) | CV Recall (mean±std) | CV F1 (mean±std) | Notes |
|------|------------|----------------------|----------------------|------------------|------|
| Logistic Regression | class_weight / SMOTE | TBD | TBD | TBD | baseline |
| Random Forest | class_weight / SMOTE | TBD | TBD | TBD | tuned |
| XGBoost/LightGBM | scale_pos_weight / SMOTE | TBD | TBD | TBD | best |
| Ensemble | - | TBD | TBD | TBD | optional |

---

## Model Selection Documentation
Final model choice and rationale are documented in:
- `docs/model_selection.md` (candidates, tuning approach, rejection reasons, final decision)

---

## Setup Instructions

### 1) Clone
```bash
git clone https://github.com/blienmoges/kifiya-week5.git
cd fraud-detection
