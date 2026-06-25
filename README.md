# Monsoon CreditTech — Credit Default Prediction

## Overview

This project predicts the probability of loan default at the time of application using historical credit bureau data.

The solution combines feature engineering on bureau accounts and enquiry history with gradient boosting models and cross-validation to build a robust binary classification pipeline.

**Evaluation Metric**

ROC-AUC

---

## Dataset

The assessment consists of three primary data sources:

- Previous credit accounts
- Credit enquiry history
- Loan contract information

No demographic or income information was provided.

---

## Pipeline

1. Load and flatten raw JSON bureau data
2. Feature engineering from account history
3. Feature engineering from enquiry history
4. Exploratory data analysis
5. Logistic Regression baseline
6. Feature selection using Mutual Information
7. XGBoost modelling
8. CatBoost modelling
9. Feature pruning
10. Stratified 5-Fold Cross Validation
11. Final model retrained on the complete training data
12. Submission generation

---

## Feature Engineering

Examples of engineered features include:

- Loan counts
- Active vs closed loans
- Closed loan ratio
- Credit utilisation patterns
- Current overdue amounts
- Enquiry recency
- Enquiry frequency
- Credit type diversity
- Behavioural ratios
- Log-transformed monetary features

---

## Models

### Logistic Regression

Used as the baseline model.

### XGBoost

- Early stopping
- Class imbalance handling
- Hyperparameter tuning

### CatBoost

- Ordered boosting
- Native categorical handling
- Early stopping
- Final selected model

---

## Validation

Stratified 5-Fold Cross Validation was performed to obtain a more reliable estimate of model performance than a single train-validation split.

---

## Results

| Model | ROC-AUC |
|--------|--------:|
| Logistic Regression | ~0.67 |
| XGBoost | ~0.68 |
| CatBoost | ~0.68 |

CatBoost achieved the best cross-validation performance and was selected as the final model.

---

## Project Structure

```
monsoon-credit-default-prediction/
│
├── notebooks/
│   └── Shivansh_Mishra_MonsoonCreditTech_Assessment_Final.ipynb
│
├── submissions/
│   └── final_submission_shivansh_mishra_monsooncredittech.csv
│
├── README.md
├── LICENSE
└── .gitignore
```

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- CatBoost
- Matplotlib

---

## Key Learnings

- Behavioural credit signals were more predictive than historical DPD values.
- Feature pruning improved generalization.
- Cross-validation provided a more reliable estimate of model performance.
- CatBoost achieved the best overall validation performance.

---

## Author

**Shivansh Mishra**

GitHub:

https://github.com/Shivanshhh2510
