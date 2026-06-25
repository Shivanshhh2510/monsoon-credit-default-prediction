# Monsoon CreditTech — Credit Default Prediction

## Overview

This project predicts the probability of loan default at the time of application using historical credit bureau data.

Rather than focusing only on training different models, this project was approached as an end-to-end machine learning workflow. It involved understanding the data, engineering behavioural credit features, validating assumptions through EDA, experimenting with multiple models, and iteratively improving performance by removing noisy features instead of continuously adding new ones.

**Evaluation Metric**

ROC-AUC

---

## Dataset

The assessment consisted of three primary data sources:

- Previous credit accounts
- Credit enquiry history
- Loan contract information

No demographic, employment, or income information was available, so the entire model relies on historical credit behaviour.

---

## Project Workflow

1. Load and flatten nested JSON bureau data
2. Engineer account-level behavioural features
3. Engineer enquiry-level behavioural features
4. Perform exploratory data analysis
5. Train Logistic Regression baseline
6. Train XGBoost and CatBoost models
7. Analyze feature importance and model behaviour
8. Remove weak and noisy features
9. Retrain models on the pruned feature set
10. Perform Stratified 5-Fold Cross Validation
11. Retrain the best-performing model on the complete training dataset
12. Generate final submission

---

# Model Development Journey

One of the most interesting parts of this assignment was that the first solution was **not** the best solution.

Initially, I engineered a large number of features from both account history and enquiry history. This included payment history statistics, DPD-based features, enquiry counts, loan behaviour metrics, ratios, interaction features, and several handcrafted behavioural signals. While this produced a comprehensive feature set, the validation ROC-AUC quickly plateaued.

Instead of continuing to add more features, I stepped back and investigated what the models were actually learning.

I repeatedly inspected feature importance, analysed individual feature behaviour, validated assumptions through EDA, and compared model performance after every significant change.

This iterative process revealed something unexpected:

> **Using fewer but more informative features consistently produced better validation performance than using every engineered feature.**

Several payment-history features that initially appeared important contributed very little predictive power, whereas behavioural signals such as enquiry recency, loan closure behaviour, current overdue information, and credit portfolio characteristics proved much more informative.

After multiple rounds of experimentation, feature pruning, retraining, and validation, the final feature set achieved better generalisation than the original larger feature set.

This project reinforced an important lesson:

**Better features are often more valuable than more features.**

---

## Feature Engineering

The final feature set includes behavioural credit features such as:

- Total loan count
- Active vs closed loans
- Closed loan ratio
- Current overdue exposure
- Credit type diversity
- Enquiry recency
- Enquiry frequency
- Behavioural ratios
- Credit portfolio statistics
- Log-transformed monetary features

Several additional engineered features were explored during experimentation but were removed after validation because they added noise without improving model performance.

---

## Models Evaluated

### Logistic Regression

- Baseline model
- Used to establish an initial benchmark

### XGBoost

- Early stopping
- Class imbalance handling
- Hyperparameter tuning
- Stratified 5-fold Cross Validation

### CatBoost

- Ordered boosting
- Native categorical handling
- Early stopping
- Stratified 5-fold Cross Validation
- Final selected model

---

## Validation Strategy

Performance was evaluated using both a train-validation split and Stratified 5-Fold Cross Validation.

The cross-validation results provided a much more reliable estimate of generalisation performance and were used to compare the final candidate models before retraining on the complete training dataset.

---

## Results

| Model | ROC-AUC |
|--------|--------:|
| Logistic Regression (Baseline) | ~0.67 |
| XGBoost | ~0.68 |
| CatBoost | ~0.68 |

CatBoost produced the most consistent cross-validation performance and was selected as the final model for submission.

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

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- CatBoost
- Matplotlib

---

## Key Learnings

- Behavioural credit signals were significantly more predictive than many historical payment-history statistics.
- More engineered features did not necessarily improve model performance.
- Iterative feature pruning improved validation ROC-AUC by reducing noise and overfitting.
- Cross-validation provided a more reliable estimate of model performance than relying on a single validation split.
- Careful feature analysis and model diagnostics proved more valuable than simply increasing model complexity.

---

## Author

**Shivansh Mishra**

GitHub:  
https://github.com/Shivanshhh2510
