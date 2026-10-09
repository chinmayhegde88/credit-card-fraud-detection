# Credit Card Fraud Detection: Cost-Sensitive Modelling

Fraud detection on highly imbalanced card transactions, using XGBoost with class weighting and a decision threshold chosen by expected cost.

## Problem
Card fraud causes direct financial loss, but declining legitimate customers also costs money and trust. Fraud is rare (about 0.17% of transactions here), so a model can reach very high accuracy while catching almost no fraud. This project builds fraud detection models and chooses the decision threshold by expected cost.

## Data
- Kaggle Credit Card Fraud Detection dataset (Worldline and ULB Machine Learning Group): https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- European cardholders, two days of transactions. After removing exact duplicates: 283,726 rows, 473 frauds.
- Features: anonymized PCA components V1-V28, plus `Time` and `Amount`. Label: `Class` (1 = fraud).
- The CSV is not included in this repo. Download it from Kaggle and place it next to the notebook.

## Approach
1. Data checks: duplicates, missing values, outliers, fraud patterns by time and amount.
2. Features: `Hour` and `Log_Amount` added. Stratified 60/20/20 train/validation/test split. Scaling fitted on training data only.
3. Models (all class-weighted): Logistic Regression baseline, Random Forest, XGBoost (`scale_pos_weight`).
4. Model comparison on the validation set: PR-AUC, ROC-AUC, precision, recall, F1, KS, Gini.
5. Threshold selection on validation: F1-optimal threshold as a reference, and a cost-based threshold where a missed fraud costs its transaction `Amount` and a false alert costs an assumed 5 units.
6. One final evaluation on the untouched test set, comparing three threshold rules against a no-model baseline.

## Results

**Validation (model comparison)**

| Model | Precision | Recall | PR-AUC | KS | Gini |
|---|---|---|---|---|---|
| Logistic Regression | 0.056 | 0.904 | 0.788 | 0.893 | 0.965 |
| Random Forest | 0.888 | 0.840 | 0.856 | 0.892 | 0.920 |
| XGBoost | 0.941 | 0.851 | 0.885 | 0.901 | 0.951 |

**Test set (XGBoost, 95 frauds)**

| Threshold rule | Threshold | Precision | Recall | F1 | Total cost |
|---|---|---|---|---|---|
| Default | 0.50 | 0.961 | 0.779 | 0.861 | 3,807.56 |
| F1-optimal | 0.98 | 0.986 | 0.726 | 0.836 | 4,069.52 |
| Cost-based | 0.46 | 0.949 | 0.779 | 0.856 | 3,812.56 |
| No model | - | - | 0 | - | 14,766.31 |

Key findings:
- XGBoost was the best model on validation (PR-AUC 0.885).
- At the cost-based threshold, estimated fraud loss was about 74% lower than with no model.
- The cost-based threshold cut loss about 6% versus the F1-optimal threshold, which pushed the threshold too high and missed more fraud.
- The default 0.5 threshold performed almost the same as the cost-based one (a difference of one false alert), so that gap is within noise.
- The best cost-based threshold stayed at 0.46 for false-alert costs of 5, 20 and 50.

## Limitations
- Features are anonymized PCA components, so the model's decisions can't be explained in business terms.
- Two days of data and a random split: drift and seasonality can't be assessed, and real-world performance may be lower.
- No card or customer ID, so each transaction is modelled on its own.
- Costs are assumptions: a false alert is set at 5 units, so loss reductions are indicative only.
- Only 95 frauds in the test set, so small differences between thresholds are within noise.

## How to run
1. Download `creditcard.csv` from the Kaggle link above.
2. Install: `pip install -r requirements.txt`
3. Open `Credit_Card_Fraud_Detection.ipynb` and run all cells (update the data path in the first cells if needed).

## Tools
Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn.

## Acknowledgements
- Dataset: Worldline and the Machine Learning Group of Université Libre de Bruxelles (ULB), via Kaggle.
- Reviewed a public Kaggle notebook on this dataset for background.
## Author
[Chinmay Hegde] · [www.linkedin.com/in/chinmaych] · [chinmayhegde88@gmail.com]
