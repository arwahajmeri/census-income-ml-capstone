# Census Income Prediction — ML Capstone

A supervised learning capstone comparing a traditional machine learning model (K-Nearest
Neighbors) against a feedforward neural network on the task of predicting whether an
individual's annual income exceeds $50,000, using 1994 U.S. Census data.

## Problem

**Dataset:** [Census Income Data](https://archive.ics.uci.edu/dataset/20/census+income) (`censusData.csv`) — demographic and employment information from the 1994 U.S. Census.

**Label:** `income_binary` — whether an individual's income is `<=50K` or `>50K`, converted to a binary numeric label (0 / 1).

**Features used:** age, workclass, education-num, marital-status, occupation, relationship, race, sex_selfID, capital-gain, capital-loss, hours-per-week, native-country (one-hot encoded where categorical).

## Approach

1. **EDA** — examined class distribution (~76% `<=50K` / ~24% `>50K`), missing values (encoded as `' ?'` in the raw data), and relationships between features and the label.
2. **Data preparation** — cleaned missing values, dropped non-predictive columns (`fnlwgt`, redundant `education` string column), one-hot encoded categorical features, used a stratified train/test split to preserve class balance.
3. **Model 1 — K-Nearest Neighbors** — scaled features with `StandardScaler`, tuned `k` via grid search cross-validation (`GridSearchCV`, scoring by F1).
4. **Model 2 — Neural Network** — a Keras `Sequential` feedforward network (two hidden layers, ReLU activations, sigmoid output), trained with SGD and binary cross-entropy loss.
5. **Evaluation** — both models evaluated on accuracy and F1 score on a held-out test set; results compared side by side.

## Results

| Metric   | KNN Model | Neural Network |
|----------|-----------|----------------|
| Accuracy | 0.837     | 0.829          |
| F1 Score | 0.632     | 0.647          |

KNN achieved slightly higher accuracy, while the neural network achieved a slightly higher F1
score. Given the modest performance gap and signs of overfitting observed in the neural
network's training curves, KNN was the recommended model for this use case, prioritizing
interpretability and training efficiency. Full reasoning is documented in the notebook.

## Repository Contents

- `Capstone.ipynb` — full notebook: EDA, data preparation, model training/tuning, evaluation, and written reflections.
- `censusData.csv` — dataset used for training and evaluation.
- `requirements.txt` — Python packages needed to run the notebook.

## Running Locally

```bash
pip install -r requirements.txt
jupyter notebook Capstone.ipynb
```

## Notes

This project was completed as part of a machine learning course capstone. AI tools were used
during parts of the development process (debugging, drafting explanations, and reviewing
reasoning); see the AI Use Attestation section at the end of the notebook for details.
