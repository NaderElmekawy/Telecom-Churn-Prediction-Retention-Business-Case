# Telecom Churn Prediction & Retention Business Case

A machine learning project that predicts telecom customer churn and identifies high-risk customers to support data-driven retention strategies.

## Project Overview

The goal is to help a telecom company prioritize customer retention campaigns by identifying subscribers with a high probability of churning within 31–60 days.

The project includes data analysis, machine learning, model evaluation, and an illustrative business case.

## Dataset

- `Client.csv` — Customer attributes (100,000 rows).
- `Record.zip` — Usage, billing, service records, and churn target.

The datasets are merged using `Customer_ID`, resulting in **100,000 customers and 100 features**.

## Methodology

- Exploratory Data Analysis (EDA)
- Data preprocessing and feature selection
- Logistic Regression and Histogram Gradient Boosting
- Model selection using validation data
- Held-out test evaluation
- Permutation feature importance
- Retention campaign and financial sensitivity analysis

## Results

**Selected Model: Histogram Gradient Boosting**

| Metric | Test Result |
|---|---:|
| ROC-AUC | 0.6742 |
| Average Precision | 0.6604 |
| Top-20% Precision | 71.00% |
| Top-20% Recall | 28.65% |

The model helps prioritize high-risk customers for retention outreach. The proposed financial results are illustrative scenarios, not guaranteed outcomes.

## Tech Stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Google Colab

## Repository Structure

```text
.
├── telecom-churn-prediction.ipynb
├── telecom-churn-retention-presentation.pdf
├── Client.csv
├── Record.zip
└── README.md
```

## How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/NaderElmekawy/Telecom-Churn-Prediction-Retention-Business-Case.git
   ```

2. Extract `Record.zip` so that `Record.csv` is available.

3. Install the required libraries:

   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```

4. Open `telecom-churn-prediction.ipynb` in Google Colab or Jupyter.

5. Update the data-loading path to your local dataset folder and run the notebook.

---

**Author:** [Nader Elmekawy](https://github.com/NaderElmekawy)
