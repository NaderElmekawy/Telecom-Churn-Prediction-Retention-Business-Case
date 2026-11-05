# Telecom Churn Prediction & Retention Business Case

A machine learning notebook that predicts which telecom subscribers are likely to churn 31-60 days after the observation date, and turns those predictions into a retention-campaign business case with financial sensitivity analysis.

**Author:** Nader Elmekawy

## Business Question

Can "Company A" reduce customer losses by prioritizing outreach to subscribers with a high predicted probability of churning? Published industry work (McKinsey, 2017) describes churn reductions of up to 15% in selected telecom implementations. That is **not** a guaranteed result for Company A, so this project treats the benefit as an assumption to be tested in a pilot.

## Dataset

- `Client.csv` (100,000 rows x 50 columns): customer and demographic attributes
- `Record.csv` (100,000 rows x 51 columns): usage, billing and service records, plus the `churn` target
- Merged on `Customer_ID` (one-to-one): 100,000 unique customers, 100 columns
- Churn rate: **49.56%** (balanced target)

Both data files are included in this repository:

- `Client.csv` is stored as a regular CSV.
- `Record.zip` is a compressed copy of `Record.csv`. **Unzip it before running the notebook**, because the code reads `Record.csv`.

## Approach

1. **EDA:** missing-value review, target balance, and churn rate by quintile of customer-care calls, handset age (`eqpdays`) and revenue (`rev_Mean`). These are associations, not causal effects.
2. **Feature selection:** 26 operational, billing and usage signals available at observation time. ID, target and sensitive demographic proxy variables are excluded.
3. **Modeling:** 75/25 stratified train/test split (seed 42). Median imputation runs inside each pipeline. Two models are compared:
   - Logistic regression (interpretable baseline)
   - Histogram gradient boosting (nonlinear)
4. **Evaluation:** ROC-AUC (primary), average precision, and precision/recall for the top 20% highest-risk accounts. Permutation importance is computed on a held-out subset.
5. **Business case:** contact the top 20% by risk, with a randomized no-contact control group to measure incremental saves, plus a financial sensitivity analysis.

## Results (test set)

| Model | ROC-AUC | Avg. precision | Top-20% precision | Top-20% recall |
|---|---|---|---|---|
| Logistic regression | 0.6028 | 0.5780 | 0.6104 | 0.2463 |
| **Gradient boosting** | **0.6742** | **0.6604** | **0.7100** | **0.2865** |

Gradient boosting is selected. The signal is moderate: the model is useful for prioritizing outreach, not for confident individual predictions.

### Illustrative financial sensitivity

Assumptions: 3 months of preserved revenue, 50% contribution margin, 2 monetary units per contact, 15 units per saved account. Currency is unspecified in the source data, so all figures are in **dataset monetary units**.

| Incremental save rate | Saved accounts | Net contribution |
|---|---|---|
| 5% | 710 | 11,887 |
| 10% (base) | 1,420 | 63,774 |
| 15% | 2,130 | 115,661 |

These figures are scenario assumptions. Predictions do not demonstrate treatment effectiveness.

## Proposed Pilot & Guardrails

- Run a 6-8 week pilot: high-risk customers randomized to outreach vs. business-as-usual, measuring 31-60 day churn and net incremental contribution.
- Tailor actions to observed problems (service troubleshooting, billing clarification, value-plan support) without claiming a feature *causes* churn.
- Privacy: avoid sensitive demographic columns, restrict access, and check performance across approved groups.
- The random split is only a feasibility check. A time-based validation is required before deployment, along with drift monitoring and recalibration.

## Tech Stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn. Developed in Google Colab.

## How to Run

1. Clone or download this repository:
   ```bash
   git clone https://github.com/USERNAME/REPO.git
   cd REPO
   ```
2. Unzip the record data so that `Record.csv` sits next to `Client.csv`:
   ```bash
   unzip Record.zip
   ```
3. Open `NaderElmekawy.ipynb` in Google Colab or Jupyter.
4. Update the data paths in the first code cell. The notebook currently mounts Google Drive and uses a course-specific folder path. Replace it with the location of `Client.csv` and `Record.csv`, for example:
   ```python
   client = pd.read_csv("Client.csv")
   record = pd.read_csv("Record.csv")
   ```
5. Run all cells.

**Dependencies:** `pip install numpy pandas matplotlib seaborn scikit-learn`

## Repository Structure

```
.
├── NaderElmekawy.ipynb   # Full analysis and business case
├── Client.csv            # Customer and demographic attributes
├── Record.zip            # Zipped Record.csv (usage, billing, churn target)
└── README.md
```
