# B2B Lead Scoring (End‑to‑End, Reproducible)

A lightweight, reproducible pipeline to score B2B leads using scikit‑learn.
It covers data prep, feature engineering, modeling, evaluation (ROC‑AUC, PR‑AUC, Precision@K, calibration), and simple scoring buckets (A/B/C).

> **Latest offline result (holdout example):**  
> ROC‑AUC ≈ **0.686**, PR‑AUC ≈ **0.527** (baseline ≈ prevalence 0.477).

---

## TL;DR
- Input: CSV of company/owner attributes (US sample).  
- Output: probability to convert (`Converted_Prob`), 0–100 **LeadScore**, and **bucket** (A/B/C).  
- Stack: `pandas`, `scikit‑learn` (`Pipeline`, `ColumnTransformer`), optional `xgboost/lightgbm`.

---

## Data

Example files (place in `data/`):
- `b2b_leads_new.csv`

**Expected columns** (case‑sensitive):
```
Company, Website, Industry, ProductOrServiceCategory, BusinessModel,
EmployeesCount, RevenueUSD, YearFounded, BBBRating, Street, City, State,
CompanyPhone, CompanyLinkedin, OwnerName, OwnerTitle, OwnerLinkedin,
OwnerPhonesNumber, OwnerEmail, Converted
```
Notes:
- `Converted` is the target label (0/1).
- Missing values are common in contact fields (`OwnerPhonesNumber`, `OwnerEmail`) → we use **presence flags** instead of dropping rows.
- Heavy‑tailed numerics (`RevenueUSD`, `EmployeesCount`) are log‑transformed.

> ⚠️ **PII**: Treat phone/email responsibly. Do not publish raw PII in public repos.

---

## Project Structure
```
.
├─ Lead Scoring.ipynb          # Main notebook: EDA → features → train → evaluate → score
├─ data/
│  ├─ b2b_leads.csv            # Sample dataset (labeled)
│  └─ b2b_leads_new.csv        # (optional) Newer sample
├─ models/                      # (created at runtime) saved pipelines
└─ README.md
```

---

## Environment

```bash
python >= 3.10
pip install -U pandas numpy scikit-learn matplotlib seaborn joblib shap
```

---

## Quick Start (Notebook)

1) Put your CSV in `data/`.  
2) Open **Lead Scoring.ipynb** and run all cells:
   - **EDA**: missing values, distributions
   - **Features**: log transforms (`RevenueUSD_log`, `Employees_log`), `CompanyAge`,
     presence flags (`HasOwnerEmail`, `HasOwnerPhone`, `HasOwnerLinkedin`, `HasAnyContact`, `ContactCount`),
     BBB ordinal mapping, one‑hot for `Industry/BusinessModel/State`.
   - **Model**: `Pipeline(preprocess) + (LogReg | GradientBoosting | XGBoost/LGBM)`
   - **Evaluate**: ROC‑AUC, PR‑AUC, **Precision@K** (Top‑K), **Calibration** & **Brier score**
   - **Export**: persisted `joblib` pipeline + scored outputs

Example scoring snippet:
```python
import joblib, pandas as pd
import numpy as np
pipe = joblib.load("models/lead_scoring_pipeline_20250814-174851.joblib")     # saved full pipeline
df_in = pd.read_csv("b2b_leads_new.csv")

def clip_quantile(s, low=0.01, high=0.99):
    ql, qh = s.quantile(low), s.quantile(high)
    return s.clip(lower=ql, upper=qh)

df_in = df_in[(df_in["RevenueUSD"]>0) & (df_in["EmployeesCount"]>0)]
df_in["YearFounded"] = df_in["YearFounded"].clip(lower=1850, upper=2025)

df_in["RevenueUSD_log"]  = np.log1p(df_in["RevenueUSD"])
df_in["Employees_log"]   = np.log1p(df_in["EmployeesCount"])

df_in["RevenueUSD_log"]  = clip_quantile(df_in["RevenueUSD_log"], 0.005, 0.995)
df_in["Employees_log"]   = clip_quantile(df_in["Employees_log"],   0.005, 0.995)

df_in["CompanyAge"] = (2025 - df_in["YearFounded"]).clip(lower=0)
df_in["RevenueUSD_log"] = (df_in["RevenueUSD_log"] + 1).map(lambda x: np.log1p(x))
df_in["Employees_log"] = (df_in["Employees_log"] + 1).map(lambda x: np.log1p(x))
df_in["HasOwnerEmail"] = df_in["OwnerEmail"].notna().astype(int)
df_in["HasOwnerPhone"] = df_in["OwnerPhonesNumber"].notna().astype(int)
df_in["HasOwnerLinkedin"] = df_in["OwnerLinkedin"].notna().astype(int)

proba = pipe.predict_proba(df_in)[:,1]
df_in["Converted_Prob"] = proba
df_in["LeadScore"] = (proba*100).round().astype(int)
df_in["Bucket"] = pd.cut(df_in["LeadScore"], bins=[-1,60,80,100], labels=["C","B","A"])
df_in.to_csv("b2b_leads_scored.csv", index=False)
```

---

## Evaluation

We report:
- **ROC‑AUC** (ranking quality)
- **PR‑AUC** (important for class imbalance; baseline ≈ prevalence)
- 
---

## Feature Engineering (overview)

- **Numeric**:  
  `RevenueUSD_log = log1p(RevenueUSD)`, `Employees_log = log1p(EmployeesCount)`, `CompanyAge = 2025 - YearFounded`  
- **Categorical**: `Industry`, `BusinessModel (B2B/B2B2C)`, `State` → OneHotEncoder
- **Ordinal**: `BBBRating` → numeric mapping (A+…B…C)
- **Contactability** (high‑value signals):  
  `HasOwnerEmail`, `HasOwnerPhone`, `HasOwnerLinkedin`, `HasAnyContact` (any of the three), `ContactCount` (0–3)
- **Preprocessing**: `ColumnTransformer` with imputers & scalers; all wrapped in a single `Pipeline`

> **Ablation**: you can toggle `HasOwnerLinkedin` ↔ `HasOwnerPhone` or use both; keep the aggregate features (`HasAnyContact`, `ContactCount`) and pick what performs best in CV.

---

## Reproducibility
- Use `train_test_split(..., stratify=y, random_state=42)` or **time‑split** if timestamps exist.
- Cross‑validation: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Save full pipeline (`joblib`) to avoid training/serving skew.

---

## Deployment (optional)
- Export the pipeline and wrap a tiny **FastAPI** endpoint for batch/real‑time scoring.  
- Log input schema, monitor **data/label drift** and **metric decay**; schedule retraining (e.g., quarterly).

---

## FAQ

**Q: Do I need to remove outliers?**  
A: Usually **no**. Use `log1p` + optional winsorization/capping. Tree models are robust; dropping large companies may throw away useful signal.

---

## License
MIT (adjust as needed).

## Acknowledgements
- scikit‑learn team & docs
- The many blog posts and papers on lead scoring, calibration, and uplift analysis
