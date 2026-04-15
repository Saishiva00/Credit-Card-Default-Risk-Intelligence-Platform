# 💳 Credit Card Default Risk Intelligence Platform
### Exploratory Data Analysis | UCI Credit Card Dataset | 30,000 Customers | 25 Features

> **"Identifying the behavioral fingerprints of financial default — 3 to 4 months before it happens."**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4c72b0)](https://seaborn.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()
[![Dataset](https://img.shields.io/badge/Dataset-UCI_ML_Repo-orange)](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

---

## 📋 Table of Contents
1. [Problem Statement](#problem-statement)
2. [Project Objectives](#objectives)
3. [Dataset Description](#dataset-description)
4. [Tech Stack](#tech-stack)
5. [Architecture / Workflow](#architecture--workflow)
6. [Key Features](#key-features)
7. [EDA Insights](#eda-insights)
8. [Key Business Insights](#key-business-insights)
9. [Visualizations](#visualizations)
10. [Results & Findings](#results--findings)
11. [Future Improvements](#future-improvements)
12. [How to Run](#how-to-run)
13. [Folder Structure](#folder-structure)

---

## 🎯 Problem Statement

Credit card defaults cost the banking industry billions annually. Traditional rule-based systems flag customers **after** default occurs — too late for meaningful intervention.

This project performs a deep exploratory analysis on **30,000 Taiwan credit card customers** to answer:

- **What behavioral patterns emerge 3–6 months before a customer defaults?**
- **Which features are the strongest predictors of default?**
- **Which customer segments carry disproportionate risk?**
- **Can we build actionable early-warning signals for the credit risk team?**

The analysis is aligned with **CIBIL/TransUnion credit bureau scoring methodology** — a real-world framework used by financial institutions globally.

---

## 🎯 Objectives

- Perform a complete end-to-end EDA on 30,000 credit card customer records
- Identify the top behavioral, financial, and demographic risk signals
- Engineer credit utilization ratio (a key CIBIL scoring component) from raw features
- Quantify default rate across all customer segments and payment behaviors
- Deliver data-driven recommendations for a Credit Risk operations team
- Lay the analytical foundation for future machine learning model development

---

## 📦 Dataset Description

| Attribute | Details |
|---|---|
| **Source** | UCI Machine Learning Repository |
| **Dataset Name** | Default of Credit Card Clients |
| **Geography** | Taiwan |
| **Records** | 30,000 customers |
| **Features** | 25 (demographic, behavioral, financial) |
| **Target Variable** | `default` — 1 = defaulted next month, 0 = did not |
| **Time Period** | April 2005 – September 2005 (6 months of payment history) |
| **Currency** | NT$ (New Taiwan Dollar) |

### Feature Groups

| Group | Features | Description |
|---|---|---|
| **Demographics** | SEX, EDUCATION, MARRIAGE, AGE | Customer profile |
| **Credit Info** | LIMIT_BAL | Approved credit limit in NT$ |
| **Payment Status** | PAY_0 to PAY_6 | Monthly repayment delay (-2 to 8 months) |
| **Bill Amounts** | BILL_AMT1 to BILL_AMT6 | Statement amounts (Sep–Apr) |
| **Payment Amounts** | PAY_AMT1 to PAY_AMT6 | Actual payments made (Sep–Apr) |
| **Target** | default | Binary: defaulted next month? |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Feature Engineering** | Pandas (binning, ratio computation) |
| **Statistical Analysis** | Pearson Correlation, Pivot Tables |
| **Environment** | Jupyter Notebook |
| **Version Control** | Git, GitHub |

---

## 🏗️ Architecture / Workflow

```
Raw CSV Data (30,000 rows × 25 cols)
        │
        ▼
┌─────────────────────────────────────┐
│  SECTION 1: Data Loading            │
│  • Load UCI CSV                     │
│  • Rename target column             │
│  • First look: shape, dtypes        │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  SECTION 2: Data Cleaning           │
│  • Null audit → 0 missing values    │
│  • Duplicate check → 0 duplicates   │
│  • Fix invalid EDUCATION values     │
│  • Fix invalid MARRIAGE values      │
│  • Drop ID column (noise)           │
│  • Create readable label columns    │
│  • Engineer AGE_group feature       │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  SECTION 3: Univariate Analysis     │
│  • Target distribution (22.1% def.) │
│  • Credit limit distribution        │
│  • Age distribution by group        │
│  • Gender, Education, Marriage      │
│  • Payment status (PAY_0) dist.     │
│  • Bill amount trends (6 months)    │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  SECTION 4: Bivariate Analysis      │
│  • Default rate by Age group        │
│  • Default rate by Education        │
│  • Credit limit vs default status   │
│  • Payment delay vs default rate    │
│  • Bill/Payment trends by group     │
│  • Default by gender & marriage     │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  SECTION 5: Multivariate Analysis   │
│  • Full correlation heatmap         │
│  • Feature correlation ranking      │
│  • Pivot heatmaps (Age×Edu, Age×Mar)│
│  • Credit utilization engineering   │
│  • 3-variable scatter plots         │
│  • Payment history pattern heatmap  │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  SECTION 6: Business Insights       │
│  • Risk factor ranking chart        │
│  • Early warning system design      │
│  • Actionable recommendations       │
│  • ML modeling roadmap              │
└─────────────────────────────────────┘
```

---

## ✨ Key Features

- **19 production-quality visualizations** covering univariate, bivariate, and multivariate analysis
- **Feature engineering** — computed Credit Utilization Ratio from raw bill and limit data
- **CIBIL-aligned analysis** — findings mapped to TransUnion credit bureau methodology
- **Segment-level risk quantification** — default rates across all demographic and behavioral groups
- **Anomaly detection** — identified and handled undocumented EDUCATION and MARRIAGE values (0, 5, 6)
- **6-month temporal analysis** — tracked payment and billing behavior over time
- **Business recommendations** — actionable early warning system design for credit risk teams

---

## 🔍 EDA Insights

### Univariate
- **22.1% overall default rate** (6,636 defaulters vs 23,364 non-defaulters) — class imbalance confirmed
- Credit limit distribution is **right-skewed**: mean NT$167K >> median NT$140K; high-value outliers at NT$1M (VIP accounts)
- Customer age is **right-skewed**: mean 35.5 years, median 34 — majority in their 20s–30s
- **60.4% female** customers; 46.8% university-educated; 53.2% single
- Most customers paid on time in September (PAY_0 = 0); only ~22% were 1+ month late

### Bivariate
- Customers **2+ months late** had 60–78% default rate vs 22.1% average — 3× the baseline
- **Lower credit limits** strongly associated with default (median NT$90K defaulters vs NT$150K non-defaulters)
- Customers aged **60+** show the highest default rate (28.3%); customers in their 30s show the lowest (20.3%)
- **High school educated** customers default more (25.2%) vs graduate school (19.2%)
- **Male customers** default at 24.2% vs 20.8% for female customers

### Multivariate
- PAY columns show **high inter-correlation (0.57–0.82)** — chronic lateness is persistent, not episodic
- BILL_AMT columns show near-perfect correlation (0.83–0.95) — spending is stable over time
- **Young + High school + Single** customers can show default rates above 30%
- Customers in the **60–90% utilization band** default at 27.6%, nearly double the 16.9% rate for the 0–30% band
- **Defaulters consistently make lower payments** across all 6 months — behavioral stress builds up months before default

---

## 📊 Key Business Insights

### 🔴 Insight 1 — Payment History is the #1 Risk Signal
Customers who were 2+ months late in September showed a **~70% default probability** — more than **3× the 22.1% baseline**. This single feature (PAY_0) carries correlation coefficient **+0.325** with default. **Action:** Flag any customer with PAY_0 ≥ 2 as HIGH RISK and trigger immediate collection outreach.

### 🔴 Insight 2 — Chronic Lateness Is More Dangerous Than One-Time Delay
The correlation between PAY_0 and PAY_2 through PAY_6 ranges from **0.57 to 0.67**. Customers who miss payments in multiple consecutive months show **70%+ default rates consistently** across all 6 months. **Action:** Build a "consecutive delay counter" as a behavioral risk feature — escalate after 2 consecutive late payments.

### 🟡 Insight 3 — Credit Limit Is Both Cause and Effect of Default Risk
Non-defaulters had a median credit limit of **NT$150K vs NT$90K** for defaulters. Banks assign lower limits to risky profiles — but low limits also increase utilization ratios, which further elevates risk. **Action:** Credit limit decisions must factor in full behavioral profile, not just income alone.

### 🟡 Insight 4 — Credit Utilization Above 60% Is a Red Flag
Customers using **60–90% of their credit limit** defaulted at **27.6%** — well above the 22.1% average. Above 90%, default rate stabilizes around **26–27%** but remains elevated. **Action:** Implement automated alerts when customer utilization crosses the 60% threshold.

### 🟡 Insight 5 — Bill-to-Payment Gap Reveals Financial Stress 3–4 Months Early
Defaulters carried **average bill amounts of NT$45–49K** against **payments of only NT$3.3–3.4K** — a structural gap. Non-defaulters paid **NT$5.3–6.5K** on similar balances. **Action:** Monitor the bill-to-payment ratio monthly; a widening gap signals emerging financial stress before default triggers.

### 🟢 Insight 6 — Education Level Is a Moderate but Consistent Risk Differentiator
Graduate school customers default at **19.2%** vs 25.2% for high school level — a **6 percentage point gap**. Education acts as a proxy for income stability. **Action:** Use education tier as a secondary segmentation variable, not a primary scoring input.

### 🟢 Insight 7 — Younger Customers (20s) Carry Above-Average Risk
Despite having the smallest credit limits, customers in their **20s default at 22.8%** — above the overall average. Short credit histories and lower income stability explain the pattern. **Action:** Apply stricter initial credit limits and more frequent monitoring for under-30 new customers.

### 🟢 Insight 8 — Risk Compounds Across Demographics
Multivariate heatmap reveals that **60+ age + University education** segments show **35% default rates** — far exceeding any single-variable analysis. **Action:** Use multi-dimensional risk segmentation (age × education × marital status) for credit policy design, not single-variable cutoffs.

---

## 📸 Visualizations

| # | Chart | Key Takeaway |
|---|---|---|
| 01 | Target Distribution (Pie + Bar) | 22.1% default rate — class imbalance confirmed |
| 02 | Age Distribution + Group Counts | Right-skewed; most customers in 20s–30s |
| 03 | Credit Limit Distribution + Box Plot | Right-skewed; outliers at NT$1M |
| 04 | Demographic Profile (Gender, Education, Marriage) | 60% female, 47% university-educated |
| 05 | Payment Status in September (PAY_0) | Majority on-time; ~22% late |
| 06 | Bill Amounts Over 6 Months | Stable, right-skewed; medians NT$17–22K |
| 07 | Age vs Default Risk | 60+ has highest rate (28.3%); 30s lowest (20.3%) |
| 08 | Education vs Default Risk | Graduate 19.2% vs High school 25.2% |
| 09 | Credit Limit vs Default (Box + Violin) | NT$90K median vs NT$150K for non-defaulters |
| 10 | Payment History vs Default Rate | 2+ months late → 70%+ default probability |
| 11 | Bill & Payment Trends — 6 Months | Defaulters spend more, pay less — compounding stress |
| 12 | Demographics vs Default Rate | Males (24.2%) and married (23.5%) above average |
| 13 | Full Correlation Heatmap | PAY columns cluster; BILL_AMTs near-perfectly correlated |
| 14 | Feature Correlation with Default (Ranked) | PAY_0 (+0.325) top; LIMIT_BAL (-0.154) protective |
| 15 | Pivot Heatmaps (Age × Education, Age × Marital) | Risk compounds across demographic combinations |
| 16 | Credit Utilization vs Default | Below 30% → 16.9%; above 60% → 27%+ |
| 17 | 3-Variable Scatter (Age, Limit, Utilization) | Defaulters cluster at low limit + high utilization |
| 18 | Payment History Heatmap (6 Months) | Chronic 2+ month delays → 51–76% default rates |
| 19 | Risk Factor Ranking | PAY_0 strongest; LIMIT_BAL protective |

---

## 📈 Results & Findings

| Metric | Value |
|---|---|
| Overall default rate | **22.1%** (6,636 / 30,000) |
| Strongest predictor | **PAY_0** (correlation: +0.325) |
| Highest-risk behavior | 2+ months late → **70%+ default rate** |
| Highest-risk demographic | 60+ age group → **28.3% default rate** |
| Lowest-risk demographic | 30s age group → **20.3% default rate** |
| Credit limit: defaulters | **NT$90K** median |
| Credit limit: non-defaulters | **NT$150K** median |
| Dangerous utilization threshold | **60%+** → default rate jumps from 17% to 28% |
| Most protective feature | **LIMIT_BAL** (correlation: -0.154) |

---

## 🚀 Future Improvements

- **Machine Learning Models** — Train Logistic Regression, Random Forest, XGBoost, LightGBM for binary classification; use SMOTE or class weighting to address the 78/22 imbalance
- **SHAP Explainability** — Apply SHAP values to explain individual prediction decisions (audit-ready, regulator-friendly)
- **Model Deployment** — Build a FastAPI endpoint + Streamlit dashboard for real-time customer risk scoring
- **Feature Store** — Engineer 15+ features (rolling payment averages, bill-to-limit ratios, trend slopes) for model input
- **Early Warning Alert System** — Automate flagging when PAY_0 ≥ 2 or utilization > 60% to trigger collection workflow
- **Time-Series Extension** — Apply temporal models (LSTM, survival analysis) if longer payment history is available
- **Segment-Based Thresholds** — Build separate risk thresholds per customer segment (age group × education) for policy differentiation

---

## ⚙️ How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn jupyter
```

### Steps
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/credit-card-default-risk-eda.git
cd credit-card-default-risk-eda

# 2. Download the dataset
# Source: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
# Place UCI_Credit_Card.csv in the project root

# 3. Launch Jupyter Notebook
jupyter notebook EDA.ipynb

# 4. Run all cells (Cell → Run All)
```

> **Note:** All visualizations are saved as PNG files in the `visualizations/` folder during execution.

---

## 📁 Folder Structure

```
credit-card-default-risk-eda/
│
├── EDA.ipynb                    # Main analysis notebook
├── UCI_Credit_Card.csv          # Raw dataset (download separately)
├── README.md                    # Project documentation
│
└── visualizations/
    ├── 01_target_distribution.png
    ├── 02_age_distribution.png
    ├── 03_credit_limit.png
    ├── 04_demographics.png
    ├── 05_payment_status.png
    ├── 06_bill_amounts.png
    ├── 07_age_vs_default.png
    ├── 08_education_vs_default.png
    ├── 09_credit_limit_vs_default.png
    ├── 10_payment_vs_default.png
    ├── 11_bill_payment_trends.png
    ├── 12_demographics_vs_default.png
    ├── 13_correlation_heatmap.png
    ├── 14_default_correlations.png
    ├── 15_pivot_heatmaps.png
    ├── 16_credit_utilization.png
    ├── 17_scatter_multivariate.png
    ├── 18_payment_history_heatmap.png
    └── 19_risk_factor_ranking.png
```

---

## 👤 Author

**Saishiva Akula**
- 📅 March 2026
- 🔗 [LinkedIn](#) | [GitHub](#) | [Portfolio](#)

---

## 📄 License

This project is open source. Dataset credit: Lichman, M. (2013). UCI Machine Learning Repository. University of California, Irvine.

---

*Built with Python • Pandas • Matplotlib • Seaborn*
