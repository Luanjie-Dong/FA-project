# Medical Procedure Loan Credit Scorecard

## Objective
Develop a credit scorecard for predicting default risk on medical procedure loans.

---

## 1. Data Exploration (`1_data_exploration.ipynb`)

### Dataset Overview

| Metric | Application | Bureau |
|--------|-------------|--------|
| **Rows** | 307,511 applicants | 1,716,428 credit records |
| **Columns** | 120 | 17 |
| **Numeric** | 104 | 14 |
| **Categorical** | 16 | 3 |
| **Target** | 8% default (11.4:1 imbalance) | — (no target) |
| **Key insight** | 307,511 unique applicants | 305,811 applicants have bureau history (avg 5.6 records/applicant) |

### Feature Analysis
- **Top predictors of default:** avg_days_credit, DAYS_EMPLOYED, REGION_RATING_CLIENT_W_CITY, DAYS_BIRTH
- **Data quality:** No major anomalies in bureau; application has outliers and missingess patterns requiring treatment

---

## 2. Data Preparation (`2_data_preparation.ipynb`)
**Input:** Application (307,511 × 120), Bureau (1,716,428 × 17)
**Output:** 252,136 rows × 21 features (24 → 21 after fair lending compliance)

### Fair Lending Compliance
**Dropped 3 protected class features** for regulatory and ethical compliance:
- **CODE_GENDER** — Gender discrimination (Equal Credit Opportunity Act violation)
- **DAYS_BIRTH** — Age discrimination in lending (ECOA protected)
- **NAME_FAMILY_STATUS** — Marital status discrimination (Fair Housing Act violation)

**Regulatory Framework:**
- Equal Credit Opportunity Act (ECOA): Prohibits discrimination based on protected characteristics
- Fair Housing Act: Extends protection to housing-related credit discrimination
- Fair Lending Principle: Scorecard must make credit decisions based on creditworthiness, not protected class membership

### Application Data Cleaning
| Stage | Dropped | Result |
|-------|---------|--------|
| High missing (>50%) | 37 columns | 120 → 83 columns |
| Fix DAYS_* (absolute value, drop > 120 years) | 55,375 rows | 252,136 rows × 83 cols |

### Bureau Data Processing
- **Rows:** No dropping needed (all data valid)
- **Aggregations:** count, sum, mean, max by applicant
- **Derived Ratios:** debt_to_credit, credit_utilization, prolongation_rate, overdue_frequency
- **Result:** 305,811 applicants × 23 aggregated features

### Data Merging & Feature Filtering
| Stage | Result |
|-------|--------|
| Application data after cleaning | 252,136 rows × 83 cols |
| Bureau data (no cleaning) | 305,811 applicants × 17 cols |
| Merge (left join on SK_ID_CURR) | 252,136 rows × 104 cols |
| Step 1: Drop \|r\| < 0.05 | 252,136 rows × 25 cols (79 dropped) |
| Step 2: Drop multicollinear pairs \|r\| > 0.8 | 252,136 rows × 24 cols (1 dropped) |
| **Step 3: Fair lending compliance** | **252,136 rows × 21 cols (3 dropped)** |
| Final dataset | 21 features (6 numeric + 15 categorical) |

### Feature Filtering (3-Step Process)

**Step 1: Low Correlation (|r| < 0.05 with TARGET)**
- Threshold: Drop features with weak predictive power (|r| < 5%)
- Drops all features with absolute correlation < 0.05 to TARGET
- **Columns dropped: 79** (104 → 25)

**Step 2: Multicollinearity (|r| > 0.8 between features)**
- Find feature pairs with high inter-correlation (> 80%)
- For each pair, drop the one with **lower correlation to TARGET**
- Keeps strongest predictors, removes redundant noise
- **Columns dropped: 1** (25 → 24)

**Step 3: Fair Lending Compliance ⭐ ETHICAL AI** 
- Drop protected class features that violate fair lending principles
- **CODE_GENDER** — Direct gender discrimination (ECOA violation)
- **DAYS_BIRTH** — Age discrimination in lending (ECOA violation)
- **NAME_FAMILY_STATUS** — Marital status discrimination (Fair Housing Act violation)
- **Columns dropped: 3** (24 → 21)

**Final Results:**
- Step 1 - Low correlation (|r| < 0.05): **Dropped 79 columns** (104 → 25)
- Step 2 - Multicollinearity (|r| > 0.8): **Dropped 1 column** (25 → 24)
- Step 3 - Fair lending compliance: **Dropped 3 columns** (24 → 21)
- **Final dataset: 252,136 rows × 21 columns (ethically compliant)**
- **Interpretation:** Multi-stage filtering balances predictive power, redundancy removal, and regulatory compliance

---

## 3. Grouping & Screening (`3_grouping_and_screening.ipynb`)
**Input:** Prepared dataset (252,136 rows × 21 features, fair lending compliant)
**Output:** 13 selected features via WOE/IV analysis (Information Value ≥ 0.02)

### Full Analysis of All 21 Columns

| # | Column | Type | Status | IV | Classification | Bins | Selected |
|---|--------|------|--------|-----|-----------------|------|----------|
| 1 | TARGET | Numeric (Binary) | Target Variable | N/A | N/A | — | ❌ |
| 2 | NAME_CONTRACT_TYPE | Categorical | Binned | 0.0196 | Not Useful | 2 | ❌ |
| 3 | FLAG_OWN_CAR | Categorical | Binned | 0.0132 | Not Useful | 2 | ❌ |
| 4 | FLAG_OWN_REALTY | Categorical | Binned | 0.0001 | Not Useful | 2 | ❌ |
| 5 | AMT_GOODS_PRICE | Numeric | Binned | 0.0670 | Weak | 7 | ✅ |
| 6 | NAME_TYPE_SUITE | Categorical | Binned | 0.0023 | Not Useful | 3 | ❌ |
| 7 | NAME_INCOME_TYPE | Categorical | Binned | 0.0290 | Weak | 3 | ✅ |
| 8 | NAME_EDUCATION_TYPE | Categorical | Binned | 0.0662 | Weak | 2 | ✅ |
| 9 | NAME_HOUSING_TYPE | Categorical | Binned | 0.0128 | Not Useful | 2 | ❌ |
| 10 | DAYS_EMPLOYED | Numeric | Binned | 0.0899 | Weak | 4 | ✅ |
| 11 | FLAG_MOBIL | Numeric | Excluded | N/A | N/A | — | ❌ |
| 12 | OCCUPATION_TYPE | Categorical | Binned | 0.0678 | Weak | 4 | ✅ |
| 13 | REGION_RATING_CLIENT_W_CITY | Numeric | Binned | 0.0594 | Weak | 3 | ✅ |
| 14 | WEEKDAY_APPR_PROCESS_START | Categorical | Binned | 0.0009 | Not Useful | 4 | ❌ |
| 15 | ORGANIZATION_TYPE | Categorical | Binned | 0.0440 | Weak | 5 | ✅ |
| 16 | HOUSETYPE_MODE | Categorical | Binned | 0.0234 | Weak | 2 | ✅ |
| 17 | EMERGENCYSTATE_MODE | Categorical | Binned | 0.0251 | Weak | 2 | ✅ |
| 18 | DAYS_LAST_PHONE_CHANGE | Numeric | Binned | 0.0476 | Weak | 3 | ✅ |
| 19 | avg_days_credit | Numeric | Binned | 0.1207 | Medium | 5 | ✅ |
| 20 | num_active | Numeric | Binned | 0.0453 | Weak | 6 | ✅ |
| 21 | debt_to_credit | Numeric | Binned | 0.0957 | Weak | 5 | ✅ |

**Note:** Rows 3, 4, 9 (FLAG_OWN_CAR, FLAG_OWN_REALTY, NAME_HOUSING_TYPE) shown in this table were in the prepared data but excluded from final selection. Fair lending protected features (CODE_GENDER, DAYS_BIRTH, NAME_FAMILY_STATUS) were removed during Data Preparation stage and do not appear here.

### Selection Results
- **Total Evaluated:** 21 features (6 numeric + 15 categorical, after fair lending compliance)
- **Selected:** 13 features (7 numeric + 6 categorical)
  - **Numeric (7/6):** 1 Medium strength + 6 Weak strength
  - **Categorical (6/15):** Preprocessed with zero-event & sparse category merging
- **Excluded (8 features):** 7 for low IV (< 0.02) + 1 for technical reasons (FLAG_MOBIL)
- **Categorical Preprocessing:** OCCUPATION_TYPE (18→6), ORGANIZATION_TYPE (57→5), etc.

**Selected Features (13):**
1. avg_days_credit (IV: 0.1207, Medium)
2. debt_to_credit (IV: 0.0957, Weak)
3. DAYS_EMPLOYED (IV: 0.0899, Weak)
4. OCCUPATION_TYPE (IV: 0.0678, Weak)
5. AMT_GOODS_PRICE (IV: 0.0670, Weak)
6. NAME_EDUCATION_TYPE (IV: 0.0662, Weak)
7. REGION_RATING_CLIENT_W_CITY (IV: 0.0594, Weak)
8. DAYS_LAST_PHONE_CHANGE (IV: 0.0476, Weak)
9. num_active (IV: 0.0453, Weak)
10. ORGANIZATION_TYPE (IV: 0.0440, Weak)
11. NAME_INCOME_TYPE (IV: 0.0290, Weak)
12. EMERGENCYSTATE_MODE (IV: 0.0251, Weak)
13. HOUSETYPE_MODE (IV: 0.0234, Weak)

---

## 4. Scorecard Generation & Tuning (`4_scorecard_generation_and_tuning.ipynb`)
**Input:** Selected features from Step 3 (252,136 rows × 13 features, fair lending compliant)
**Output:** Calibrated credit scorecard with optimal cutoff score and business impact analysis

### WOE Encoding & Feature Engineering
- **Numeric Features:** Binned with monotonic WOE trends
- **Categorical Features:** Preprocessed categories merged per IV analysis
- **Result:** 13 WOE-encoded features + TARGET = 14 columns for model training

### Model Development
**Logistic Regression (Class-Weighted)**
- Data split: 70% training | 30% test (stratified)
- Class weights: balanced (accounts for 91% vs 9% imbalance)
- Hyperparameter tuning: GridSearchCV on C ∈ [0.001, 0.01, 0.1, 1, 10, 100]

### Credit Score Transformation
- **Base Score:** 600 points
- **Odds:** 1:50 (default assumption)
- **PDO:** 20 points (Points to Double Odds)
- **Formula:** Score = 600 + (20/ln(2)) × ln(odds / odds₀)
- **Interpretation:** Each 20 points = 2× change in odds

### Cutoff Optimization & Business Analysis
**Dual-Axis Analysis:** Evaluates profit vs approval rate across cutoff range
- **Approval Rate:** % of applications approved at each cutoff
- **Expected Default Rate:** Calculated from formula: P(default) = odds / (1 + odds)
- **Net Profit:** Revenue per approval - losses from defaults
- **Recommended Cutoff:** Maximizes profit while balancing business objectives


