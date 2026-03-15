# 🔧 AI-Powered Predictive Maintenance System

> **Predict machine failures before they happen. Reduce unplanned downtime. Save millions.**

[![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=databricks&logoColor=white)](https://databricks.com)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)](https://spark.apache.org)
[![Delta Lake](https://img.shields.io/badge/Delta%20Lake-003366?style=for-the-badge&logo=delta&logoColor=white)](https://delta.io)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge&logo=python&logoColor=white)](https://xgboost.readthedocs.io)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com)

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Overview](#-solution-overview)
- [Architecture](#-architecture)
- [Datasets](#-datasets)
- [Project Structure](#-project-structure)
- [Pipeline Walkthrough](#-pipeline-walkthrough)
- [Feature Engineering](#-feature-engineering)
- [Machine Learning Results](#-machine-learning-results)
- [SQL Dashboard](#-sql-dashboard)
- [Orchestration](#-orchestration)
- [Key Insights](#-key-insights)
- [Business Impact](#-business-impact)
- [How to Run](#-how-to-run)
- [Tech Stack](#-tech-stack)

---

## 🚨 Problem Statement

Unplanned industrial equipment failure costs the global manufacturing industry over **$50 billion annually**. A single turbine failure at a power plant can cost $250,000+ per hour in lost production. Most of these failures are preventable — if you know they are coming.

Traditional maintenance strategies fall into two categories:
- **Reactive maintenance** — fix it after it breaks. Expensive. Dangerous. Unpredictable.
- **Preventive maintenance** — service everything on a fixed schedule. Wasteful. 30–40% of preventive maintenance is unnecessary.

**Predictive maintenance** is the third way: use sensor data to predict *exactly* when a machine will fail, and only intervene when necessary.

This project builds a full end-to-end predictive maintenance system on Databricks that answers three questions:

| Question | ML Task | Output |
|---|---|---|
| **Will this machine fail?** | Binary Classification | Failure probability (0–1) |
| **When will it fail?** | Regression | Remaining Useful Life in cycles |
| **How confident are we?** | Quantile Regression | 80% confidence interval for RUL |

---

## 💡 Solution Overview

This system ingests raw sensor data from industrial turbofan engines, engineers 51 time-series features using PySpark Window Functions, trains and tracks multiple ML models using MLflow, and delivers business-ready predictions to a live SQL dashboard — all orchestrated in a single automated Databricks Job.

**Key differentiators:**
- **Auto Loader** for streaming sensor ingestion — new data files are picked up automatically
- **Delta Lake** Medallion Architecture — Bronze → Silver → Gold with full audit trail
- **Databricks Feature Store** — prevents data leakage with point-in-time correct feature lookup
- **SHAP Explainability** — every prediction explained, not just made
- **Uncertainty Quantification** — confidence intervals on RUL predictions, not just point estimates
- **End-to-end automation** — one Databricks Job runs the full pipeline

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        MEDALLION ARCHITECTURE                            │
│                                                                          │
│   RAW FILES          BRONZE              SILVER              GOLD        │
│  ┌──────────┐      ┌──────────┐       ┌──────────┐       ┌──────────┐    │
│  │NASA CMAPSS│ →   │ 4 Delta  │  →    │ 3 Delta  │  →    │ 3 Delta  │    │
│  │Azure PdM  │     │  tables  │       │  tables  │       │  tables  │    │
│  │ 6 files   │     │          │       │          │       │          │    │
│  └──────────┘      │Auto Load │       │51 feats  │       │Prediction│    │
│                    │Schema  ✓ │       │Enriched  │       │Schedule  │    │
│                    │TimeTrav ✓│       │Feat Store│       │MERGE INTO│    │
│                    │OPTIMIZE ✓│       │OPTIMIZE ✓│       │OPTIMIZE ✓│   │
│                    └──────────┘       └──────────┘       └──────────┘    │
└──────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       ML PIPELINE                                   │
│                                                                     │
│  Feature Store  →  MLflow Exp 1  →  MLflow Exp 2  →  Model          │
│  gold_ml_input     Classification  RUL Regression    Registry       │
│  Point-in-time     LR + RF + XGB   LR + RF + XGB    @champion       │
│  correct           F1=0.94         RMSE=19.24        alias          │
│                    AUC=0.998       R²=0.93                          │
│                    SHAP plots      Quantile UQ                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       OUTPUTS                                       │
│                                                                     │
│  SQL Dashboard      Databricks Job       Business Impact            │
│  5 live charts      6 notebooks          Maintenance schedule       │
│  RED/YLW/GRN        Automated            Cost savings per machine   │
│  Fleet health       End-to-end           Urgency ranking            │
└─────────────────────────────────────────────────────────────────────┘
```

> 📊 See `/architecture/architecture.svg` for the full visual diagram.

---

## 📦 Datasets

### Primary — NASA CMAPSS (C-MAPSS)
**Source:** [NASA Prognostics Data Repository](https://data.nasa.gov)

The Commercial Modular Aero-Propulsion System Simulation (CMAPSS) dataset contains run-to-failure data from turbofan jet engines under different operating conditions.

| File | Description | Rows |
|---|---|---|
| `train_FD001.txt` | FD001 training — single operating condition | ~20,000 |
| `test_FD001.txt` | FD001 testing | ~13,000 |
| `train_FD002.txt` | FD002 training — 6 operating conditions | ~53,000 |
| `test_FD002.txt` | FD002 testing | ~33,000 |

Each row represents one sensor reading cycle. 21 sensors + 3 operational settings per cycle. Machines run until failure — final cycle RUL = 0.

**Why FD001 + FD002?** FD001 provides a clean single-condition baseline. FD002 adds operating condition complexity, making the model more robust and realistic.

### Supplementary — Azure Predictive Maintenance
**Source:** [Kaggle — Microsoft Azure PdM](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)

| File | Description | Rows |
|---|---|---|
| `PdM_telemetry.csv` | Voltage, rotation, pressure, vibration per machine | 876,100 |
| `PdM_maint.csv` | Maintenance event logs with component replaced | 3,286 |

**Why this dataset?** It provides real-world machine context — operational telemetry averages and maintenance history — that enriches the NASA sensor features with business-level signals.

---

## 📁 Project Structure

```
predictive-maintenance-databricks/
│
├── notebooks/
│   ├── 01_bronze_ingestion.ipynb       # Raw data → 3 Bronze Delta tables
│   ├── 02_eda.ipynb                    # 8 EDA charts, sensor selection
│   ├── 03_silver_cleaning.ipynb        # Null handling, type casting, RUL target
│   ├── 04_feature_engineering.ipynb    # 51 engineered features
│   ├── 05_silver_enrichment.ipynb      # Metadata + maintenance JOIN
│   ├── 06_feature_store.ipynb          # Feature Store registration
│   ├── 07_mlflow_classification.ipynb  # 3 classification runs + SHAP
│   ├── 08_mlflow_regression.ipynb      # 3 regression runs + quantile UQ
│   └── 09_inference_gold.ipynb         # Batch inference → Gold tables
│
├── architecture/
│   └── architecture.svg                # Medallion architecture diagram
│
├── data/
│   └── sample/                         # Sample rows for quick reference
│
└── README.md
```

---

## 🔄 Pipeline Walkthrough

### Notebook 01 — Bronze Ingestion
**Input:** Raw CSV/TXT files in Databricks Unity Catalog Volumes
**Output:** 3 Bronze Delta tables

```python
# Auto Loader pattern — streaming ingestion
df = (spark.readStream
    .format('cloudFiles')
    .option('cloudFiles.format', 'csv')
    .option('cloudFiles.schemaLocation', 'dbfs:/schema/sensor')
    .load('/Volumes/workspace/predictive_maintenance/raw_data/'))
```

**Why Auto Loader?** Standard `spark.read.csv` re-reads all files on every run. Auto Loader tracks which files have been processed and only ingests new ones — critical for production pipelines where sensors stream data continuously.

**Schema Enforcement** was applied to all Bronze tables using `ALTER TABLE SET TBLPROPERTIES` — any write that doesn't match the schema is rejected automatically, protecting data quality from the first layer.

**Delta Lake Time Travel** is available on all Bronze tables:
```sql
SELECT * FROM workspace.predictive_maintenance.bronze_nasa_sensor_raw VERSION AS OF 0
```

| Table | Rows | Columns | Key Feature |
|---|---|---|---|
| `bronze_nasa_sensor_raw` | 121,477 | 27 | Schema enforced, Auto Loader |
| `bronze_machine_metadata` | 876,100 | 6 | Azure telemetry |
| `bronze_maintenance_logs` | 3,286 | 3 | Maintenance events |

---

### Notebook 02 — Exploratory Data Analysis
**8 charts generated. Key findings documented.**

**Sensor Variance Analysis** revealed that 14 of 21 sensors have near-zero variance — they carry no predictive signal and were dropped. Only the top 7 sensors by variance were retained: `sensor_3`, `sensor_4`, `sensor_7`, `sensor_8`, `sensor_9`, `sensor_11`, `sensor_12`.

**Correlation Analysis** showed all 7 selected sensors are 0.77–1.00 correlated with each other. This high redundancy confirms dropping the others was correct — and motivates using rolling window features to differentiate their signals over time rather than raw values.

**Class Balance** confirmed that `fail_30` has ~25% positive rate — manageable imbalance. F1-score was chosen as the primary metric instead of accuracy to account for this.

**RUL Distribution** showed FD001 machines averaging ~206 cycles, FD002 averaging ~267 cycles — confirming the multi-condition complexity of FD002.

---

### Notebook 03 — Silver Cleaning
**Input:** `bronze_nasa_sensor_raw`
**Output:** `silver_sensor_cleaned` — 57,984 rows | 16 columns

Key steps:
- **14 sensors dropped** — near-zero variance, no predictive signal
- **Null handling** — forward-fill per `unit_id` using PySpark Window Function, then `dropna()`
- **Type casting** — all sensors to `DoubleType`, `unit_id`/`cycle` to `IntegerType`
- **Deduplication** — `dropDuplicates(["unit_id", "cycle"])` removes any duplicate readings
- **RUL engineering** — `RUL = max_cycle - current_cycle` per machine using Window aggregation
- **Binary targets** — `fail_30 = (RUL <= 30).cast(int)`, `fail_15 = (RUL <= 15).cast(int)`

**RUL Verification:** RUL = 0 at the last cycle of every machine confirmed ✅

---

### Notebook 04 — Feature Engineering
**Input:** `silver_sensor_cleaned`
**Output:** `silver_sensor_features` — 56,684 rows | 67 columns | **51 engineered features**

This is the most important engineering step. Raw sensor readings at a single point in time carry limited signal — degradation is a *temporal pattern*, not a snapshot.

```python
# Window specs — all features calculated PER MACHINE, not globally
w10  = Window.partitionBy("unit_id").orderBy("cycle").rowsBetween(-9, 0)
w30  = Window.partitionBy("unit_id").orderBy("cycle").rowsBetween(-29, 0)
w_lag = Window.partitionBy("unit_id").orderBy("cycle")
```

| Feature Group | Count | What It Captures |
|---|---|---|
| Rolling mean (10-cycle window) | 7 | Short-term trend — noise-smoothed degradation |
| Rolling std (10-cycle window) | 7 | Short-term variability — instability signal |
| Rolling mean (30-cycle window) | 7 | Long-term drift toward failure |
| Rolling std (30-cycle window) | 7 | Long-term volatility pattern |
| Lag features (lag-1, lag-5) | 14 | Where was the sensor recently? |
| Delta features | 7 | Rate of change — accelerating degradation |
| Health index features | 2 | Composite lifecycle position + deviation score |
| **Total** | **51** | |

**Why rolling windows?** A single sensor reading on cycle 150 means nothing in isolation. But a consistently declining `sensor_7_mean_w10` over 10 cycles, combined with rising `sensor_7_std_w30`, is a strong degradation signal. The model learns *patterns*, not snapshots.

**Why delta features?** A sudden spike in `sensor_3_delta` — the rate of change — indicates accelerating wear. This is the earliest warning signal available in the data.

**OPTIMIZE + ZORDER** applied on `(unit_id, cycle)` for fast ML query performance.

---

### Notebook 05 — Silver Enrichment
**Input:** `silver_sensor_features` + `bronze_machine_metadata` + `bronze_maintenance_logs`
**Output:** `silver_machine_enriched` — 56,684 rows | 73 columns

Raw sensor features don't tell the full story. A machine showing slightly unusual readings that was serviced last week is far less concerning than an identical machine that has never been maintained.

This notebook adds **operational context** via two LEFT JOINs:

**From Azure telemetry (machine_metadata):**
- `avg_volt` — average operating voltage
- `avg_rotate` — average rotation speed
- `avg_pressure` — average pressure
- `avg_vibration` — average vibration level

**From maintenance logs:**
- `maintenance_count` — total times the machine has been serviced
- `unique_components_replaced` — variety of maintenance performed

**LEFT JOIN** was used to preserve all sensor records. Machines with no metadata match receive 0-filled context columns — no data loss.

**Schema Evolution** tracked via `DESCRIBE HISTORY` — version history visible across all Silver tables.

---

### Notebook 06 — Feature Store
**Input:** `silver_machine_enriched`
**Output:** Feature Store table + `gold_ml_input`

**Why Feature Store?**

Without Feature Store, training and inference can silently use different features. The model performs well in development but fails in production because the feature pipeline has drifted. Feature Store enforces consistency by registering features with a version and timestamp.

**Point-in-time correctness** ensures that when training on a failure event at cycle 200, we only use features that existed *at* cycle 200 — not features calculated with data from cycle 201+. This prevents data leakage that would inflate model performance artificially.

**Data Quality Validation** built into this notebook checks:
- Row count > 50,000 ✅
- Zero nulls in critical columns ✅
- RUL range valid (0–380) ✅
- Binary targets are only 0 or 1 ✅
- 260 unique machines confirmed ✅
- Feature nulls: 0 ✅

---

### Notebook 07 — MLflow Classification
**Experiment:** `PredMaint-Classification`
**Task:** Predict machine failure within 30 cycles (binary)

Three models trained, tracked, and compared in MLflow:

| Model | F1-Score | AUC-ROC | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | baseline | baseline | — | — |
| Random Forest | improved | improved | — | — |
| **XGBoost** ⭐ | **0.9420** | **0.9984** | — | — |

**Why XGBoost wins:** Gradient boosting captures non-linear interactions between rolling window features and lag features that logistic regression cannot model. Random Forest builds independent trees — XGBoost builds them sequentially, each correcting the errors of the previous.

**SHAP Explainability** was added to the best XGBoost model:

```python
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test.iloc[:500])
```

Three SHAP artifacts logged to MLflow:
1. **Global feature importance** — which features drive predictions across the full fleet
2. **Summary dot plot** — feature impact direction and magnitude
3. **Waterfall chart** — single-machine failure explanation

**Best model registered:** `PredMaint-Classifier@champion` in MLflow Model Registry.

---

### Notebook 08 — MLflow Regression
**Experiment:** `PredMaint-RUL`
**Task:** Predict Remaining Useful Life (continuous)

| Model | RMSE (cycles) | R² |
|---|---|---|
| Linear Regression | 27.95 | 0.8517 |
| Random Forest | 22.26 | 0.9059 |
| **XGBoost** ⭐ | **19.24** | **0.9298** |

**RMSE of 19.24 cycles** means the model's RUL predictions are off by ~19 cycles on average. For a maintenance window of 30 cycles, this gives ~10 cycles of reliable warning time — more than enough for planned maintenance scheduling.

#### ⚡ Uncertainty Quantification — Quantile Regression

Instead of a single RUL estimate, the system produces a **confidence interval** using XGBoost Quantile Regression:

```python
xgb_q10 = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.1)  # lower bound
xgb_q50 = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5)  # median
xgb_q90 = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.9)  # upper bound
```

**Example output:**
> *"Machine 47 will fail between 23 and 41 cycles with 80% confidence. Schedule maintenance in 23 cycles."*

This is how production ML systems at Boeing, GE, and Siemens actually work — maintenance engineers need to know *how uncertain* the prediction is, not just the point estimate.

**Best model registered:** `PredMaint-RUL@champion` in MLflow Model Registry.

---

### Notebook 09 — Gold Layer Inference
**Input:** Production models from Registry + `silver_machine_enriched`
**Output:** `gold_failure_predictions` + `gold_maintenance_schedule`

Batch inference runs against the latest features from the Silver layer, generating predictions for every machine in the fleet.

**Traffic light risk classification:**
- 🔴 **RED** — bottom 25% RUL — IMMEDIATE action required
- 🟡 **YELLOW** — 25th–50th percentile RUL — planned maintenance within 1 week
- 🟢 **GREEN** — top 50% RUL — monitor only

**MERGE INTO pattern** for production-grade upserts:
```sql
MERGE INTO gold_failure_predictions AS target
USING new_predictions AS source
ON target.unit_id = source.unit_id AND target.cycle = source.cycle
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
```
This ensures subsequent pipeline runs *update* existing predictions rather than overwriting the entire table — critical for audit trail integrity.

---

## 🔬 Feature Engineering

51 features engineered from 7 base sensors using PySpark Window Functions:

```
sensor_3, sensor_4, sensor_7, sensor_8, sensor_9, sensor_11, sensor_12
```

Each sensor contributes:
```
sensor_X_mean_w10    Rolling mean, 10-cycle window
sensor_X_std_w10     Rolling std, 10-cycle window
sensor_X_mean_w30    Rolling mean, 30-cycle window
sensor_X_std_w30     Rolling std, 30-cycle window
sensor_X_lag1        Reading 1 cycle ago
sensor_X_lag5        Reading 5 cycles ago
sensor_X_delta       Rate of change (current - previous)
```

Plus 2 composite health index features:
```
normalized_cycle         Position in machine lifecycle (0=new, 1=failure)
sensor_deviation_score   Sum of z-scores across 7 sensors — composite health
```

**Feature correlation vs RUL** confirmed all 51 features carry predictive signal before training.

---

## 📊 Machine Learning Results

### Classification — Failure Prediction

```
Model:      XGBoost Classifier
F1-Score:   0.9420   ← primary metric (handles class imbalance)
AUC-ROC:    0.9984   ← near-perfect discrimination
Target:     fail_30 (failure within 30 cycles)
Features:   51 engineered time-series features
Train/Test: 80/20 stratified split
```

### Regression — RUL Estimation

```
Model:      XGBoost Regressor
RMSE:       19.24 cycles
MAE:        (logged in MLflow)
R²:         0.9298
Target:     RUL (continuous)
Uncertainty: 80% confidence interval via quantile regression
```

### Model Registry

| Model Name | Alias | Version |
|---|---|---|
| `workspace.default.predmaint-classifier` | `@champion` | 1 |
| `workspace.default.predmaint-rul` | `@champion` | 1 |

---

## 📈 SQL Dashboard

**Dashboard:** `Predictive Maintenance — Fleet Health Monitor`

5 live charts built in Databricks SQL:

| Chart | Type | What It Shows |
|---|---|---|
| Top 10 machines by failure probability | Bar | Which machines need attention NOW |
| Fleet risk zone distribution | Pie | RED / YELLOW / GREEN fleet overview |
| Predicted RUL across all machines | Bar | How much life each machine has left |
| Maintenance schedule table | Table | Urgency rank + recommended action + cost savings |
| Average RUL by risk zone | Bar | Confirms GREEN > YELLOW > RED ordering |

All charts refresh from `gold_maintenance_schedule` — the single source of truth for business decisions.

---

## ⚙️ Orchestration

**Databricks Job:** `Predictive Maintenance — Full Pipeline`

6 notebooks chained in sequence with dependency management:

```
01_bronze_ingestion
        ↓
03_silver_cleaning
        ↓
04_feature_engineering
        ↓
05_silver_enrichment
        ↓
06_feature_store
        ↓
09_inference_gold
```

- **Run mode:** Serverless compute
- **Failure alert:** Email notification on any task failure
- **Last run:** All 6 tasks succeeded ✅

One click runs the entire pipeline from raw sensor files to updated predictions and refreshed dashboard.

---

## 💡 Key Insights

**1. Degradation is temporal, not instantaneous**
A single sensor reading has limited predictive value. Rolling window features over 10 and 30 cycles capture the *trend* of degradation — the model learns patterns, not snapshots.

**2. Rate of change beats absolute values**
Delta features (sensor_X_delta) consistently ranked among the top SHAP features. Accelerating change is the earliest failure signal.

**3. High sensor correlation doesn't mean redundancy**
All 7 selected sensors are 0.77–1.00 correlated with each other in raw form. But after window transformation, their rolling statistics diverge meaningfully — each captures a slightly different aspect of the degradation process.

**4. Class imbalance requires the right metric**
With ~25% positive rate for `fail_30`, a naive model predicting "no failure" every time achieves 75% accuracy. F1-score was chosen as the primary metric to penalise both false positives and false negatives equally.

**5. Point estimates are insufficient for maintenance decisions**
A maintenance engineer doesn't just need "machine 47 will fail in 30 cycles." They need "machine 47 will fail between 23 and 41 cycles — 80% confidence." Quantile regression provides actionable uncertainty bounds that justify scheduling decisions.

---

## 💰 Business Impact

For a manufacturing plant operating 260 machines:

| Metric | Value |
|---|---|
| Machines monitored | 260 |
| RED zone (immediate action) | ~65 machines |
| YELLOW zone (planned maintenance) | ~65 machines |
| Average cost of unplanned failure | $50,000 |
| Average cost of planned maintenance | $5,000 |
| Cost savings per prevented failure | $45,000 |
| **Estimated monthly savings** | **$2.9M+** |

> *"This system gives maintenance engineers a 48-hour warning before failure happens. For a plant running 200 machines, preventing 4–5 failures per month saves an estimated $500K annually in downtime costs alone."*

---

## 🚀 How to Run

### Prerequisites
- Databricks Community Edition account
- Unity Catalog enabled
- Python 3.8+ with PySpark, XGBoost, MLflow, SHAP

### Step 1 — Clone Repository
```bash
git clone https://github.com/abhinavjajoo19/predictive-maintenance-databricks.git
```

### Step 2 — Upload Data
Upload these files to `/Volumes/workspace/predictive_maintenance/raw_data/`:
- `train_FD001.txt`, `test_FD001.txt` — [NASA CMAPSS](https://data.nasa.gov)
- `train_FD002.txt`, `test_FD002.txt` — [NASA CMAPSS](https://data.nasa.gov)
- `PdM_telemetry.csv`, `PdM_maint.csv` — [Azure PdM Kaggle](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)

### Step 3 — Create Schema
```sql
CREATE SCHEMA IF NOT EXISTS workspace.predictive_maintenance;
CREATE VOLUME IF NOT EXISTS workspace.predictive_maintenance.raw_data;
```

### Step 4 — Run Notebooks in Order
```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09
```

### Step 5 — Run Full Pipeline (Automated)
Trigger the Databricks Job `Predictive Maintenance — Full Pipeline` to run the complete pipeline end-to-end in one click.

### Step 6 — View Dashboard
Open `Predictive Maintenance — Fleet Health Monitor` in Databricks SQL Dashboards.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Platform** | Databricks Community Edition | Unified analytics platform |
| **Storage** | Delta Lake | ACID transactions, time travel, schema enforcement |
| **Ingestion** | Auto Loader (cloudFiles) | Streaming incremental ingestion |
| **Processing** | Apache Spark (PySpark) | Distributed data transformation |
| **Features** | Databricks Feature Store | Point-in-time correct feature lookup |
| **ML Tracking** | MLflow | Experiment tracking, model registry |
| **Models** | XGBoost, Random Forest, Logistic/Linear Regression | Classification + regression |
| **Explainability** | SHAP (TreeExplainer) | Global + local feature importance |
| **Uncertainty** | XGBoost Quantile Regression | 80% prediction confidence intervals |
| **Visualization** | Databricks SQL Dashboard | Business-ready fleet health monitor |
| **Orchestration** | Databricks Jobs | End-to-end pipeline automation |
| **Version Control** | GitHub + Databricks Repos | Code management + notebook sync |
| **Language** | Python 3.8+ | All notebooks |

---

## 📄 Dataset Citations

```
A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set",
NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA
```

```
Microsoft Azure Predictive Maintenance Dataset
Available at: https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance
```

---

## 🏆 Contest Submission

**Contest:** Databricks AI Challenge
**Submission Date:** March 17, 2026
**Category:** Advanced — End-to-End ML Pipeline

**Final Results:**
```
Classification F1:   0.9420
Classification AUC:  0.9984
RUL RMSE:            19.24 cycles
RUL R²:              0.9298
Features engineered: 51
Delta tables:        9
MLflow runs:         7
```

---

*Built with ❤️ on Databricks | Powered by Delta Lake + MLflow + XGBoost*
