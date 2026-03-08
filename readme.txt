Customer Health / Churn Risk Scoring (customerhealthfinal)
====================================================

This repo is an end-to-end Customer Health / Churn Risk scoring mini-project.

- Backend: build account-level panels from multiple raw tables → engineer QoQ (quarter-over-quarter)
  features + missing/contract flags → train a scikit-learn Logistic Regression model → export artifacts
  (joblib model, metrics, coefficients, latest risk output).
- Frontend: a Streamlit app that lets you upload data → maps column names (aliases) → builds the
  required feature set → scores churn risk → shows explanations and lets you export results.

Note: the sample dataset is RavenStack synthetic data (see raw data/README.md).


Project Structure
-----------------
customerhealthfinal/
  raw data/                         Raw input tables (CSV) + dataset README
  output/                           All intermediate outputs + final model artifacts
  model/                            Training script(s)
  streamlit_app/                    Streamlit UI ("frontend")
  eda_custom_outputs/               EDA outputs (charts/tables)
  edavv/                            EDA/report generation scripts (optional)
  check/                            Data checking utilities (optional)
  *.py                              Backend ETL / feature scripts


Raw Inputs (raw data/)
----------------------
Expected files (RavenStack synthetic dataset):
- ravenstack_accounts.csv
- ravenstack_subscriptions.csv
- ravenstack_feature_usage.csv
- ravenstack_support_tickets.csv
- ravenstack_churn_events.csv


Outputs (output/)
-----------------
Typical outputs you should see after running the backend pipeline:

Panels / snapshots:
- account_month_panel.csv
- account_quarter_panel.csv
- account_latest_snapshot_month.csv
- account_latest_snapshot_quarter.csv

QoQ processed panels:
- account_quarter_panel_qoq_processed.csv
- account_quarter_panel_qoq_processed_with_missing_flags.csv
- account_quarter_panel_qoq_processed_with_contract_flags.csv

Model artifacts:
- output/model_outputs/churn_risk_model.joblib
- output/model_outputs/coefficient_table.csv
- output/model_outputs/validation_metrics.* (format may vary)
- output/model_outputs/account_latest_risk_output.csv

Demo / example:
- output/demo_risk_output.csv
- output/example_upload.csv


End-to-End Flow
---------------
A) Backend (ETL + feature engineering + training)
1) builddata.py
   - Reads raw data tables and builds fact tables and account month/quarter panels.
2) qoq_transform.py
   - Finds QoQ delta/change columns and applies robust processing (e.g., clipping/winsorization +
     standardization) and creates *_z features.
3) add_missing_flags.py
   - Adds missingness flag columns for key signals (e.g., satisfaction/contract missing).
4) add_contract_flags.py
   - Adds contract-related derived columns and ensures rolling churn label(s) exist.
5) model/train_churn_risk_model.py
   - Trains a Logistic Regression model (sklearn pipeline) and exports joblib + metrics + coefficients +
     latest risk output.

B) Frontend (Streamlit scoring UI)
- streamlit_app/app.py provides an upload-and-score interface.
- It supports:
  (1) Multi-table raw upload (accounts/subscriptions required; usage/tickets optional)
  (2) Single-table upload (a wide table that already contains current/previous period signals)
- It uses schema aliases to map column names and then calls scorer.py to compute:
  - risk_probability
  - risk_score (0–100)
  - risk_tier (Low/Medium/High)
  - top drivers / recommended actions (based on model coefficients and simple rules)


Quickstart (Recommended)
------------------------
Python 3.10+ is recommended (3.11 usually works as well).

IMPORTANT: If your ZIP/repo contains streamlit_app/.venv/ (a full virtual environment),
do NOT rely on it across machines/OS. Delete it and rebuild your venv using requirements.txt.

1) Create a virtual environment
   (macOS/Linux)
     cd customerhealthfinal
     python -m venv .venv
     source .venv/bin/activate
     pip install -U pip

   (Windows PowerShell)
     cd customerhealthfinal
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     python -m pip install -U pip

2) Run the backend pipeline (from repo root)
     python builddata.py
     python qoq_transform.py
     python add_missing_flags.py
     python add_contract_flags.py
     python model/train_churn_risk_model.py
     python make_example_upload.py     # optional: generates output/example_upload.csv

   After this, confirm:
     output/model_outputs/churn_risk_model.joblib
     output/model_outputs/account_latest_risk_output.csv

3) Run the Streamlit app
     cd streamlit_app
     pip install -r requirements.txt
     streamlit run app.py

   Open the URL printed by Streamlit (usually http://localhost:8501).


Backend Scripts (More Detail)
-----------------------------

builddata.py
- Inputs: raw data/*.csv
- Outputs (output/):
  - subscription_fact.csv, usage_fact.csv, tickets_fact.csv, churn_fact.csv
  - account_month_panel.csv, account_quarter_panel.csv
  - account_latest_snapshot_month.csv, account_latest_snapshot_quarter.csv

qoq_transform.py
- Input: output/account_quarter_panel.csv
- Output: output/account_quarter_panel_qoq_processed.csv
- Behavior:
  - Detects QoQ delta/change numeric columns (e.g., *qoq_delta*, *qoq_change*)
  - Applies robust transforms and creates standardized *_z columns

add_missing_flags.py
- Input: output/account_quarter_panel_qoq_processed.csv
- Output: output/account_quarter_panel_qoq_processed_with_missing_flags.csv
- Behavior:
  - Adds missing flags for key columns (exact column names depend on pipeline)

add_contract_flags.py
- Input: output/account_quarter_panel_qoq_processed_with_missing_flags.csv
- Output: output/account_quarter_panel_qoq_processed_with_contract_flags.csv
- Behavior:
  - Adds contract-derived features/flags (e.g., capped days-to-end)
  - Ensures rolling churn label columns exist (e.g., churn_label_q2_rolling)

model/train_churn_risk_model.py
- Input: output/account_quarter_panel_qoq_processed_with_contract_flags.csv
- Outputs: output/model_outputs/
  - churn_risk_model.joblib (sklearn pipeline)
  - coefficient_table.csv (feature coefficients)
  - validation metrics file(s)
  - account_latest_risk_output.csv (latest quarter scoring for each account)


Streamlit App (More Detail)
---------------------------
Key files under streamlit_app/:
- app.py
  UI, upload logic, display/export.
- raw_builders.py
  Converts uploaded raw inputs into the feature table expected by the model.
- schema_aliases.py
  Column name alias mapping to support different data warehouse naming conventions.
- scorer.py
  Loads output/model_outputs/churn_risk_model.joblib and returns risk_probability, tier, drivers, etc.

Input modes:
1) Multi-table upload (recommended)
   Required:
   - accounts.csv
   - subscriptions.csv
   Optional:
   - feature_usage.csv
   - support_tickets.csv

2) Single-table upload
   - A single wide CSV with the required "current" and "previous" signals.
   - Use output/example_upload.csv as a reference template.


Troubleshooting
---------------

1) Streamlit cannot find the model file
   Symptom: FileNotFoundError for churn_risk_model.joblib
   Fix:
   - Run: python model/train_churn_risk_model.py
   - Verify: output/model_outputs/churn_risk_model.joblib exists

2) KeyError / missing feature columns during scoring
   Root causes:
   - Your upload is missing required columns
   - Column names don’t match and are not covered by schema aliases
   Fix:
   - Compare with output/example_upload.csv
   - Add/update mappings in streamlit_app/schema_aliases.py
   - Ensure raw_builders.py produces the full feature set used by the model

3) Dependency / version issues (sklearn/joblib/pandas)
   Recommendation:
   - Install from requirements.txt
   - Consider pinning versions (especially scikit-learn) for stable joblib compatibility

4) Repo/ZIP is huge (tens of thousands of files)
   Likely cause: streamlit_app/.venv/ included.
   Fix:
   - Remove .venv/
   - Rebuild env from requirements
   - Add .venv/ to .gitignore


Suggested Engineering Improvements (Optional)
---------------------------------------------
- Remove committed virtual environments; rely on requirements + venv.
- Pin critical library versions (scikit-learn) for reproducible training and scoring.
- Version model artifacts (e.g., artifacts/v1/) and decouple training from UI runtime.
- Add minimal schema validation and unit tests for feature builders (raw_builders.py).
- Add a Makefile/task runner for one-command pipeline execution.


License / Disclaimer
--------------------
- Uses synthetic data (RavenStack).
- Model outputs are for demonstration/learning; production use requires stronger evaluation,
  monitoring, and business-aligned definitions.
