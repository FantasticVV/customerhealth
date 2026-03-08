from __future__ import annotations

import numpy as np
import pandas as pd

# =============================================================================
# Single Source of Truth:
# - These are the exact feature columns expected by both:
#   (1) modeling pipeline training
#   (2) Streamlit frontend scoring
# =============================================================================

FEATURES = [
    "usage_count_qoq_delta_z",
    "tickets_opened_qoq_delta_z",
    "days_to_contract_end_capped",
    "arr_amount_qoq_delta_z",
    "seats_qoq_delta_z",
    "seats",
    "tenure",
    "tenure_z",
    "tenure_x_usage_drop_flag",
    "tenure_x_tickets_opened_qoq_delta_z",
    "avg_satisfaction",
    "satisfaction_missing_flag",
    "contract_missing_flag",
    "usage_drop_flag",
    "contract_ending_soon_flag",
    "plan_tier",
]

PRIMARY_PREDICTORS = [
    "usage_count_qoq_delta_z",
    "tickets_opened_qoq_delta_z",
    "arr_amount_qoq_delta_z",
]

RISK_TYPE_MAP = {
    "Usage Risk": ["usage_count_qoq_delta_z"],
    "Support Risk": ["tickets_opened_qoq_delta_z"],
    "Commercial Risk": ["arr_amount_qoq_delta_z"],
}


def _to_numeric(s: pd.Series, default: float | int | None = None) -> pd.Series:
    """Coerce series to numeric; optionally fillna with default."""
    out = pd.to_numeric(s, errors="coerce")
    if default is not None:
        out = out.fillna(default)
    return out


def add_tenure_quarters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add tenure (in quarters, starting at 1) from signup_date and quarter_end.

    Required columns:
      - signup_date
      - quarter_end

    Output:
      - tenure
    """
    out = df.copy()
    signup_q = pd.to_datetime(out["signup_date"], errors="coerce").dt.to_period("Q")
    qend_q = pd.to_datetime(out["quarter_end"], errors="coerce").dt.to_period("Q")

    # (qend_q - signup_q) -> pd.offsets style; use .n
    tenure_q = (qend_q - signup_q).apply(lambda x: x.n if pd.notna(x) else np.nan) + 1
    out["tenure"] = tenure_q
    return out


def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build / finalize the model feature set. This function is used by BOTH
    modeling and frontend.

    Expected input df should contain (at minimum):
      - usage_count_qoq_delta_z
      - tickets_opened_qoq_delta_z
      - usage_drop_flag
      - days_to_contract_end_asof_qend
      - arr_amount_qoq_delta_z (optional; default 0)
      - seats_qoq_delta_z (optional; default 0)
      - seats (optional; default 0)
      - plan_tier
      - avg_satisfaction (optional; can be NaN)
      - satisfaction_missing_flag (optional)
      - contract_missing_flag (optional)
      - signup_date, quarter_end (if tenure not already present)

    Output:
      - a DataFrame containing exactly FEATURES columns (in that order).
    """
    out = df.copy()

    # -----------------------------
    # Ensure required numeric inputs exist
    # -----------------------------
    for col in ["usage_count_qoq_delta_z", "tickets_opened_qoq_delta_z"]:
        if col not in out.columns:
            raise ValueError(f"Missing required column: {col}")

    # Optional numeric inputs
    if "arr_amount_qoq_delta_z" not in out.columns:
        out["arr_amount_qoq_delta_z"] = 0.0
    if "seats_qoq_delta_z" not in out.columns:
        out["seats_qoq_delta_z"] = 0.0
    if "seats" not in out.columns:
        out["seats"] = 0.0

    # Cast numeric
    out["usage_count_qoq_delta_z"] = _to_numeric(out["usage_count_qoq_delta_z"], default=0.0)
    out["tickets_opened_qoq_delta_z"] = _to_numeric(out["tickets_opened_qoq_delta_z"], default=0.0)
    out["arr_amount_qoq_delta_z"] = _to_numeric(out["arr_amount_qoq_delta_z"], default=0.0)
    out["seats_qoq_delta_z"] = _to_numeric(out["seats_qoq_delta_z"], default=0.0)
    out["seats"] = _to_numeric(out["seats"], default=0.0)

    # -----------------------------
    # Plan tier (categorical)
    # -----------------------------
    if "plan_tier" not in out.columns:
        raise ValueError("Missing required column: plan_tier")
    out["plan_tier"] = out["plan_tier"].astype(str).fillna("unknown")

    # -----------------------------
    # Contract features
    # -----------------------------
    if "days_to_contract_end_asof_qend" not in out.columns:
        raise ValueError("Missing required column: days_to_contract_end_asof_qend")

    out["days_to_contract_end_asof_qend"] = _to_numeric(out["days_to_contract_end_asof_qend"], default=np.nan)

    # contract_missing_flag: if not provided, derive it
    if "contract_missing_flag" not in out.columns:
        out["contract_missing_flag"] = out["days_to_contract_end_asof_qend"].isna().astype(int)
    else:
        out["contract_missing_flag"] = _to_numeric(out["contract_missing_flag"], default=0).astype(int)

    # Fill missing contract days with conservative default (same as modeling script earlier)
    out["days_to_contract_end_asof_qend"] = out["days_to_contract_end_asof_qend"].fillna(365)

    out["days_to_contract_end_capped"] = out["days_to_contract_end_asof_qend"].clip(upper=365)
    out["contract_ending_soon_flag"] = (out["days_to_contract_end_asof_qend"] <= 90).astype(int)

    # -----------------------------
    # Usage drop flag (binary)
    # -----------------------------
    if "usage_drop_flag" not in out.columns:
        # safest default: 0
        out["usage_drop_flag"] = 0
    out["usage_drop_flag"] = _to_numeric(out["usage_drop_flag"], default=0).astype(int)

    # -----------------------------
    # Satisfaction + missing flag
    # -----------------------------
    if "avg_satisfaction" not in out.columns:
        out["avg_satisfaction"] = np.nan
    out["avg_satisfaction"] = _to_numeric(out["avg_satisfaction"], default=np.nan)

    if "satisfaction_missing_flag" not in out.columns:
        out["satisfaction_missing_flag"] = out["avg_satisfaction"].isna().astype(int)
    else:
        out["satisfaction_missing_flag"] = _to_numeric(out["satisfaction_missing_flag"], default=0).astype(int)

    # -----------------------------
    # Tenure + interactions
    # -----------------------------
    if "tenure" not in out.columns:
        # need signup_date & quarter_end
        if ("signup_date" not in out.columns) or ("quarter_end" not in out.columns):
            raise ValueError("Missing tenure and missing signup_date/quarter_end to compute it.")
        out = add_tenure_quarters(out)

    out["tenure"] = _to_numeric(out["tenure"], default=np.nan)

    tenure_std = out["tenure"].std()
    if pd.isna(tenure_std) or tenure_std <= 1e-12:
        out["tenure_z"] = 0.0
    else:
        out["tenure_z"] = (out["tenure"] - out["tenure"].mean()) / tenure_std

    out["tenure_x_usage_drop_flag"] = out["tenure_z"] * out["usage_drop_flag"].astype(float)
    out["tenure_x_tickets_opened_qoq_delta_z"] = out["tenure_z"] * out["tickets_opened_qoq_delta_z"].astype(float)

    # -----------------------------
    # Final output: ensure all required FEATURES exist
    # -----------------------------
    missing = [c for c in FEATURES if c not in out.columns]
    if missing:
        raise ValueError(f"build_feature_set() still missing required feature columns: {missing}")

    return out[FEATURES].copy()
