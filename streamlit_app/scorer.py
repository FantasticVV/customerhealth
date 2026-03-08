from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import load

# single source of truth for model inputs
from shared import FEATURES as MODEL_FEATURES


@lru_cache(maxsize=1)
def _load_pipeline():
    root = Path(__file__).resolve().parents[1]
    best_auc_path = root / "output" / "model_outputs" / "best_auc_model.joblib"
    model_path = root / "output" / "model_outputs" / "churn_risk_model.joblib"
    if best_auc_path.exists():
        return load(best_auc_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Make sure churn_risk_model.joblib is shipped with the app."
        )
    return load(model_path)


def _validate_model_features(df: pd.DataFrame) -> None:
    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            "Input features are missing required model columns.\n"
            f"Missing: {missing}\n\n"
            "Fix: make sure raw_builders returns shared.build_feature_set(df) output "
            "and you pass that into scorer.score()."
        )


def _cast_types(X: pd.DataFrame) -> pd.DataFrame:
    """
    Keep casting minimal. The pipeline already has imputers/scaler/onehot,
    but we still coerce types to avoid streamlit csv weirdness.
    """
    X = X.copy()

    # numeric-like
    numeric_cols = [
        "usage_count_qoq_delta_z",
        "tickets_opened_qoq_delta_z",
        "days_to_contract_end_capped",
        "arr_amount_qoq_delta_z",
        "seats_qoq_delta_z",
        "seats",
        "tenure",
        "tenure_z",
        "avg_satisfaction",
        "tenure_x_usage_drop_flag",
        "tenure_x_tickets_opened_qoq_delta_z",
    ]
    for c in numeric_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # binary-like
    for c in [
        "usage_drop_flag",
        "contract_ending_soon_flag",
        "satisfaction_missing_flag",
        "contract_missing_flag",
    ]:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype(int)

    # categorical
    if "plan_tier" in X.columns:
        X["plan_tier"] = X["plan_tier"].astype(str)

    return X


def _risk_tier_abs(p: float, high_th: float, med_th: float) -> str:
    if p >= high_th:
        return "High"
    if p >= med_th:
        return "Medium"
    return "Low"


def _pct_rank(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.rank(pct=True, method="average")


# ---------------------------
# Self-baseline helpers
# ---------------------------
def _has_self_baseline(df: pd.DataFrame) -> bool:
    return any(c in df.columns for c in [
        "hist_usage_median",
        "hist_tickets_median",
        "hist_csat_median",
        "hist_resolution_median",
        "hist_first_response_median",
    ]) and any(c in df.columns for c in [
        "usage_count_total",
        "ticket_count",
        "avg_csat",
        "avg_resolution_hours",
        "avg_first_response_minutes",
    ])


def _explain_usage_self(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    cur = pd.to_numeric(df.get("usage_count_total", np.nan), errors="coerce")
    base = pd.to_numeric(df.get("hist_usage_median", np.nan), errors="coerce")

    low_vs_self = cur.notna() & base.notna() & (cur <= 0.80 * base)
    usage_drop = df.get("usage_drop_flag", pd.Series([0] * len(df), index=df.index)).fillna(0).astype(int)
    drop_flag = usage_drop == 1

    flag = low_vs_self | drop_flag

    drivers = []
    for i in range(len(df)):
        parts = []
        if drop_flag.iloc[i]:
            parts.append("Usage declined vs previous quarter.")
        if low_vs_self.iloc[i]:
            c = float(cur.iloc[i]) if pd.notna(cur.iloc[i]) else np.nan
            b = float(base.iloc[i]) if pd.notna(base.iloc[i]) else np.nan
            parts.append(f"Usage is below its historical baseline (current {c:.0f} vs baseline ~{b:.0f}).")
        drivers.append(" ".join(parts) if parts else None)

    return flag, pd.Series(drivers, index=df.index, dtype="object")


def _explain_support_self(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    cur = pd.to_numeric(df.get("ticket_count", np.nan), errors="coerce")
    base = pd.to_numeric(df.get("hist_tickets_median", np.nan), errors="coerce")

    spike_vs_self = cur.notna() & base.notna() & (cur >= 1.30 * base)

    escalation = pd.to_numeric(df.get("escalation_count", 0), errors="coerce").fillna(0)
    has_escalation = escalation > 0

    tickets_z = pd.to_numeric(df.get("tickets_opened_qoq_delta_z", 0.0), errors="coerce").fillna(0.0)
    z_spike = tickets_z > 0.6

    flag = spike_vs_self | has_escalation | z_spike

    drivers = []
    for i in range(len(df)):
        parts = []
        if has_escalation.iloc[i]:
            parts.append("Escalations detected in the quarter.")
        if spike_vs_self.iloc[i]:
            c = float(cur.iloc[i]) if pd.notna(cur.iloc[i]) else np.nan
            b = float(base.iloc[i]) if pd.notna(base.iloc[i]) else np.nan
            parts.append(f"Ticket volume is above its historical baseline (current {c:.0f} vs baseline ~{b:.0f}).")
        elif z_spike.iloc[i]:
            parts.append("Support ticket activity increased vs previous quarter.")
        drivers.append(" ".join(parts) if parts else None)

    return flag, pd.Series(drivers, index=df.index, dtype="object")


def _explain_satisfaction_self(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    cur = pd.to_numeric(df.get("avg_csat", np.nan), errors="coerce")
    base = pd.to_numeric(df.get("hist_csat_median", np.nan), errors="coerce")

    missing = cur.isna()
    low_abs = cur.notna() & (cur <= 2.5)
    drop_vs_self = cur.notna() & base.notna() & ((base - cur) >= 0.5)

    flag = low_abs | drop_vs_self

    drivers = []
    for i in range(len(df)):
        if missing.iloc[i]:
            drivers.append(None)
            continue
        parts = []
        if low_abs.iloc[i]:
            parts.append(f"Satisfaction is low (avg CSAT ~{float(cur.iloc[i]):.1f}).")
        if drop_vs_self.iloc[i]:
            parts.append(f"Satisfaction dropped vs its baseline (baseline ~{float(base.iloc[i]):.1f}).")
        drivers.append(" ".join(parts) if parts else None)

    return flag, pd.Series(drivers, index=df.index, dtype="object")


def _explain_commercial(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    d = pd.to_numeric(df.get("days_to_contract_end_capped", 365.0), errors="coerce").fillna(365.0)
    soon = d <= 90

    auto_renew = df.get("auto_renew_flag", None)
    if auto_renew is not None:
        auto_renew = pd.to_numeric(auto_renew, errors="coerce")
        not_auto = (auto_renew == 0)
    else:
        not_auto = pd.Series([False] * len(df), index=df.index)

    drivers = []
    for i in range(len(df)):
        if soon.iloc[i]:
            msg = f"Contract renewal is within 90 days (ending in ~{int(d.iloc[i])} days)."
            if not_auto.iloc[i]:
                msg += " Auto-renew is OFF."
            drivers.append(msg)
        else:
            drivers.append(None)

    flag = soon | (soon & not_auto)
    return flag, pd.Series(drivers, index=df.index, dtype="object")


# ---------------------------
# Peer-in-upload explanations (fallback)
# ---------------------------
def _explain_usage_peers(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    usage_drop = df.get("usage_drop_flag", pd.Series([0] * len(df))).fillna(0).astype(int)
    usage_z = pd.to_numeric(df.get("usage_count_qoq_delta_z", 0.0), errors="coerce").fillna(0.0)
    usage_z_pct = _pct_rank(usage_z)
    low_usage_vs_peers = usage_z_pct <= 0.20
    usage_flag = (usage_drop == 1) | low_usage_vs_peers

    drivers = []
    for i in range(len(df)):
        parts = []
        if usage_drop.iloc[i] == 1:
            parts.append("Usage trend is declining vs previous quarter.")
        if low_usage_vs_peers.iloc[i]:
            parts.append("Usage is in the bottom range compared to peers in this upload.")
        drivers.append(" ".join(parts) if parts else None)

    return usage_flag, pd.Series(drivers, index=df.index, dtype="object")


def _explain_support_peers(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    tickets_z = pd.to_numeric(df.get("tickets_opened_qoq_delta_z", 0.0), errors="coerce").fillna(0.0)
    tickets_pct = _pct_rank(tickets_z)
    high_tickets_vs_peers = tickets_pct >= 0.80
    tickets_spike = tickets_z > 0.6
    support_flag = high_tickets_vs_peers | tickets_spike

    drivers = []
    for i in range(len(df)):
        parts = []
        if tickets_spike.iloc[i]:
            parts.append("Support ticket activity is increasing vs previous quarter.")
        if high_tickets_vs_peers.iloc[i]:
            parts.append("Ticket increase is in the top range compared to peers in this upload.")
        drivers.append(" ".join(parts) if parts else None)

    return support_flag, pd.Series(drivers, index=df.index, dtype="object")


def _explain_satisfaction_peers(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    sat = pd.to_numeric(df.get("avg_satisfaction", np.nan), errors="coerce")
    sat_missing = sat.isna()

    sat_pct = _pct_rank(sat.fillna(sat.median() if sat.notna().any() else 3.0))
    low_vs_peers = sat_pct <= 0.20
    low_abs = sat.notna() & (sat <= 2.5)

    flag = low_abs | (low_vs_peers & sat.notna())

    drivers = []
    for i in range(len(df)):
        parts = []
        if sat_missing.iloc[i]:
            drivers.append(None)
            continue
        if low_abs.iloc[i]:
            parts.append(f"Satisfaction is low (avg CSAT ~{float(sat.iloc[i]):.1f}).")
        if low_vs_peers.iloc[i]:
            parts.append("Satisfaction is in the bottom range compared to peers in this upload.")
        drivers.append(" ".join(parts) if parts else None)

    return flag, pd.Series(drivers, index=df.index, dtype="object")


def _safe_mode_message(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    # in our new pipeline, "insufficient fields" is unlikely if raw_builders is used,
    # but we keep it to avoid crashing if someone calls scorer.score() directly.
    enough = all(c in df.columns for c in ["usage_count_qoq_delta_z", "tickets_opened_qoq_delta_z", "days_to_contract_end_capped", "plan_tier"])
    if enough:
        return (
            pd.Series([False] * len(df), index=df.index),
            pd.Series([None] * len(df), index=df.index, dtype="object"),
            pd.Series([None] * len(df), index=df.index, dtype="object"),
        )

    drivers = "Insufficient fields for rule-based guidance."
    reco = "Please provide usage, support tickets, contract end date, and/or satisfaction fields."
    return (
        pd.Series([True] * len(df), index=df.index),
        pd.Series([drivers] * len(df), index=df.index, dtype="object"),
        pd.Series([reco] * len(df), index=df.index, dtype="object"),
    )


def score(
    feats: pd.DataFrame,
    high_threshold: float = 0.50,
    medium_threshold: float = 0.35,
) -> pd.DataFrame:
    """
    Input feats MUST already contain the 16 model features from shared.build_feature_set().

    Outputs:
      - risk_probability / risk_score / risk_tier
      - risk_type / drivers / recommendation (rule based, same as your original)
    """
    df = feats.copy()

    # hard validation: no silent drift
    _validate_model_features(df)

    pipeline = _load_pipeline()

    X = _cast_types(df[MODEL_FEATURES])
    proba = pipeline.predict_proba(X)[:, 1]

    df["risk_probability"] = proba
    df["risk_score"] = (df["risk_probability"] * 100).round(0).astype(int)

    df["risk_tier"] = df["risk_probability"].apply(
        lambda p: _risk_tier_abs(float(p), high_threshold, medium_threshold)
    )

    safe_flag, safe_drivers, safe_reco = _safe_mode_message(df)

    # Commercial (works in all modes)
    comm_flag, comm_driver = _explain_commercial(df)

    # Usage/Support/Satisfaction: pick self-baseline if available
    if _has_self_baseline(df):
        usage_flag, usage_driver = _explain_usage_self(df)
        support_flag, support_driver = _explain_support_self(df)
        sat_flag, sat_driver = _explain_satisfaction_self(df)
    else:
        usage_flag, usage_driver = _explain_usage_peers(df)
        support_flag, support_driver = _explain_support_peers(df)
        sat_flag, sat_driver = _explain_satisfaction_peers(df)

    risk_type = []
    drivers = []

    # Priority: Commercial > Usage > Support > Satisfaction
    for i in range(len(df)):
        if safe_flag.iloc[i]:
            risk_type.append("Unknown")
            drivers.append(safe_drivers.iloc[i])
            continue

        candidates: List[Tuple[str, Optional[str]]] = []
        if comm_flag.iloc[i]:
            candidates.append(("Commercial", comm_driver.iloc[i]))
        if usage_flag.iloc[i]:
            candidates.append(("Usage", usage_driver.iloc[i]))
        if support_flag.iloc[i]:
            candidates.append(("Support", support_driver.iloc[i]))
        if sat_flag.iloc[i]:
            candidates.append(("Satisfaction", sat_driver.iloc[i]))

        if not candidates:
            risk_type.append("General")
            drivers.append("No dominant rule-based driver detected from available fields.")
        else:
            risk_type.append(candidates[0][0])
            d_parts = [d for _, d in candidates if isinstance(d, str) and d.strip()]
            drivers.append(
                " | ".join(d_parts[:3])
                if d_parts
                else "Signals detected, but insufficient detail for explanation."
            )

    df["risk_type"] = pd.Series(risk_type, index=df.index, dtype="object")
    df["drivers"] = pd.Series(drivers, index=df.index, dtype="object")

    def _reco(rt: str) -> str:
        if rt == "Unknown":
            return safe_reco.iloc[0] if len(safe_reco) else "Please provide more fields for guidance."
        if rt == "Commercial":
            return "Focus: Commercial | Confirm renewal timeline, decision makers, and success criteria. Align early on the renewal plan."
        if rt == "Usage":
            return "Focus: Usage | Identify which workflows/features dropped (by role/team). Run enablement and adoption recovery."
        if rt == "Support":
            return "Focus: Support | Review recurring themes and known issues. Validate SLA/response quality and close-loop with the customer."
        if rt == "Satisfaction":
            return "Focus: Success | Confirm stakeholder sentiment and run a recovery plan. Identify top pain points."
        return "Focus: General | Validate health signals and request missing data fields if needed."

    df["recommendation"] = [_reco(rt) for rt in df["risk_type"].astype(str)]

    out_cols = [
        "account_id",
        "risk_score",
        "risk_probability",
        "risk_tier",
        "risk_type",
        "drivers",
        "recommendation",
    ]

    # keep useful context columns if present
    keep = [
        "plan_tier", "seats", "as_of_date", "signup_date", "quarter_end",
        "signup_date_inferred_flag", "as_of_date_inferred_flag",
        # quarterly enrich fields (optional)
        "as_of_quarter", "usage_count_total", "ticket_count", "avg_csat",
        "days_to_contract_end", "auto_renew_flag", "escalation_count",
        "hist_usage_median", "hist_tickets_median", "hist_csat_median",
    ]
    for c in keep:
        if c in df.columns and c not in out_cols:
            out_cols.append(c)

    return df[out_cols].copy()
