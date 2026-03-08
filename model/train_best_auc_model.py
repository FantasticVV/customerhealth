from __future__ import annotations

from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =============================================================================
# Shared feature engineering import (single source of truth)
# =============================================================================
def _try_import_shared():
    """
    Try import shared.feature_engineering from project root.
    train_best_auc_model.py is expected to live under <root>/model/train_best_auc_model.py
    so parents[1] should be <root>.
    """
    base_dir = Path(__file__).resolve().parents[1]
    import sys

    if str(base_dir) not in sys.path:
        sys.path.append(str(base_dir))

    try:
        from shared.feature_engineering import build_feature_set, add_tenure_quarters, FEATURES
        return build_feature_set, add_tenure_quarters, FEATURES, True
    except Exception:
        return None, None, None, False


_shared_build_feature_set, _shared_add_tenure_quarters, _SHARED_FEATURES, _HAS_SHARED = _try_import_shared()

if _HAS_SHARED:
    build_feature_set = _shared_build_feature_set
    add_tenure_quarters = _shared_add_tenure_quarters
    FEATURES = _SHARED_FEATURES
else:
    # -----------------------------
    # Fallback local implementations
    # -----------------------------
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

    def add_tenure_quarters(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["signup_date"] = pd.to_datetime(out.get("signup_date"), errors="coerce")
        out["quarter_end"] = pd.to_datetime(out.get("quarter_end"), errors="coerce")
        signup_q = out["signup_date"].dt.to_period("Q")
        qend_q = out["quarter_end"].dt.to_period("Q")
        tenure_q = (qend_q - signup_q).apply(lambda x: x.n if pd.notna(x) else np.nan) + 1
        out["tenure"] = tenure_q
        return out

    def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # --- contract ---
        if "days_to_contract_end_asof_qend" not in df.columns:
            # fallbacks
            if "days_to_contract_end" in df.columns:
                df["days_to_contract_end_asof_qend"] = df["days_to_contract_end"]
            elif "days_to_contract_end_current" in df.columns:
                df["days_to_contract_end_asof_qend"] = df["days_to_contract_end_current"]
            else:
                df["days_to_contract_end_asof_qend"] = np.nan

        df["days_to_contract_end_asof_qend"] = pd.to_numeric(df["days_to_contract_end_asof_qend"], errors="coerce")
        df["contract_missing_flag"] = df["days_to_contract_end_asof_qend"].isna().astype(int)
        df["days_to_contract_end_asof_qend"] = df["days_to_contract_end_asof_qend"].fillna(365.0)

        df["days_to_contract_end_capped"] = df["days_to_contract_end_asof_qend"].clip(upper=365.0)
        df["contract_ending_soon_flag"] = (df["days_to_contract_end_asof_qend"] <= 90).astype(int)

        # --- numeric defaults ---
        for c in ["usage_count_qoq_delta_z", "tickets_opened_qoq_delta_z", "arr_amount_qoq_delta_z", "seats_qoq_delta_z", "seats"]:
            if c not in df.columns:
                df[c] = 0.0
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        # --- satisfaction ---
        if "avg_satisfaction" not in df.columns:
            if "avg_csat" in df.columns:
                df["avg_satisfaction"] = df["avg_csat"]
            else:
                df["avg_satisfaction"] = np.nan
        df["avg_satisfaction"] = pd.to_numeric(df["avg_satisfaction"], errors="coerce")
        if "satisfaction_missing_flag" not in df.columns:
            df["satisfaction_missing_flag"] = df["avg_satisfaction"].isna().astype(int)

        # --- usage drop flag ---
        if "usage_drop_flag" not in df.columns:
            df["usage_drop_flag"] = 0
        df["usage_drop_flag"] = pd.to_numeric(df["usage_drop_flag"], errors="coerce").fillna(0).astype(int)

        # --- tenure_z and interactions ---
        if "tenure" not in df.columns:
            df["tenure"] = np.nan
        t = pd.to_numeric(df["tenure"], errors="coerce")
        sd = float(np.nanstd(t.to_numpy(dtype=float))) if len(t) else 0.0
        mu = float(np.nanmean(t.to_numpy(dtype=float))) if len(t) else 0.0
        if not np.isfinite(sd) or sd <= 1e-12:
            df["tenure_z"] = 0.0
        else:
            df["tenure_z"] = (t - mu) / sd

        df["tenure_x_usage_drop_flag"] = df["tenure_z"] * df["usage_drop_flag"].astype(float)
        df["tenure_x_tickets_opened_qoq_delta_z"] = df["tenure_z"] * df["tickets_opened_qoq_delta_z"].astype(float)

        # --- categorical ---
        if "plan_tier" not in df.columns:
            df["plan_tier"] = "Unknown"
        df["plan_tier"] = df["plan_tier"].astype(str)

        return df


# =============================================================================
# Config
# =============================================================================
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


def bucket_quarter_order(series: pd.Series) -> pd.Series:
    return pd.PeriodIndex(series.astype(str), freq="Q").astype(str)


def risk_tier_from_prob(p: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(p >= 0.50, "High", np.where(p >= 0.35, "Medium", "Low")),
        index=p.index,
    )


def driver_category(feature_name: str) -> str:
    driver_map = {
        "usage_count_qoq_delta_z": "Usage decline",
        "tickets_opened_qoq_delta_z": "Support pressure",
        "arr_amount_qoq_delta_z": "Commercial contraction",
        "days_to_contract_end_capped": "Contract risk",
    }
    return driver_map.get(feature_name, "Other")


def base_feature_name(feature_name: str) -> str:
    if "__" in feature_name:
        return feature_name.split("__", 1)[1]
    return feature_name


def action_from_driver(driver: str) -> str:
    if driver == "Usage decline":
        return "Usage re-engagement / training"
    if driver == "Support pressure":
        return "Proactive support / escalation review"
    if driver == "Contract risk":
        return "Renewal outreach"
    if driver == "Commercial contraction":
        return "Commercial check-in"
    return "General retention check-in"


def build_demo_output(latest_df: pd.DataFrame) -> pd.DataFrame:
    def predicted_churn_quarter_from_row(row: pd.Series) -> str:
        if "quarter" in row and pd.notna(row["quarter"]):
            return str(pd.Period(row["quarter"], freq="Q") + 1)
        if "quarter_end" in row and pd.notna(row["quarter_end"]):
            return str(pd.Period(pd.to_datetime(row["quarter_end"]), freq="Q") + 1)
        return ""

    required_demo_cols = ["risk_type", "top_drivers", "recommended_action", "risk_tier", "churn_probability_q1"]
    missing_demo_cols = [c for c in required_demo_cols if c not in latest_df.columns]
    if missing_demo_cols:
        raise ValueError(f"Missing required demo columns: {missing_demo_cols}")

    records = []
    for _, row in latest_df.iterrows():
        records.append(
            {
                "account_id": row["account_id"],
                "churn_probability_q1": row["churn_probability_q1"],
                "churn_prob_q1": row.get("churn_prob_q1"),
                "churn_prob_q2": row.get("churn_prob_q2"),
                "churn_prob_q3": row.get("churn_prob_q3"),
                "churn_prob_q4": row.get("churn_prob_q4"),
                "churn_date_window": row.get("churn_date_window"),
                "risk_tier": row["risk_tier"],
                "risk_type": row["risk_type"],
                "predicted_churn_quarter": predicted_churn_quarter_from_row(row),
                "key_risks": row["top_drivers"],
                "recommended_actions": row["recommended_action"],
            }
        )
    return pd.DataFrame(records)


def _parse_class_weight(value: str):
    if value == "balanced":
        return "balanced"
    return literal_eval(value)


def _rank_metrics(y_true: pd.Series, probs: np.ndarray) -> dict:
    n_valid = int(y_true.shape[0])
    if n_valid == 0:
        return {
            "precision_top5": None,
            "recall_top5": None,
            "precision_top10": None,
            "recall_top20": None,
            "lift_top10": None,
            "best_precision_at_k": None,
            "best_precision_k": None,
            "best_recall_at_k": None,
            "best_recall_k": None,
        }

    top_5_n = int(np.ceil(0.05 * n_valid))
    top_10_n = int(np.ceil(0.10 * n_valid))
    top_20_n = int(np.ceil(0.20 * n_valid))
    sorted_idx = np.argsort(-probs)
    y_sorted = y_true.iloc[sorted_idx].reset_index(drop=True)
    total_positives = int(y_sorted.sum())
    cumulative_tp = y_sorted.cumsum()
    k_range = np.arange(1, n_valid + 1)
    precision_at_k = cumulative_tp / k_range
    recall_at_k = cumulative_tp / total_positives if total_positives > 0 else pd.Series([0] * n_valid)

    precision_top5 = float(precision_at_k.iloc[top_5_n - 1]) if top_5_n else None
    recall_top5 = float(recall_at_k.iloc[top_5_n - 1]) if top_5_n else None
    precision_top10 = float(precision_at_k.iloc[top_10_n - 1]) if top_10_n else None
    recall_top20 = float(recall_at_k.iloc[top_20_n - 1]) if top_20_n else None
    base_rate = float(y_true.mean()) if n_valid else None
    lift_top10 = (precision_top10 / base_rate) if base_rate else None

    best_precision_value = float(precision_at_k.max()) if n_valid else None
    best_precision_k = int(precision_at_k.idxmax() + 1) if n_valid else None
    best_recall_value = float(recall_at_k.max()) if n_valid else None
    best_recall_k = int(recall_at_k.idxmax() + 1) if n_valid else None

    return {
        "precision_top5": precision_top5,
        "recall_top5": recall_top5,
        "precision_top10": precision_top10,
        "recall_top20": recall_top20,
        "lift_top10": lift_top10,
        "best_precision_at_k": best_precision_value,
        "best_precision_k": best_precision_k,
        "best_recall_at_k": best_recall_value,
        "best_recall_k": best_recall_k,
    }


def _ensure_label(df: pd.DataFrame, label_col: str = "churn_label_q1_rolling") -> pd.DataFrame:
    """
    Ensure df has churn_label_q1_rolling.
    If missing, derive it from other churn event columns when possible.
    """
    df = df.copy()
    # normalize col names (防止有空格/奇怪字符)
    df.columns = [str(c).strip() for c in df.columns]

    if label_col in df.columns:
        return df

    # Candidates in priority order
    if "account_churn_event_in_next_quarter" in df.columns:
        df[label_col] = pd.to_numeric(df["account_churn_event_in_next_quarter"], errors="coerce")
        df[label_col] = df[label_col].where(df[label_col].isin([0, 1]), np.nan)
        print(f"[label] Created {label_col} from account_churn_event_in_next_quarter")
        return df

    if "account_churn_event_count_in_next_quarter" in df.columns:
        x = pd.to_numeric(df["account_churn_event_count_in_next_quarter"], errors="coerce")
        df[label_col] = np.where(x.isna(), np.nan, (x > 0).astype(int))
        print(f"[label] Created {label_col} from account_churn_event_count_in_next_quarter (>0 => 1)")
        return df

    # If still missing, fail with actionable message
    churn_like = [c for c in df.columns if "churn" in c.lower()]
    raise ValueError(
        f"Label column not found: {label_col}. "
        f"Also could not derive it from known alternatives.\n"
        f"Churn-like columns present: {churn_like}\n"
        f"Fix: run your label/flag pipeline (e.g., add_contract_flags.py) "
        f"or include account_churn_event_in_next_quarter in the training CSV."
    )


def _clean_label_binary(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Robust label cleanup:
    - coercing to numeric
    - keep NaN (right-censored / unobservable)
    - keep only 0/1 or NaN
    """
    df = df.copy()
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    keep_mask = df[label_col].isin([0, 1]) | df[label_col].isna()
    df = df[keep_mask].copy()
    return df


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    input_path = base_dir / "output" / "account_quarter_panel_qoq_processed_with_contract_flags.csv"
    out_dir = base_dir / "output" / "model_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Training data:", input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {input_path}")

    # Default best hyperparams (fallback)
    best_row = {"C": 0.05, "class_weight": "{0: 1, 1: 2}", "penalty": "elasticnet", "l1_ratio": 0.2}

    tuning_path = out_dir / "tuning_nonzero_results.csv"
    if tuning_path.exists():
        tuning_df = pd.read_csv(tuning_path)
        if not tuning_df.empty:
            best_row = (
                tuning_df.sort_values(["auc", "zero_rate"], ascending=[False, True])
                .iloc[0]
                .to_dict()
            )

    df = pd.read_csv(input_path)
    print("Columns:", len(df.columns))

    # --- Ensure label exists (this is the fix for your KeyError) ---
    label_col = "churn_label_q1_rolling"
    df = _ensure_label(df, label_col=label_col)

    # --- Minimum required INPUT columns (before feature engineering) ---
    # （missing flags / ending_soon 等等后面 build_feature_set 会补，不在这里卡死）
    required_input_cols = [
        "account_id",
        "signup_date",
        "quarter_end",
        "plan_tier",
        "usage_count_qoq_delta_z",
        "tickets_opened_qoq_delta_z",
        "arr_amount_qoq_delta_z",
        "seats_qoq_delta_z",
        "seats",
        "avg_satisfaction",
        # contract base:
        # allow missing days_to_contract_end_asof_qend; build_feature_set will create from fallbacks but we try to have it
    ]
    missing_input = [c for c in required_input_cols if c not in df.columns]
    if missing_input:
        raise ValueError(
            f"Missing required input columns (pre-feature-engineering): {missing_input}\n"
            f"Tip: check your upstream processing output CSV."
        )

    # --- Feature engineering ---
    df = _clean_label_binary(df, label_col)
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    df["quarter_end"] = pd.to_datetime(df["quarter_end"], errors="coerce")
    df["account_id"] = df["account_id"].astype(str)
    df["plan_tier"] = df["plan_tier"].astype(str)

    df = add_tenure_quarters(df)

    feat_df = build_feature_set(df)  # shared 版本会只返回 FEATURES
    # 把 FEATURES 写回原 df，保留 label 和其他 meta 列
    for c in FEATURES:
        df[c] = feat_df[c]

    # IMPORTANT: use shared FEATURES ordering if available
    features = list(FEATURES)

    # --- Ensure all modeling features exist now ---
    missing_feats = [c for c in features if c not in df.columns]
    if missing_feats:
        raise ValueError(
            f"After feature engineering, still missing model FEATURES: {missing_feats}\n"
            f"Fix: align build_feature_set() / raw_builders outputs with FEATURES list."
        )

    df_model = df[df[label_col].notna()].copy()
    if df_model.empty:
        raise ValueError("No non-NaN labels available for training (all labels are NaN).")

    y = df_model[label_col].astype(int)
    X = df_model[features].copy()

    # --- Column groups ---
    binary_cols = [
        "satisfaction_missing_flag",
        "contract_missing_flag",
        "usage_drop_flag",
        "contract_ending_soon_flag",
    ]
    categorical_cols = ["plan_tier"]
    continuous_cols = [c for c in features if c not in binary_cols + categorical_cols]

    # --- Transformers ---
    num_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    bin_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, continuous_cols),
            ("bin", bin_transformer, binary_cols),
            ("cat", cat_transformer, categorical_cols),
        ]
    )

    # --- Time-based split (last 2 quarters validation) ---
    if "quarter" in df_model.columns and df_model["quarter"].notna().any():
        df_model["quarter_order"] = bucket_quarter_order(df_model["quarter"])
    else:
        df_model["quarter_order"] = pd.to_datetime(df_model["quarter_end"]).dt.to_period("Q").astype(str)

    ordered_quarters = sorted(df_model["quarter_order"].dropna().unique())
    if len(ordered_quarters) < 2:
        raise ValueError("Need at least 2 quarters for time-based split.")
    valid_quarters = ordered_quarters[-2:]

    train_idx = ~df_model["quarter_order"].isin(valid_quarters)
    valid_idx = df_model["quarter_order"].isin(valid_quarters)

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_valid, y_valid = X.loc[valid_idx], y.loc[valid_idx]

    model = LogisticRegression(
        class_weight=_parse_class_weight(str(best_row["class_weight"])),
        penalty=str(best_row["penalty"]),
        C=float(best_row["C"]),
        l1_ratio=None if pd.isna(best_row.get("l1_ratio")) else float(best_row["l1_ratio"]),
        max_iter=3000,
        solver="saga",
    )
    pipeline = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    valid_proba = pipeline.predict_proba(X_valid)[:, 1]
    valid_pred = (valid_proba >= 0.5).astype(int)

    metrics = {
        "validation_auc": float(roc_auc_score(y_valid, valid_proba)) if y_valid.nunique() > 1 else None,
        "validation_precision_at_0_5": float(precision_score(y_valid, valid_pred, zero_division=0)),
        "validation_recall_at_0_5": float(recall_score(y_valid, valid_pred, zero_division=0)),
        "best_c": float(best_row["C"]),
        "class_weight": str(best_row["class_weight"]),
        "penalty": str(best_row["penalty"]),
        "l1_ratio": None if pd.isna(best_row.get("l1_ratio")) else float(best_row["l1_ratio"]),
        "using_shared_feature_module": bool(_HAS_SHARED),
    }
    metrics.update(_rank_metrics(y_valid, valid_proba))
    metrics_path = out_dir / "best_auc_model_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    # --- Feature importance (abs coef) ---
    feature_names = preprocessor.get_feature_names_out()
    coef = pipeline.named_steps["model"].coef_[0]
    coef_table = pd.DataFrame({"feature": feature_names, "coefficient": coef, "abs_coefficient": np.abs(coef)})
    coef_table["base_feature"] = coef_table["feature"].map(base_feature_name)
    coef_table = coef_table.sort_values("abs_coefficient", ascending=False)
    coef_table.to_csv(out_dir / "best_auc_feature_importance.csv", index=False)

    plt.figure(figsize=(9.5, 5.5))
    plot_df = coef_table.head(12).sort_values("abs_coefficient", ascending=True)
    plt.barh(plot_df["feature"], plot_df["abs_coefficient"])
    plt.xlabel("Absolute standardized coefficient")
    plt.ylabel("")
    plt.title("Feature Importance (Best AUC Model)")
    plt.grid(axis="x", linestyle="--", alpha=0.25)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / "best_auc_feature_importance.png", dpi=220)
    plt.close()

    # --- Latest quarter outputs ---
    latest_idx = df.groupby("account_id")["quarter_end"].transform("max") == df["quarter_end"]
    latest_df = df.loc[latest_idx].copy()

    latest_X = latest_df[features]
    latest_proba = pipeline.predict_proba(latest_X)[:, 1]
    latest_df["churn_probability_q1"] = latest_proba
    latest_df["risk_score"] = (latest_df["churn_probability_q1"] * 100).round(0).astype(int)
    latest_df["risk_tier"] = risk_tier_from_prob(latest_df["churn_probability_q1"])

    # multi-quarter mapping
    latest_df["churn_prob_q1"] = latest_df["churn_probability_q1"]
    latest_df["churn_prob_q2"] = 1 - (1 - latest_df["churn_probability_q1"]) ** 2
    latest_df["churn_prob_q3"] = 1 - (1 - latest_df["churn_probability_q1"]) ** 3
    latest_df["churn_prob_q4"] = 1 - (1 - latest_df["churn_probability_q1"]) ** 4
    latest_df["churn_prob_within_2q"] = 1 - (1 - latest_df["churn_probability_q1"]) ** 2
    latest_df["churn_prob_within_4q"] = 1 - (1 - latest_df["churn_probability_q1"]) ** 4

    def churn_date_window_from_probs(row: pd.Series) -> str:
        if row["churn_prob_q1"] >= 0.35:
            return "0-90 days"
        if row["churn_prob_q2"] >= 0.35:
            return "3-6 months"
        if row["churn_prob_q3"] >= 0.35:
            return "6-12 months"
        if row["churn_prob_q4"] >= 0.35:
            return "> 365 days"
        return "N/A"

    latest_df["churn_date_window"] = latest_df.apply(churn_date_window_from_probs, axis=1)

    # Driver extraction (primary predictors only)
    transformed_latest = pipeline.named_steps["prep"].transform(latest_X)
    contrib = transformed_latest * coef
    transformed_feature_names = pipeline.named_steps["prep"].get_feature_names_out()
    primary_set = set(PRIMARY_PREDICTORS)

    top_driver_labels = []
    top_actions = []
    for row in contrib:
        driver_scores: dict[str, float] = {}
        for fname, contribution in zip(transformed_feature_names, row):
            base = base_feature_name(fname)
            if base not in primary_set:
                continue
            cat = driver_category(base)
            driver_scores[cat] = driver_scores.get(cat, 0.0) + float(contribution)

        ranked = sorted(driver_scores, key=driver_scores.get, reverse=True)[:3]
        actions = [action_from_driver(c) for c in ranked[:2]]
        top_driver_labels.append(", ".join(ranked))
        top_actions.append(" / ".join(actions))

    latest_df["top_drivers"] = top_driver_labels
    latest_df["recommended_action"] = top_actions

    # Risk type assignment by family contribution
    feature_to_risk_type = {feat: rt for rt, feats in RISK_TYPE_MAP.items() for feat in feats}
    risk_types = []
    for row in contrib:
        family_scores = {rt: 0.0 for rt in RISK_TYPE_MAP}
        for fname, contribution in zip(transformed_feature_names, row):
            base = base_feature_name(fname)
            rt = feature_to_risk_type.get(base)
            if rt is None:
                continue
            family_scores[rt] += abs(float(contribution))
        risk_types.append(max(family_scores, key=family_scores.get))
    latest_df["risk_type"] = risk_types

    if "quarter" in latest_df.columns and latest_df["quarter"].notna().any():
        latest_df["predicted_churn_quarter"] = (pd.PeriodIndex(latest_df["quarter"], freq="Q") + 1).astype(str)
    else:
        latest_df["predicted_churn_quarter"] = (pd.PeriodIndex(pd.to_datetime(latest_df["quarter_end"]), freq="Q") + 1).astype(str)

    out_dir.mkdir(parents=True, exist_ok=True)
    output_cols = [
        "account_id",
        "churn_probability_q1",
        "churn_prob_q1",
        "churn_prob_q2",
        "churn_prob_q3",
        "churn_prob_q4",
        "churn_date_window",
        "churn_prob_within_2q",
        "churn_prob_within_4q",
        "risk_score",
        "risk_tier",
        "risk_type",
        "predicted_churn_quarter",
        "top_drivers",
        "recommended_action",
    ]
    latest_df[output_cols].to_csv(out_dir / "best_auc_account_latest_risk_output.csv", index=False)

    demo_output = build_demo_output(latest_df)
    demo_output.to_csv(base_dir / "output" / "best_auc_demo_risk_output.csv", index=False)

    dump(pipeline, out_dir / "best_auc_model.joblib")

    print("Saved:", metrics_path)
    print("Saved:", out_dir / "best_auc_feature_importance.png")
    print("Saved:", out_dir / "best_auc_model.joblib")
    print("Saved:", out_dir / "best_auc_account_latest_risk_output.csv")
    print("Saved:", base_dir / "output" / "best_auc_demo_risk_output.csv")
    print("Using shared feature module:", _HAS_SHARED)


if __name__ == "__main__":
    main()
