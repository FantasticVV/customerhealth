# streamlit_app/app.py
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

from raw_builders import build_from_single_raw_current_prev
from scorer import score


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from raw_builders import build_from_single_raw_current_prev, build_from_raw_multi
from scorer import score



# ----------------- Page config -----------------
st.set_page_config(page_title="Customer Health • Risk Scoring", layout="wide")

# ----------------- Light UI CSS (module cards) -----------------
st.markdown(
    """
    <style>
      :root{
        --bg0:#f4f7fb;
        --bg1:#eef3ff;
        --card:#ffffff;
        --stroke: rgba(15, 23, 42, .10);
        --text: rgba(15, 23, 42, .92);
        --muted: rgba(15, 23, 42, .62);
        --muted2: rgba(15, 23, 42, .50);
        --shadow: 0 14px 40px rgba(15, 23, 42, .10);
        --shadow2: 0 6px 16px rgba(15, 23, 42, .08);
        --radius: 18px;

        --primary:#2563eb;
        --green:#22c55e;
        --amber:#f59e0b;
        --red:#ef4444;
      }

      header[data-testid="stHeader"]{
        background: transparent !important;
        box-shadow: none !important;
      }
      div[data-testid="stToolbar"]{
        background: transparent !important;
      }
      div[data-testid="stDecoration"]{
        display: none !important;
        height: 0 !important;
      }
      div[data-testid="stStatusWidget"]{
        display: none !important;
        height: 0 !important;
      }

      .stApp{
        background:
          radial-gradient(900px 450px at 15% 8%, rgba(37,99,235,.18), transparent 55%),
          radial-gradient(900px 500px at 85% 18%, rgba(34,197,94,.14), transparent 55%),
          linear-gradient(180deg, var(--bg1) 0%, var(--bg0) 45%, #f7fafc 100%);
      }

      .block-container{
        padding-top: 1.2rem;
        max-width: 1500px;
      }

      h1,h2,h3,h4,h5,h6,p,div,span,label{ color: var(--text); }
      .subtle{ color: var(--muted); font-size:12px; }
      .micro{ color: var(--muted2); font-size:11px; }

      section[data-testid="stSidebar"]{
        background: rgba(255,255,255,.95) !important;
        border-right: 1px solid var(--stroke) !important;
        box-shadow: 8px 0 24px rgba(15, 23, 42, .05);
      }
      section[data-testid="stSidebar"] *{
        color: var(--text) !important;
      }

      .card{
        background: var(--card);
        border: 1px solid var(--stroke);
        border-radius: var(--radius);
        padding: 14px 16px;
        box-shadow: var(--shadow2);
        margin-bottom: 12px;
      }

      .hero{
        border-radius: 22px;
        padding: 18px 18px 14px 18px;
        border: 1px solid var(--stroke);
        background:
          radial-gradient(700px 260px at 18% 20%, rgba(37,99,235,.12), transparent 55%),
          radial-gradient(650px 240px at 75% 0%, rgba(34,197,94,.10), transparent 55%),
          linear-gradient(180deg, #ffffff, rgba(255,255,255,.88));
        box-shadow: var(--shadow);
      }
      .hero-title{
        font-weight: 950;
        letter-spacing: -0.8px;
        font-size: 34px;
        line-height: 1.08;
        margin-bottom: 6px;
      }
      .hero-sub{
        color: var(--muted);
        font-size: 13px;
        margin-bottom: 10px;
      }

      .pill{
        display:inline-flex;
        gap:8px;
        align-items:center;
        padding:6px 10px;
        border: 1px solid var(--stroke);
        border-radius: 999px;
        background: rgba(37,99,235,.06);
        color: rgba(37,99,235,.92);
        font-size: 12px;
        font-weight: 800;
      }
      .pill.gray{
        background: rgba(15,23,42,.04);
        color: rgba(15,23,42,.80);
      }

      .divider{
        height: 1px;
        background: var(--stroke);
        margin: 12px 0;
      }

      .kpi-title{ font-size:12px; color: var(--muted); margin-bottom: 4px; }
      .kpi-value{ font-size: 26px; font-weight: 950; letter-spacing: -0.4px; }
      .kpi-note{ font-size: 12px; color: var(--muted2); margin-top: 2px; }

      div[data-testid="stFileUploaderDropzone"]{
        background: rgba(37,99,235,.04) !important;
        border: 1px dashed rgba(37,99,235,.35) !important;
        border-radius: 16px !important;
      }

      .stButton>button, .stDownloadButton>button{
        background: rgba(37,99,235,.92) !important;
        border: 1px solid rgba(37,99,235,.95) !important;
        border-radius: 14px !important;
        color: white !important;
        box-shadow: 0 10px 18px rgba(37,99,235,.18) !important;
      }
      .stButton>button:hover, .stDownloadButton>button:hover{
        background: rgba(37,99,235,1) !important;
      }

      div[data-testid="stDataFrame"]{
        background: rgba(255,255,255,.98) !important;
        border-radius: 16px !important;
        border: 1px solid var(--stroke) !important;
        padding: 8px !important;
        box-shadow: var(--shadow2);
      }

      details{
        border-radius: 16px !important;
        border: 1px solid var(--stroke) !important;
        background: rgba(255,255,255,.95) !important;
        box-shadow: var(--shadow2);
      }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- Helpers -----------------
def pct(x, digits=1):
    try:
        return f"{float(x) * 100:.{digits}f}%"
    except Exception:
        return "N/A"

def safe_list_from_pipe(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    txt = str(s).strip()
    if not txt:
        return []
    return [t.strip() for t in txt.split("|") if t.strip()]

def primary_token(s):
    toks = safe_list_from_pipe(s)
    return toks[0] if toks else ""

def tier_order_value(tier: str) -> int:
    return {"High": 0, "Medium": 1, "Low": 2}.get(str(tier), 9)

def tier_badge(tier: str) -> str:
    if str(tier) == "High":
        return "🔴 High"
    if str(tier) == "Medium":
        return "🟠 Medium"
    return "🟢 Low"

def kpi_card(title, value, note=""):
    st.markdown(
        f"""
        <div class="card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def donut_chart(values, labels, title=""):
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=140)
    ax.pie(values, startangle=90, wedgeprops=dict(width=0.38, edgecolor="white"))
    ax.set(aspect="equal")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.legend(labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    return fig

def bar_chart_series(series, title=""):
    fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=140)
    ax.bar(series.index.astype(str), series.values)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    return fig

def hist_chart(values, bins=20, title=""):
    fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=140)
    ax.hist(values, bins=bins, range=(0, 1))
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("risk_probability")
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.25)
    return fig

def coverage_summary_multi(accounts, subs, usage, tickets):
    if accounts is None or subs is None:
        return None
    n_acc = len(accounts)
    n_sub = len(subs)
    usage_cov = None
    ticket_cov = None

    if usage is not None and "subscription_id" in usage.columns and "subscription_id" in subs.columns:
        sub_with_usage = usage["subscription_id"].nunique()
        usage_cov = sub_with_usage / max(1, subs["subscription_id"].nunique())

    if tickets is not None and "account_id" in tickets.columns and "account_id" in accounts.columns:
        acc_with_tickets = tickets["account_id"].nunique()
        ticket_cov = acc_with_tickets / max(1, accounts["account_id"].nunique())

    return {
        "Accounts": n_acc,
        "Subscriptions": n_sub,
        "Usage coverage (subs)": usage_cov,
        "Ticket coverage (accounts)": ticket_cov,
    }

def _safe_datetime(s):
    return pd.to_datetime(s, errors="coerce")

def _to_quarter_end(qstart: pd.Timestamp) -> pd.Timestamp:
    return (pd.Timestamp(qstart).to_period("Q").end_time).normalize()

def _ensure_quarter_col(df: pd.DataFrame, col: str) -> pd.Series:
    s = _safe_datetime(df[col])
    return s.dt.to_period("Q").dt.start_time

def _dedup_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "account_id" in df.columns and "as_of_quarter" in df.columns:
        df["account_id"] = df["account_id"].astype(str)
        df["as_of_quarter"] = _ensure_quarter_col(df, "as_of_quarter")
        df = df.sort_values(["account_id", "as_of_quarter"])
        df = df.drop_duplicates(["account_id", "as_of_quarter"], keep="last")
    return df

def build_from_quarterly_history_new(history_df: pd.DataFrame | None, new_df: pd.DataFrame):
    """
    Quarterly mode:
      - new_df is required and should contain ONE quarter.
      - history_df is optional:
          * if provided: prev quarter + baseline medians added => scorer uses self-baseline explanations
          * if not: prev quarter defaults to 0 + no baseline => scorer falls back to peer explanations
    Returns:
      feats2: features + enrich columns merged
      enrich: enrich-only table
    """
    if new_df is None:
        raise ValueError("New quarterly CSV is required for this mode.")

    n = _dedup_history(new_df)

    if "account_id" not in n.columns or "as_of_quarter" not in n.columns:
        raise ValueError("Quarterly CSV must include: account_id, as_of_quarter")

    n["account_id"] = n["account_id"].astype(str)
    n["as_of_quarter"] = _ensure_quarter_col(n, "as_of_quarter")

    uq = sorted(n["as_of_quarter"].dropna().unique().tolist())
    if len(uq) > 1:
        raise ValueError("New quarterly upload should contain a single quarter (one as_of_quarter).")

    # prev quarter per account from history (if available)
    if history_df is not None and len(history_df) > 0:
        h = _dedup_history(history_df)

        if "account_id" not in h.columns or "as_of_quarter" not in h.columns:
            # if someone gives bad history, ignore it instead of breaking demo
            h = None

        if h is not None:
            h["account_id"] = h["account_id"].astype(str)
            h["as_of_quarter"] = _ensure_quarter_col(h, "as_of_quarter")

            prev = (h.sort_values(["account_id", "as_of_quarter"])
                      .groupby("account_id", as_index=False)
                      .tail(1)
                      .copy())

            merged = n.merge(
                prev[["account_id", "as_of_quarter",
                      "usage_count_total", "ticket_count", "avg_csat",
                      "seats", "arr_amount"]],
                on="account_id",
                how="left",
                suffixes=("", "_prevq")
            )
        else:
            merged = n.copy()
            merged["usage_count_total_prevq"] = 0
            merged["ticket_count_prevq"] = 0
            merged["seats_prevq"] = merged.get("seats", 0)
            merged["arr_amount_prevq"] = merged.get("arr_amount", 0)
    else:
        merged = n.copy()
        merged["usage_count_total_prevq"] = 0
        merged["ticket_count_prevq"] = 0
        merged["seats_prevq"] = merged.get("seats", 0)
        merged["arr_amount_prevq"] = merged.get("arr_amount", 0)

    # Build "single raw current/prev" for the model feature builder
    single = pd.DataFrame({
        "account_id": merged["account_id"].astype(str),
        "plan_tier": merged.get("plan_tier", "Unknown").astype(str),

        "usage_count_current": pd.to_numeric(merged.get("usage_count_total", 0), errors="coerce").fillna(0),
        "usage_count_prev": pd.to_numeric(merged.get("usage_count_total_prevq", 0), errors="coerce").fillna(0),

        "tickets_opened_current": pd.to_numeric(merged.get("ticket_count", 0), errors="coerce").fillna(0),
        "tickets_opened_prev": pd.to_numeric(merged.get("ticket_count_prevq", 0), errors="coerce").fillna(0),

        "days_to_contract_end_current": pd.to_numeric(merged.get("days_to_contract_end", np.nan), errors="coerce"),

        "seats_current": pd.to_numeric(merged.get("seats", 0), errors="coerce").fillna(0),
        "seats_prev": pd.to_numeric(merged.get("seats_prevq", merged.get("seats", 0)), errors="coerce").fillna(0),

        "arr_current": pd.to_numeric(merged.get("arr_amount", 0), errors="coerce").fillna(0),
        "arr_prev": pd.to_numeric(merged.get("arr_amount_prevq", merged.get("arr_amount", 0)), errors="coerce").fillna(0),

        "avg_satisfaction_current": pd.to_numeric(merged.get("avg_csat", np.nan), errors="coerce"),

        "as_of_date": merged["as_of_quarter"].apply(_to_quarter_end),
        "signup_date": _safe_datetime(merged.get("signup_date", pd.NaT)),
    })

    feats = build_from_single_raw_current_prev(single)

    # Build baseline medians (only if history exists)
    if history_df is not None and len(history_df) > 0 and "account_id" in history_df.columns and "as_of_quarter" in history_df.columns:
        h = _dedup_history(history_df)

        hist_meds = (h.groupby("account_id", as_index=False)
                       .agg(
                           hist_usage_median=("usage_count_total", "median"),
                           hist_tickets_median=("ticket_count", "median"),
                           hist_csat_median=("avg_csat", "median"),
                           hist_resolution_median=("avg_resolution_hours", "median"),
                           hist_first_response_median=("avg_first_response_minutes", "median"),
                       ))
    else:
        hist_meds = pd.DataFrame({"account_id": merged["account_id"].unique().tolist()})
        for c in ["hist_usage_median", "hist_tickets_median", "hist_csat_median", "hist_resolution_median", "hist_first_response_median"]:
            hist_meds[c] = np.nan

    enrich = n.merge(hist_meds, on="account_id", how="left")

    enrich_cols = [
        "account_id",
        "as_of_quarter",
        "usage_count_total",
        "ticket_count",
        "avg_csat",
        "avg_resolution_hours",
        "avg_first_response_minutes",
        "escalation_count",
        "auto_renew_flag",
        "days_to_contract_end",
        "hist_usage_median",
        "hist_tickets_median",
        "hist_csat_median",
        "hist_resolution_median",
        "hist_first_response_median",
        "plan_tier",
        "seats",
        "arr_amount",
        "mrr_amount",
    ]
    for c in enrich_cols:
        if c not in enrich.columns:
            enrich[c] = np.nan

    enrich = enrich[enrich_cols].copy()
    enrich["account_id"] = enrich["account_id"].astype(str)

    feats2 = feats.merge(enrich, on="account_id", how="left", suffixes=("", "_enrich"))
    return feats2, enrich


# ----------------- HERO header -----------------
left, right = st.columns([0.72, 0.28], gap="large")
with left:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-title">Customer Health — Risk Scoring</div>
          <div class="hero-sub">Upload raw data → validate coverage → score churn risk → explain drivers → export for CS/AE action</div>
          <span class="pill">✨ Enterprise demo</span>
          <span class="pill gray">🧠 Model-backed scoring</span>
          <span class="pill gray">🧾 Rule-based explanations</span>
        </div>
        """,
        unsafe_allow_html=True
    )
with right:
    st.markdown(
        """
        <div class="card">
          <div style="font-weight:900; font-size:14px; margin-bottom:6px;">Workflow</div>
          <div class="micro">1) Upload → 2) Coverage → 3) Score → 4) Inspect → 5) Export</div>
          <div class="divider"></div>
          <div class="micro">Tip: Start with <b>Overview</b>, then inspect a <b>High-risk</b> account to show drivers and recommended actions.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------- Sidebar controls -----------------
st.sidebar.markdown("## Upload")
mode = st.sidebar.radio(
    "Input format",
    ["Quarterly history + new CSV (self baseline)", "RAW multi-tables", "Single RAW CSV (current/prev)"],
    index=0
)

st.sidebar.markdown("## Tier thresholds")
st.sidebar.caption("Tiers depend on absolute probability.")
medium_th = st.sidebar.slider("Medium threshold", 0.0, 1.0, 0.10, 0.01)
high_th = st.sidebar.slider("High threshold", 0.0, 1.0, 0.40, 0.01)
if high_th < medium_th:
    st.sidebar.error("High threshold should be ≥ Medium threshold.")

st.sidebar.markdown("## What we output")
st.sidebar.markdown(
    "- risk_score (0–100)\n"
    "- risk_probability (0–1)\n"
    "- risk_tier\n"
    "- risk_type\n"
    "- drivers\n"
    "- recommendation"
)

# ----------------- Upload widgets -----------------
feats = None
build_error = None
missing_optional = []

raw_accounts = raw_subs = raw_usage = raw_tickets = None

if "history_df" not in st.session_state:
    st.session_state.history_df = None
if "history_stack" not in st.session_state:
    st.session_state.history_stack = []

if mode == "Quarterly history + new CSV (self baseline)":
    st.sidebar.markdown("### Quarterly CSV uploads")
    hist_f = st.sidebar.file_uploader("History quarterly CSV (optional, but recommended)", type=["csv"], key="hist_q")
    new_f = st.sidebar.file_uploader("New quarterly CSV (required)", type=["csv"], key="new_q")

    if hist_f is not None:
        try:
            hist_df = pd.read_csv(hist_f)
            st.session_state.history_df = _dedup_history(hist_df)
        except Exception as e:
            build_error = e

    new_df = None
    if new_f is not None:
        try:
            new_df = pd.read_csv(new_f)
            new_df = _dedup_history(new_df)
        except Exception as e:
            build_error = e

    st.sidebar.markdown("### History renewal (demo-safe)")
    cA, cB = st.sidebar.columns(2)
    with cA:
        if st.button("Renew (Append)", key="renew_btn"):
            if new_df is None:
                st.sidebar.error("Upload a new quarterly CSV first.")
            else:
                if st.session_state.history_df is None:
                    st.session_state.history_df = new_df.copy()
                else:
                    st.session_state.history_stack.append(st.session_state.history_df.copy())
                    combined = pd.concat([st.session_state.history_df, new_df], ignore_index=True)
                    combined = _dedup_history(combined)
                    st.session_state.history_df = combined

    with cB:
        if st.button("Undo", key="undo_btn"):
            if st.session_state.history_stack:
                st.session_state.history_df = st.session_state.history_stack.pop()

    st.sidebar.caption(
        f"History rows: {0 if st.session_state.history_df is None else len(st.session_state.history_df):,}  "
        f"| Undo steps: {len(st.session_state.history_stack)}"
    )

    if st.session_state.history_df is not None:
        st.sidebar.download_button(
            "Download updated history CSV",
            data=st.session_state.history_df.to_csv(index=False).encode("utf-8"),
            file_name="history_quarterly_updated.csv",
            mime="text/csv",
        )

    if build_error is None and new_df is not None:
        try:
            feats, _enrich = build_from_quarterly_history_new(st.session_state.history_df, new_df)
        except Exception as e:
            build_error = e
    else:
        feats = None

elif mode == "RAW multi-tables":
    st.sidebar.markdown("### Required")
    acc_f = st.sidebar.file_uploader("accounts.csv", type=["csv"])
    sub_f = st.sidebar.file_uploader("subscriptions.csv", type=["csv"])

    st.sidebar.markdown("### Optional")
    use_f = st.sidebar.file_uploader("feature_usage.csv (optional)", type=["csv"])
    tik_f = st.sidebar.file_uploader("support_tickets.csv (optional)", type=["csv"])

    if acc_f and sub_f:
        try:
            raw_accounts = pd.read_csv(acc_f)
            raw_subs = pd.read_csv(sub_f)
            raw_usage = pd.read_csv(use_f) if use_f else None
            raw_tickets = pd.read_csv(tik_f) if tik_f else None

            if use_f is None:
                missing_optional.append("feature_usage.csv (usage-based explanations may be reduced)")
            if tik_f is None:
                missing_optional.append("support_tickets.csv (support-based explanations may be reduced)")

            feats = build_from_raw_multi(raw_accounts, raw_subs, raw_usage, raw_tickets)
        except Exception as e:
            build_error = e

else:
    up = st.sidebar.file_uploader("Upload single RAW CSV", type=["csv"])
    if up:
        try:
            df = pd.read_csv(up)
            feats = build_from_single_raw_current_prev(df)
        except Exception as e:
            build_error = e

# ----------------- Minimum required columns expander -----------------
with st.expander("Minimum required columns (send this to the data team)", expanded=False):
    st.markdown(
        """
**Quarterly history + new CSV (self baseline)**

Required:
- `account_id`, `as_of_quarter` (any date in quarter is OK; we normalize to quarter start)
- `usage_count_total` (quarter total)
- `ticket_count` (quarter count)

Recommended (better explanations):
- `avg_csat`, `avg_resolution_hours`, `avg_first_response_minutes`
- `escalation_count`
- `days_to_contract_end`, `auto_renew_flag`
- `plan_tier`, `seats`, `arr_amount`, `mrr_amount`
- `signup_date`

**RAW multi-tables — minimum required**

✅ Required:
- accounts.csv: `account_id`, `signup_date`
- subscriptions.csv: `subscription_id`, `account_id`, `start_date`, `end_date`, `plan_tier`, `seats`, `arr_amount`

🟡 Optional:
- feature_usage.csv: `subscription_id`, `usage_date`, `usage_count`
- support_tickets.csv: `account_id`, `submitted_at`, `ticket_id` (+ `satisfaction_score` optional)

**Single raw CSV (current/prev)**

- `account_id`, `signup_date`, `as_of_date`, `plan_tier`
- `usage_count_current`, `usage_count_prev`
- `tickets_opened_current`, `tickets_opened_prev`
- `days_to_contract_end_current`
- `seats_current` (prev optional), `arr_current` (prev optional)
- `avg_satisfaction_current` (optional)
"""
    )

# ----------------- Build guardrails -----------------
if build_error is not None:
    st.error("Failed to build features from uploaded data.")
    st.exception(build_error)
    st.stop()

if feats is None:
    st.markdown("<div class='card'>Upload files in the left sidebar to start scoring.</div>", unsafe_allow_html=True)
    st.stop()

if missing_optional:
    st.warning("Optional tables missing — scoring will still run, but some explanations may be reduced:\n- " + "\n- ".join(missing_optional))

# ----------------- Score -----------------
try:
    out = score(feats, high_threshold=high_th, medium_threshold=medium_th)
except Exception as e:
    st.error("Scoring failed.")
    st.exception(e)
    st.stop()

out = out.copy()

prob_unique = out["risk_probability"].nunique()
prob_std = float(out["risk_probability"].std()) if len(out) else 0.0
if prob_unique <= 3 or prob_std < 1e-4:
    st.warning(
        "Risk probabilities look nearly identical across rows. "
        "This usually means key inputs are missing or constant (e.g., usage/tickets/contract fields)."
    )
    if feats is not None:
        check_cols = [
            "usage_count_qoq_delta_z",
            "tickets_opened_qoq_delta_z",
            "usage_drop_flag",
            "days_to_contract_end_capped",
            "arr_amount_qoq_delta_z",
            "seats_qoq_delta_z",
            "seats",
            "tenure",
            "avg_satisfaction",
        ]
        present = [c for c in check_cols if c in feats.columns]
        if present:
            stats = feats[present].agg(["nunique", "std"]).T
            stats = stats.rename(columns={"nunique": "unique_values", "std": "std_dev"})
            stats = stats.sort_values(["unique_values", "std_dev"])
            st.dataframe(stats, use_container_width=True)

out["risk_probability_pct"] = out["risk_probability"].apply(lambda x: pct(x, 1))
out["primary_driver"] = out["drivers"].apply(primary_token)

# ✅ recommendation is not pipe-delimited; show it directly as "primary_reco"
out["primary_reco"] = out["recommendation"].astype(str)

out["_tier_order"] = out["risk_tier"].apply(tier_order_value)
out["risk_badge"] = out["risk_tier"].apply(tier_badge)

# ----------------- Tabs -----------------
tab_overview, tab_table, tab_dist, tab_drivers, tab_export = st.tabs(
    ["✨ Overview", "📋 Table", "📈 Distribution", "🧩 Drivers", "⬇️ Export"]
)

# ===================== OVERVIEW =====================
with tab_overview:
    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        kpi_card("Rows scored", f"{len(out):,}", "Accounts/subscriptions scored")
    with c2:
        kpi_card("Avg probability", pct(out["risk_probability"].mean(), 1), "Mean churn risk")
    with c3:
        kpi_card("High risk", f"{(out['risk_tier']=='High').sum():,}", f"Tier ≥ {high_th:.2f}")
    with c4:
        kpi_card("Medium risk", f"{(out['risk_tier']=='Medium').sum():,}", f"{medium_th:.2f} ≤ Tier < {high_th:.2f}")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    v1, v2, v3 = st.columns([0.30, 0.40, 0.30], gap="large")

    with v1:
        tier_counts = out["risk_tier"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).astype(int)
        labels = [f"High ({tier_counts['High']})", f"Medium ({tier_counts['Medium']})", f"Low ({tier_counts['Low']})"]
        fig = donut_chart(tier_counts.values.tolist(), labels, title="Risk mix")
        st.pyplot(fig, use_container_width=True)

    with v2:
        fig = hist_chart(out["risk_probability"].values, bins=18, title="Probability distribution")
        st.pyplot(fig, use_container_width=True)

    with v3:
        st.markdown("**Coverage / data quality**")
        if mode == "RAW multi-tables":
            cov = coverage_summary_multi(raw_accounts, raw_subs, raw_usage, raw_tickets)
            if cov:
                a, b = st.columns(2)
                a.metric("Accounts", f"{cov['Accounts']:,}")
                b.metric("Subscriptions", f"{cov['Subscriptions']:,}")

                u = cov["Usage coverage (subs)"]
                t = cov["Ticket coverage (accounts)"]
                st.metric("Usage coverage", "N/A" if u is None else f"{u*100:.0f}%")
                st.metric("Ticket coverage", "N/A" if t is None else f"{t*100:.0f}%")
                st.caption("Missing usage/tickets reduces rule-based explanations, but scoring still runs.")
        elif mode == "Quarterly history + new CSV (self baseline)":
            if st.session_state.history_df is None:
                st.caption("Quarterly mode — no history provided. Explanations fall back to peer-in-upload comparisons.")
            else:
                st.caption("Quarterly mode — drivers use each account's own historical baseline when history is provided.")
        else:
            st.caption("Single raw mode — coverage is implied by the required columns.")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown("### 🔥 Top highest risk")
    top20 = out.sort_values(["_tier_order", "risk_probability"], ascending=[True, False]).head(20).copy()
    show_cols = ["account_id", "risk_badge", "risk_probability_pct", "risk_type", "primary_driver", "primary_reco"]
    st.dataframe(top20[show_cols], use_container_width=True, hide_index=True)

    st.markdown("### 🔍 Inspect an account")

    # ✅ if top20 is empty (edge case), fall back to whole table
    pick_list = top20["account_id"].tolist() if len(top20) else out["account_id"].astype(str).tolist()
    if not pick_list:
        st.info("No rows to inspect.")
    else:
        pick = st.selectbox("Select account_id", pick_list, index=0)
        row = out[out["account_id"].astype(str) == str(pick)].iloc[0]

        summary_html = (
            f"<div class='card'>"
            f"<b>{row['account_id']}</b><br/>"
            f"<span class='subtle'>Tier: <b>{row['risk_tier']}</b> • "
            f"Probability: <b>{pct(row['risk_probability'], 1)}</b> • "
            f"Type: <b>{row['risk_type']}</b> • "
            f"Score: <b>{row['risk_score']}</b></span>"
            f"</div>"
        )
        st.markdown(summary_html, unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        dcol, rcol = st.columns([0.55, 0.45], gap="large")
        with dcol:
            dlist = safe_list_from_pipe(row.get("drivers", ""))
            if dlist:
                drivers_items = "".join(f"<li>{d}</li>" for d in dlist)
                drivers_body = f"<ul>{drivers_items}</ul>"
            else:
                drivers_body = "<div class='micro'>No drivers detected for this row.</div>"
            drivers_html = (
                "<div class='card'>"
                "<b>Drivers</b>"
                f"{drivers_body}"
                "</div>"
            )
            st.markdown(drivers_html, unsafe_allow_html=True)

        with rcol:
            # recommendation is a sentence; show as one block
            reco_txt = str(row.get("recommendation", "")).strip()
            if reco_txt:
                reco_body = f"<div class='micro'>{reco_txt}</div>"
            else:
                reco_body = "<div class='micro'>No recommendation available for this row.</div>"
            reco_html = (
                "<div class='card'>"
                "<b>Recommendation</b>"
                f"{reco_body}"
                "</div>"
            )
            st.markdown(reco_html, unsafe_allow_html=True)

# ===================== TABLE =====================
with tab_table:
    st.markdown("### 📋 Full scored table")
    st.caption("Demo-friendly default: show High + Medium first. You can include Low if needed.")

    f1, f2, f3 = st.columns([0.34, 0.33, 0.33], gap="large")
    with f1:
        tier_sel = st.multiselect("Risk tier", ["High", "Medium", "Low"], default=["High", "Medium"])
    with f2:
        min_prob = st.slider("Min probability", 0.0, 1.0, 0.0, 0.01)
    with f3:
        rt_vals = sorted(out["risk_type"].astype(str).unique().tolist())
        rt_sel = st.multiselect("Risk type", rt_vals, default=rt_vals)

    view = out.copy()
    view = view[view["risk_tier"].isin(tier_sel)]
    view = view[view["risk_probability"] >= min_prob]
    view = view[view["risk_type"].astype(str).isin(rt_sel)]

    table_cols = ["account_id", "risk_score", "risk_probability_pct", "risk_badge", "risk_type", "primary_driver", "primary_reco"]
    view2 = view.sort_values(["_tier_order", "risk_probability"], ascending=[True, False])[table_cols].copy()
    st.dataframe(view2, use_container_width=True, hide_index=True)

# ===================== DISTRIBUTION =====================
with tab_dist:
    st.markdown("### 📈 Distribution")
    a, b = st.columns([0.5, 0.5], gap="large")

    with a:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        tier_counts = out["risk_tier"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).astype(int)
        fig = bar_chart_series(tier_counts, title="Tier counts")
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        fig = hist_chart(out["risk_probability"].values, bins=24, title="Probability histogram")
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ===================== DRIVERS =====================
with tab_drivers:
    st.markdown("### 🧩 Drivers (High + Medium)")
    st.caption("Counts drivers across High/Medium accounts to keep the story focused.")

    focus = out[out["risk_tier"].isin(["High", "Medium"])].copy()
    dd = focus[["drivers"]].dropna()

    if len(dd) == 0:
        st.info("No drivers detected (this can happen if optional signals are missing).")
    else:
        tokens = dd["drivers"].astype(str).str.split(r"\s*\|\s*").explode().str.strip()
        tokens = tokens[tokens.astype(str).str.len() > 0]
        top = tokens.value_counts().head(12)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        fig = bar_chart_series(top, title="Top drivers across higher-risk accounts")
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Example rows (High/Medium first)")
    ex_cols = ["account_id", "risk_probability_pct", "risk_badge", "risk_type", "drivers", "recommendation"]
    st.dataframe(
        out.sort_values(["_tier_order", "risk_probability"], ascending=[True, False])[ex_cols].head(40),
        use_container_width=True
    )

# ===================== EXPORT =====================
with tab_export:
    st.markdown("### ⬇️ Export")
    st.caption("Export includes the original risk_probability (0–1) plus full drivers/recommendations.")

    st.download_button(
        "Download scored CSV",
        data=out.drop(columns=["risk_probability_pct", "primary_driver", "primary_reco", "_tier_order", "risk_badge"])
              .to_csv(index=False)
              .encode("utf-8"),
        file_name="scored_risk_output.csv",
        mime="text/csv",
    )
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.caption("Notes: scoring is model-backed. Drivers/recommendations are rule-based and degrade gracefully if some fields are missing.")


