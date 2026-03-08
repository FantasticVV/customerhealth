from pathlib import Path
from label_observability import add_label_availability
import warnings
import pandas as pd
import numpy as np

# ============================================================
# Silence ONLY the noisy pandas concat FutureWarning (optional)
# ============================================================
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*DataFrame concatenation with empty or all-NA entries.*",
)

# =========================
# Paths / IO
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "raw data"
OUT_DIR = BASE_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)


def save_csv(df: pd.DataFrame, name: str):
    df.to_csv(OUT_DIR / f"{name}.csv", index=False)


def load():
    accounts = pd.read_csv(DATA_DIR / "ravenstack_accounts.csv")
    subs     = pd.read_csv(DATA_DIR / "ravenstack_subscriptions.csv")
    usage    = pd.read_csv(DATA_DIR / "ravenstack_feature_usage.csv")
    tickets  = pd.read_csv(DATA_DIR / "ravenstack_support_tickets.csv")
    churn    = pd.read_csv(DATA_DIR / "ravenstack_churn_events.csv")
    return accounts, subs, usage, tickets, churn


def coerce_types(accounts, subs, usage, tickets, churn):
    # IDs as string
    for df in (accounts, subs, tickets, churn):
        if "account_id" in df.columns:
            df["account_id"] = df["account_id"].astype(str)

    if "subscription_id" in subs.columns:
        subs["subscription_id"] = subs["subscription_id"].astype(str)
    if "subscription_id" in usage.columns:
        usage["subscription_id"] = usage["subscription_id"].astype(str)

    # Dates
    if "signup_date" in accounts.columns:
        accounts["signup_date"] = pd.to_datetime(accounts["signup_date"], errors="coerce")

    for c in ["start_date", "end_date"]:
        if c in subs.columns:
            subs[c] = pd.to_datetime(subs[c], errors="coerce")

    if "usage_date" in usage.columns:
        usage["usage_date"] = pd.to_datetime(usage["usage_date"], errors="coerce")

    if "submitted_at" in tickets.columns:
        tickets["submitted_at"] = pd.to_datetime(tickets["submitted_at"], errors="coerce")
    if "closed_at" in tickets.columns:
        tickets["closed_at"] = pd.to_datetime(tickets["closed_at"], errors="coerce")

    if "churn_date" in churn.columns:
        churn["churn_date"] = pd.to_datetime(churn["churn_date"], errors="coerce")

    return accounts, subs, usage, tickets, churn


# =========================
# Fact builders
# =========================
def make_subscription_fact(subs: pd.DataFrame) -> pd.DataFrame:
    """
    subscriptions 原表不改；只增加分析用列：
    - subscription_seq: account 内第几条订阅（按 start_date）
    - next_start_date: 下一条订阅开始时间
    - end_date_inferred: end_date 为空且下一条开始了 => 用 next_start_date 作为分析用“结束点”(半开区间)
    """
    sf = subs.copy()
    sf = sf.sort_values(["account_id", "start_date"]).reset_index(drop=True)
    sf["subscription_seq"] = sf.groupby("account_id").cumcount() + 1
    sf["next_start_date"] = sf.groupby("account_id")["start_date"].shift(-1)

    sf["end_date_inferred"] = sf["end_date"]
    mask = sf["end_date_inferred"].isna() & sf["next_start_date"].notna()
    sf.loc[mask, "end_date_inferred"] = sf.loc[mask, "next_start_date"]

    keep = [
        "account_id", "subscription_id", "start_date", "end_date", "end_date_inferred",
        "next_start_date", "subscription_seq",
        "plan_tier", "seats", "mrr_amount", "arr_amount", "billing_frequency",
        "is_trial", "auto_renew_flag", "upgrade_flag", "downgrade_flag", "churn_flag"
    ]
    keep = [c for c in keep if c in sf.columns]
    sf = sf[keep].copy()

    # Rename subscription-level churn for clarity: "cancel"
    if "churn_flag" in sf.columns:
        sf = sf.rename(columns={"churn_flag": "subscription_cancel_flag"})

    return sf


def make_usage_fact(usage: pd.DataFrame, subs_fact: pd.DataFrame) -> pd.DataFrame:
    """usage 明细保留全部行，只补 account_id（通过 subscription_id 映射）"""
    map_df = subs_fact[["subscription_id", "account_id"]].drop_duplicates()
    return usage.merge(map_df, on="subscription_id", how="left")


def make_tickets_fact(tickets: pd.DataFrame) -> pd.DataFrame:
    """tickets 明细保留全部行，加 submitted_date/closed_date 方便聚合"""
    tf = tickets.copy()
    if "submitted_at" in tf.columns:
        tf["submitted_date"] = tf["submitted_at"].dt.floor("D")
    if "closed_at" in tf.columns:
        tf["closed_date"] = tf["closed_at"].dt.floor("D")
    return tf


def make_churn_fact(churn: pd.DataFrame) -> pd.DataFrame:
    """churn_events 明细按时间排序，加 churn_seq（并改名更清晰）"""
    cf = churn.sort_values(["account_id", "churn_date"]).reset_index(drop=True)
    cf["churn_seq"] = cf.groupby("account_id").cumcount() + 1
    return cf


# =========================
# Window helpers
# =========================
def month_windows(min_date: pd.Timestamp, max_date: pd.Timestamp):
    min_m = pd.Period(min_date, freq="M")
    max_m = pd.Period(max_date, freq="M")
    for p in pd.period_range(min_m, max_m, freq="M"):
        yield str(p), p.start_time.normalize(), p.end_time.normalize()


def quarter_windows(min_date: pd.Timestamp, max_date: pd.Timestamp):
    min_q = pd.Period(min_date, freq="Q")
    max_q = pd.Period(max_date, freq="Q")
    for p in pd.period_range(min_q, max_q, freq="Q"):
        yield str(p), p.start_time.normalize(), p.end_time.normalize()


def _find_usage_duration_col(usage_fact: pd.DataFrame):
    for c in ["usage_duration_secs", "usage_duration_seconds", "usage_duration_sec"]:
        if c in usage_fact.columns:
            return c
    return None


def pick_subscription_asof(sub_fact: pd.DataFrame, t: pd.Timestamp, id_col_name: str) -> pd.DataFrame:
    """
    每个 account 取 as-of 时间点 t 的“有效订阅”（若多条取 start_date 最新的）
    有效：start_date <= t 且 (end_date_inferred 为空 或 end_date_inferred > t)
    """
    if sub_fact.empty:
        return pd.DataFrame(columns=["account_id"])

    sf = sub_fact.dropna(subset=["start_date"]).copy()
    eff = sf[(sf["start_date"] <= t) & (sf["end_date_inferred"].isna() | (sf["end_date_inferred"] > t))]
    if eff.empty:
        return pd.DataFrame(columns=["account_id"])

    eff = eff.sort_values(["account_id", "start_date", "subscription_seq"])
    chosen = eff.groupby("account_id", as_index=False).tail(1)

    keep = ["account_id", "subscription_id", "plan_tier", "seats", "mrr_amount", "arr_amount", "billing_frequency", "end_date_inferred"]
    keep = [c for c in keep if c in chosen.columns]
    out = chosen[keep].rename(columns={"subscription_id": id_col_name})
    return out


def _build_subscription_end_labels(sub_fact: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                                  count_col: str, flag_col: str) -> pd.DataFrame:
    """
    订阅结束事件（subscription-level end）在窗口内的计数 + boolean
    用 end_date 作为“结束发生时间”。
    """
    if sub_fact.empty or "end_date" not in sub_fact.columns:
        return pd.DataFrame(columns=["account_id", count_col, flag_col])

    ended = sub_fact.dropna(subset=["end_date"]).copy()
    ended = ended[(ended["end_date"] >= start) & (ended["end_date"] <= end)]
    if ended.empty:
        return pd.DataFrame(columns=["account_id", count_col, flag_col])

    out = ended.groupby("account_id").size().rename(count_col).reset_index()
    out[flag_col] = True
    return out


def _base_date_range(accounts, sub_fact, usage_fact, tickets_fact, churn_fact):
    candidates = []
    for s in [
        accounts.get("signup_date"),
        usage_fact.get("usage_date"),
        tickets_fact.get("submitted_at"),
        sub_fact.get("start_date"),
        sub_fact.get("end_date"),
        churn_fact.get("churn_date"),
    ]:
        if s is None:
            continue
        s2 = pd.to_datetime(s, errors="coerce")
        if s2.notna().any():
            candidates += [s2.min(), s2.max()]

    if not candidates:
        raise ValueError("No valid dates found across tables. Check date columns and parsing.")
    return min(candidates), max(candidates)


# =========================
# Panel builders
# =========================
def make_account_month_panel(accounts: pd.DataFrame,
                            sub_fact: pd.DataFrame,
                            usage_fact: pd.DataFrame,
                            tickets_fact: pd.DataFrame,
                            churn_fact: pd.DataFrame) -> pd.DataFrame:
    """
    输出 account × month 的面板（满足 month-by-month + own-history trend）
    - account_churn_event_* 来自 churn_events（账号层面的“流失事件”）
    - subscription_end_* 来自 subscriptions 的 end_date（订阅结束）
    """
    min_date, max_date = _base_date_range(accounts, sub_fact, usage_fact, tickets_fact, churn_fact)

    base = accounts[["account_id", "signup_date"]].copy()
    churn_dates = (
        churn_fact[["account_id", "churn_date"]].dropna()
        if not churn_fact.empty else pd.DataFrame(columns=["account_id", "churn_date"])
    )
    duration_col = _find_usage_duration_col(usage_fact)

    panels = []
    for m, m_start, m_end in month_windows(min_date, max_date):
        # ---- usage ----
        uf = usage_fact.dropna(subset=["account_id", "usage_date"]).copy()
        uf_m = uf[(uf["usage_date"] >= m_start) & (uf["usage_date"] <= m_end)]

        if uf_m.empty:
            usage_m = pd.DataFrame(columns=["account_id", "usage_count", "usage_hours", "active_days", "errors", "features_used"])
        else:
            agg = {"active_days": ("usage_date", "nunique")}
            if "usage_count" in uf_m.columns:
                agg["usage_count"] = ("usage_count", "sum")
            if duration_col is not None:
                agg["usage_hours"] = (duration_col, lambda s: s.sum() / 3600.0)
            if "error_count" in uf_m.columns:
                agg["errors"] = ("error_count", "sum")
            if "feature_name" in uf_m.columns:
                agg["features_used"] = ("feature_name", "nunique")

            usage_m = uf_m.groupby("account_id", as_index=False).agg(**agg)

        # ---- tickets ----
        tf = tickets_fact.dropna(subset=["account_id", "submitted_at"]).copy()
        tf_m = tf[(tf["submitted_at"] >= m_start) & (tf["submitted_at"] <= m_end)]

        if tf_m.empty:
            tickets_m = pd.DataFrame(columns=["account_id", "tickets_opened", "escalations", "avg_satisfaction"])
        else:
            tickets_m = tf_m.groupby("account_id", as_index=False).agg(
                tickets_opened=("ticket_id", "count") if "ticket_id" in tf_m.columns else ("submitted_at", "count"),
                escalations=("escalation_flag", "sum") if "escalation_flag" in tf_m.columns else ("submitted_at", lambda s: 0),
                avg_satisfaction=("satisfaction_score", "mean") if "satisfaction_score" in tf_m.columns else ("submitted_at", lambda s: pd.NA),
            )

        # ---- commercial as-of month end ----
        commercial = pick_subscription_asof(sub_fact, m_end, "subscription_id_asof_period_end")

        # ---- account churn events (from churn_events.csv) ----
        churn_in_m = churn_dates[(churn_dates["churn_date"] >= m_start) & (churn_dates["churn_date"] <= m_end)]
        churn_in_m = churn_in_m.groupby("account_id").size().rename("account_churn_event_count_in_period").reset_index()
        churn_in_m["account_churn_event_in_period"] = True

        next_m_end = (pd.Period(m_end, freq="M") + 1).end_time.normalize()
        churn_next = churn_dates[(churn_dates["churn_date"] > m_end) & (churn_dates["churn_date"] <= next_m_end)]
        churn_next = churn_next.groupby("account_id").size().rename("account_churn_event_count_in_next_period").reset_index()
        churn_next["account_churn_event_in_next_period"] = True

        # ---- subscription end labels (based on end_date) ----
        sub_end_m = _build_subscription_end_labels(
            sub_fact, m_start, m_end,
            count_col="subscription_ends_in_period",
            flag_col="subscription_end_in_period",
        )
        next_sub_end = _build_subscription_end_labels(
            sub_fact, m_end + pd.Timedelta(days=1), next_m_end,
            count_col="subscription_ends_in_next_period",
            flag_col="subscription_end_in_next_period",
        )

        # ---- assemble ----
        panel = base.copy()
        panel["period"] = m
        panel["period_start"] = m_start
        panel["period_end"] = m_end

        panel = (panel
                 .merge(usage_m, on="account_id", how="left")
                 .merge(tickets_m, on="account_id", how="left")
                 .merge(commercial, on="account_id", how="left")
                 .merge(churn_in_m, on="account_id", how="left")
                 .merge(churn_next, on="account_id", how="left")
                 .merge(sub_end_m, on="account_id", how="left")
                 .merge(next_sub_end, on="account_id", how="left"))

        # Ensure boolean columns exist
        for b in [
            "account_churn_event_in_period",
            "account_churn_event_in_next_period",
            "subscription_end_in_period",
            "subscription_end_in_next_period",
        ]:
            if b not in panel.columns:
                panel[b] = False
            panel[b] = panel[b].astype("boolean").fillna(False).astype(bool)

        panels.append(panel)

    panels = [p for p in panels if p is not None and not p.empty]
    if not panels:
        return pd.DataFrame(columns=["account_id", "signup_date", "period", "period_start", "period_end"])

    out = (pd.concat(panels, ignore_index=True)
           .sort_values(["account_id", "period_end"])
           .reset_index(drop=True))

    # ---- drop pre-signup periods ----
    out = out[out["period_end"] >= out["signup_date"]].copy()

    # ---- count-like NaN -> 0 ----
    count_like = [
        "usage_count", "usage_hours", "active_days", "errors", "features_used",
        "tickets_opened", "escalations",
        "account_churn_event_count_in_period", "account_churn_event_count_in_next_period",
        "subscription_ends_in_period", "subscription_ends_in_next_period",
    ]
    for c in count_like:
        if c in out.columns:
            out[c] = out[c].fillna(0)

    # ---- commercial normalization ----
    if "subscription_id_asof_period_end" not in out.columns:
        out["subscription_id_asof_period_end"] = pd.NA
    out["active_sub_at_period_end"] = out["subscription_id_asof_period_end"].notna()

    for c in ["seats", "mrr_amount", "arr_amount"]:
        if c in out.columns:
            out.loc[~out["active_sub_at_period_end"], c] = out.loc[~out["active_sub_at_period_end"], c].fillna(0)

    for c in ["plan_tier", "billing_frequency"]:
        if c in out.columns:
            out.loc[~out["active_sub_at_period_end"], c] = out.loc[~out["active_sub_at_period_end"], c].fillna("none")

    # timeline-ish feature: days to contract end (as-of period_end)
    if "end_date_inferred" in out.columns:
        out["days_to_contract_end_asof_period_end"] = (out["end_date_inferred"] - out["period_end"]).dt.days
        out.loc[~out["active_sub_at_period_end"], "days_to_contract_end_asof_period_end"] = np.nan

    # ---- satisfaction missing flag (keep avg_satisfaction NaN) ----
    if "avg_satisfaction" not in out.columns:
        out["avg_satisfaction"] = pd.NA
    if "tickets_opened" not in out.columns:
        out["tickets_opened"] = 0
    out["satisfaction_missing_flag"] = ((out["tickets_opened"] > 0) & (out["avg_satisfaction"].isna())).astype(int)

    # ---- deltas + rolling own-history features ----
    out = out.sort_values(["account_id", "period_end"]).reset_index(drop=True)
    base_cols = ["usage_count", "usage_hours", "active_days", "errors", "features_used",
                 "tickets_opened", "escalations", "seats", "mrr_amount", "arr_amount"]
    for col in base_cols:
        if col in out.columns:
            out[f"{col}_mom_delta"] = out.groupby("account_id")[col].diff()

    roll_cols = ["usage_count", "usage_hours", "tickets_opened", "errors", "features_used", "mrr_amount", "seats"]
    for col in roll_cols:
        if col in out.columns:
            g = out.groupby("account_id")[col]
            out[f"{col}_roll3_mean"] = g.transform(lambda s: s.rolling(3, min_periods=1).mean())
            out[f"{col}_vs_roll3_mean"] = out[col] - out[f"{col}_roll3_mean"]
            out[f"{col}_roll3_change"] = g.transform(lambda s: s.diff(2))  # current - two months ago

    out["usage_drop_flag"] = (
        (out.get("usage_count", 0) < out.get("usage_count_roll3_mean", 0)) &
        (out.get("usage_count_mom_delta", 0) < 0)
    ).astype(int)

    out["tickets_spike_flag"] = (
        (out.get("tickets_opened", 0) > out.get("tickets_opened_roll3_mean", 0)) &
        (out.get("tickets_opened_mom_delta", 0) > 0)
    ).astype(int)

    out["downsell_flag"] = 0
    if "seats_mom_delta" in out.columns:
        out["downsell_flag"] = (out["seats_mom_delta"] < 0).astype(int)
    if "mrr_amount_mom_delta" in out.columns:
        out["downsell_flag"] = np.maximum(out["downsell_flag"], (out["mrr_amount_mom_delta"] < 0).astype(int))

    return out


def make_account_quarter_panel(accounts: pd.DataFrame,
                              sub_fact: pd.DataFrame,
                              usage_fact: pd.DataFrame,
                              tickets_fact: pd.DataFrame,
                              churn_fact: pd.DataFrame) -> pd.DataFrame:
    """
    输出 account × quarter 的高层面板（便于高层/EDA）
    - account_churn_event_* 来自 churn_events（账号层面的“流失事件”）
    - subscription_end_* 来自 subscriptions 的 end_date（订阅结束）
    """
    min_date, max_date = _base_date_range(accounts, sub_fact, usage_fact, tickets_fact, churn_fact)

    base = accounts[["account_id", "signup_date"]].copy()
    churn_dates = (
        churn_fact[["account_id", "churn_date"]].dropna()
        if not churn_fact.empty else pd.DataFrame(columns=["account_id", "churn_date"])
    )
    duration_col = _find_usage_duration_col(usage_fact)

    panels = []
    for q, q_start, q_end in quarter_windows(min_date, max_date):
        # ---- usage ----
        uf = usage_fact.dropna(subset=["account_id", "usage_date"]).copy()
        uf_q = uf[(uf["usage_date"] >= q_start) & (uf["usage_date"] <= q_end)]

        if uf_q.empty:
            usage_q = pd.DataFrame(columns=["account_id", "usage_count", "usage_hours", "active_days", "errors", "features_used"])
        else:
            agg = {"active_days": ("usage_date", "nunique")}
            if "usage_count" in uf_q.columns:
                agg["usage_count"] = ("usage_count", "sum")
            if duration_col is not None:
                agg["usage_hours"] = (duration_col, lambda s: s.sum() / 3600.0)
            if "error_count" in uf_q.columns:
                agg["errors"] = ("error_count", "sum")
            if "feature_name" in uf_q.columns:
                agg["features_used"] = ("feature_name", "nunique")

            usage_q = uf_q.groupby("account_id", as_index=False).agg(**agg)

        # ---- tickets ----
        tf = tickets_fact.dropna(subset=["account_id", "submitted_at"]).copy()
        tf_q = tf[(tf["submitted_at"] >= q_start) & (tf["submitted_at"] <= q_end)]

        if tf_q.empty:
            tickets_q = pd.DataFrame(columns=["account_id", "tickets_opened", "escalations", "avg_satisfaction"])
        else:
            tickets_q = tf_q.groupby("account_id", as_index=False).agg(
                tickets_opened=("ticket_id", "count") if "ticket_id" in tf_q.columns else ("submitted_at", "count"),
                escalations=("escalation_flag", "sum") if "escalation_flag" in tf_q.columns else ("submitted_at", lambda s: 0),
                avg_satisfaction=("satisfaction_score", "mean") if "satisfaction_score" in tf_q.columns else ("submitted_at", lambda s: pd.NA),
            )

        # ---- commercial as-of quarter end ----
        commercial = pick_subscription_asof(sub_fact, q_end, "subscription_id_asof_qend")

        # ---- account churn events (from churn_events.csv) ----
        churn_in_q = churn_dates[(churn_dates["churn_date"] >= q_start) & (churn_dates["churn_date"] <= q_end)]
        churn_in_q = churn_in_q.groupby("account_id").size().rename("account_churn_event_count_in_quarter").reset_index()
        churn_in_q["account_churn_event_in_quarter"] = True

        next_q_end = (pd.Period(q_end, freq="Q") + 1).end_time.normalize()
        churn_next = churn_dates[(churn_dates["churn_date"] > q_end) & (churn_dates["churn_date"] <= next_q_end)]
        churn_next = churn_next.groupby("account_id").size().rename("account_churn_event_count_in_next_quarter").reset_index()
        churn_next["account_churn_event_in_next_quarter"] = True

        second_next_q_end = (pd.Period(q_end, freq="Q") + 2).end_time.normalize()
        churn_second_next = churn_dates[
            (churn_dates["churn_date"] > next_q_end) & (churn_dates["churn_date"] <= second_next_q_end)
        ]
        churn_second_next = churn_second_next[["account_id"]].drop_duplicates()
        churn_second_next["account_churn_event_in_second_next_quarter"] = True

        # ---- subscription end labels ----
        sub_end_q = _build_subscription_end_labels(
            sub_fact, q_start, q_end,
            count_col="subscription_ends_in_quarter",
            flag_col="subscription_end_in_quarter",
        )
        next_sub_end = _build_subscription_end_labels(
            sub_fact, q_end + pd.Timedelta(days=1), next_q_end,
            count_col="subscription_ends_in_next_quarter",
            flag_col="subscription_end_in_next_quarter",
        )

        # ---- assemble ----
        panel = base.copy()
        panel["quarter"] = q
        panel["quarter_start"] = q_start
        panel["quarter_end"] = q_end

        panel = (panel
                 .merge(usage_q, on="account_id", how="left")
                 .merge(tickets_q, on="account_id", how="left")
                 .merge(commercial, on="account_id", how="left")
                 .merge(churn_in_q, on="account_id", how="left")
                 .merge(churn_next, on="account_id", how="left")
                 .merge(churn_second_next, on="account_id", how="left")
                 .merge(sub_end_q, on="account_id", how="left")
                 .merge(next_sub_end, on="account_id", how="left"))

        # IMPORTANT: keep as pandas nullable boolean so we can later set pd.NA for unobservable rows
        for b in [
            "account_churn_event_in_quarter",
            "account_churn_event_in_next_quarter",
            "account_churn_event_in_second_next_quarter",
            "subscription_end_in_quarter",
            "subscription_end_in_next_quarter",
        ]:
            if b not in panel.columns:
                panel[b] = pd.NA
            panel[b] = panel[b].astype("boolean").fillna(False)

        # churn labels (rolling). Use nullable Int64 so we can set pd.NA later.
        panel["churn_label_q1_rolling"] = (
            panel["account_churn_event_in_next_quarter"].astype("Int64")
        )
        panel["churn_label_q2_rolling"] = (
            (
                panel["account_churn_event_in_next_quarter"]
                | panel["account_churn_event_in_second_next_quarter"]
            ).astype("Int64")
        )

        panels.append(panel)

    panels = [p for p in panels if p is not None and not p.empty]
    if not panels:
        return pd.DataFrame(columns=["account_id", "signup_date", "quarter", "quarter_start", "quarter_end"])

    out = (pd.concat(panels, ignore_index=True)
           .sort_values(["account_id", "quarter_end"])
           .reset_index(drop=True))

    # ---- drop pre-signup quarters ----
    out = out[out["quarter_end"] >= out["signup_date"]].copy()

    # ---- count-like NaN -> 0 ----
    count_like = [
        "usage_count", "usage_hours", "active_days", "errors", "features_used",
        "tickets_opened", "escalations",
        "account_churn_event_count_in_quarter", "account_churn_event_count_in_next_quarter",
        "subscription_ends_in_quarter", "subscription_ends_in_next_quarter",
    ]
    for c in count_like:
        if c in out.columns:
            out[c] = out[c].fillna(0)

    # ---- commercial normalization ----
    if "subscription_id_asof_qend" not in out.columns:
        out["subscription_id_asof_qend"] = pd.NA
    out["active_sub_at_qend"] = out["subscription_id_asof_qend"].notna()

    for c in ["seats", "mrr_amount", "arr_amount"]:
        if c in out.columns:
            out.loc[~out["active_sub_at_qend"], c] = out.loc[~out["active_sub_at_qend"], c].fillna(0)

    for c in ["plan_tier", "billing_frequency"]:
        if c in out.columns:
            out.loc[~out["active_sub_at_qend"], c] = out.loc[~out["active_sub_at_qend"], c].fillna("none")

    # timeline-ish feature: days to contract end (as-of quarter_end)
    if "end_date_inferred" in out.columns:
        out["days_to_contract_end_asof_qend"] = (out["end_date_inferred"] - out["quarter_end"]).dt.days
        out.loc[~out["active_sub_at_qend"], "days_to_contract_end_asof_qend"] = np.nan

    # ---- satisfaction missing flag ----
    if "avg_satisfaction" not in out.columns:
        out["avg_satisfaction"] = pd.NA
    if "tickets_opened" not in out.columns:
        out["tickets_opened"] = 0
    out["satisfaction_missing_flag"] = ((out["tickets_opened"] > 0) & (out["avg_satisfaction"].isna())).astype(int)

    # ---- QoQ deltas + 4-quarter rolling (1 year) ----
    out = out.sort_values(["account_id", "quarter_end"]).reset_index(drop=True)
    base_cols = ["usage_count", "usage_hours", "active_days", "errors", "features_used",
                 "tickets_opened", "escalations", "seats", "mrr_amount", "arr_amount"]
    for col in base_cols:
        if col in out.columns:
            out[f"{col}_qoq_delta"] = out.groupby("account_id")[col].diff()

    roll_cols = ["usage_count", "usage_hours", "tickets_opened", "errors", "features_used", "mrr_amount", "seats"]
    for col in roll_cols:
        if col in out.columns:
            g = out.groupby("account_id")[col]
            out[f"{col}_roll4_mean"] = g.transform(lambda s: s.rolling(4, min_periods=1).mean())
            out[f"{col}_vs_roll4_mean"] = out[col] - out[f"{col}_roll4_mean"]
            out[f"{col}_roll4_change"] = g.transform(lambda s: s.diff(3))  # current - 3 quarters ago

    out["usage_drop_flag"] = (
        (out.get("usage_count", 0) < out.get("usage_count_roll4_mean", 0)) &
        (out.get("usage_count_qoq_delta", 0) < 0)
    ).astype(int)

    out["tickets_spike_flag"] = (
        (out.get("tickets_opened", 0) > out.get("tickets_opened_roll4_mean", 0)) &
        (out.get("tickets_opened_qoq_delta", 0) > 0)
    ).astype(int)

    out["downsell_flag"] = 0
    if "seats_qoq_delta" in out.columns:
        out["downsell_flag"] = (out["seats_qoq_delta"] < 0).astype(int)
    if "mrr_amount_qoq_delta" in out.columns:
        out["downsell_flag"] = np.maximum(out["downsell_flag"], (out["mrr_amount_qoq_delta"] < 0).astype(int))

    # ============================================================
    # Label observability (FIX right-censoring -> fake zeros)
    # ============================================================
    out = add_label_availability(out)

    # Q1 horizon label: last quarter is not observable
    if "account_churn_event_in_next_quarter" in out.columns and "label_available_q1" in out.columns:
        out.loc[~out["label_available_q1"], "account_churn_event_in_next_quarter"] = pd.NA
    if "churn_label_q1_rolling" in out.columns and "label_available_q1" in out.columns:
        out.loc[~out["label_available_q1"], "churn_label_q1_rolling"] = pd.NA
        # keep dtype nullable Int64 after setting NA
        try:
            out["churn_label_q1_rolling"] = out["churn_label_q1_rolling"].astype("Int64")
        except Exception:
            pass

    # Q2 rolling label: last TWO quarters are not observable
    if "account_churn_event_in_second_next_quarter" in out.columns and "label_available_q2" in out.columns:
        out.loc[~out["label_available_q2"], "account_churn_event_in_second_next_quarter"] = pd.NA

    if "churn_label_q2_rolling" in out.columns and "label_available_q2" in out.columns:
        out.loc[~out["label_available_q2"], "churn_label_q2_rolling"] = pd.NA
        # keep dtype nullable Int64 after setting NA
        try:
            out["churn_label_q2_rolling"] = out["churn_label_q2_rolling"].astype("Int64")
        except Exception:
            pass

    return out


def make_latest_snapshot(panel: pd.DataFrame, end_col: str) -> pd.DataFrame:
    """给 demo/汇总用：每个 account 取最新一期（月末 or 季度末）一行"""
    if panel.empty:
        return panel
    latest = (panel.sort_values(["account_id", end_col])
              .groupby("account_id", as_index=False)
              .tail(1)
              .reset_index(drop=True))
    return latest


# =========================
# Main
# =========================
def main():
    accounts, subs, usage, tickets, churn = load()
    accounts, subs, usage, tickets, churn = coerce_types(accounts, subs, usage, tickets, churn)

    sub_fact = make_subscription_fact(subs)
    usage_fact = make_usage_fact(usage, sub_fact)
    tickets_fact = make_tickets_fact(tickets)
    churn_fact = make_churn_fact(churn)

    # Panels
    month_panel = make_account_month_panel(accounts, sub_fact, usage_fact, tickets_fact, churn_fact)
    quarter_panel = make_account_quarter_panel(accounts, sub_fact, usage_fact, tickets_fact, churn_fact)

    # Latest snapshots
    latest_month = make_latest_snapshot(month_panel, end_col="period_end")
    latest_quarter = make_latest_snapshot(quarter_panel, end_col="quarter_end")

    # Save outputs
    save_csv(month_panel, "account_month_panel")
    save_csv(quarter_panel, "account_quarter_panel")
    save_csv(latest_month, "account_latest_snapshot_month")
    save_csv(latest_quarter, "account_latest_snapshot_quarter")

    save_csv(sub_fact, "subscription_fact")
    save_csv(usage_fact, "usage_fact")
    save_csv(tickets_fact, "tickets_fact")
    save_csv(churn_fact, "churn_fact")

    print("✅ Done. Saved tables to output/:")
    print(" - account_month_panel.csv")
    print(" - account_quarter_panel.csv")
    print(" - account_latest_snapshot_month.csv")
    print(" - account_latest_snapshot_quarter.csv")
    print(" - subscription_fact.csv")
    print(" - usage_fact.csv")
    print(" - tickets_fact.csv")
    print(" - churn_fact.csv")


if __name__ == "__main__":
    main()
