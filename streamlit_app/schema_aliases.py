from __future__ import annotations

import re
from typing import Optional, Sequence

# -----------------------------------------------------------------------------
# Canonical schema aliases
# IMPORTANT:
# - Avoid ambiguous mappings across tables (e.g., created_at exists in many places).
# - Keep aliases scoped by "table context" so we don't rename the wrong column.
# -----------------------------------------------------------------------------

ALIASES = {
    # -------------------------
    # Accounts table / profile
    # -------------------------
    "accounts": {
        "account_id": ["account_id", "customer_id", "customerid", "consumer_id", "consumerid", "client_id", "org_id", "company_id", "acct_id"],
        "signup_date": ["signup_date", "account_created_at", "account_creation_date", "created_at"],  # only use in accounts context
        "industry": ["industry", "vertical"],
        "company_size": ["company_size", "employee_count", "employees", "org_size"],
    },

    # -------------------------
    # Single RAW current/prev
    # -------------------------
    "single_raw": {
        "account_id": ["account_id", "customer_id", "customerid", "consumer_id", "consumerid", "client_id", "org_id", "company_id", "acct_id"],
        "plan_tier": ["plan_tier", "plan", "tier", "plan_name"],
        "signup_date": ["signup_date", "created_at", "account_created_at", "account_creation_date"],
        "as_of_date": ["as_of_date", "snapshot_date", "report_date", "quarter_end"],

        "usage_count_current": ["usage_count_current", "usage_current", "current_usage_count"],
        "usage_count_prev": ["usage_count_prev", "usage_prev", "previous_usage_count"],

        "tickets_opened_current": ["tickets_opened_current", "tickets_current", "current_tickets_opened"],
        "tickets_opened_prev": ["tickets_opened_prev", "tickets_prev", "previous_tickets_opened"],

        "avg_satisfaction_current": ["avg_satisfaction_current", "satisfaction_current", "csat_current", "avg_csat_current"],
        "days_to_contract_end_current": ["days_to_contract_end_current", "days_to_contract_end", "days_to_end_current"],

        "seats_current": ["seats_current", "current_seats", "seat_count_current"],
        "seats_prev": ["seats_prev", "previous_seats", "seat_count_prev"],

        "arr_current": ["arr_current", "current_arr", "arr_amount_current", "arr_amount", "arr", "annual_recurring_revenue"],
        "arr_prev": ["arr_prev", "previous_arr", "arr_amount_prev"],
    },

    # -------------------------
    # Subscriptions table
    # -------------------------
    "subscriptions": {
        "subscription_id": ["subscription_id", "sub_id", "subscriptionid", "contract_id", "plan_id"],
        "account_id": ["account_id", "customer_id", "customerid", "consumer_id", "consumerid", "client_id", "org_id", "company_id", "acct_id"],
        "plan_tier": ["plan_tier", "plan", "tier", "plan_name"],
        "start_date": ["start_date", "contract_start", "subscription_start", "sub_start_date"],
        "end_date": ["end_date", "contract_end_date", "contract_end", "subscription_end", "sub_end_date"],
        "seats": ["seats", "licensed_seats", "licenses", "seat_count"],
        "arr_amount": ["arr_amount", "arr", "annual_recurring_revenue"],
        "mrr_amount": ["mrr_amount", "mrr", "monthly_recurring_revenue"],
    },

    # -------------------------
    # Usage table
    # -------------------------
    "usage": {
        "subscription_id": ["subscription_id", "sub_id", "subscriptionid", "contract_id", "plan_id"],
        "usage_date": ["usage_date", "event_date", "timestamp", "date"],
        "usage_count": ["usage_count", "count", "usage", "events", "event_count", "usage_events"],
    },

    # -------------------------
    # Tickets table
    # -------------------------
    "tickets": {
        "account_id": ["account_id", "customer_id", "customerid", "consumer_id", "consumerid", "client_id", "org_id", "company_id", "acct_id"],
        "submitted_at": ["submitted_at", "ticket_created_at", "opened_at", "created_at"],  # only use in tickets context
        "ticket_id": ["ticket_id", "case_id", "support_ticket_id", "support_id", "id"],
        "satisfaction_score": ["satisfaction_score", "csat", "satisfaction", "rating", "csat_score", "nps_score"],
    },
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _norm(s: str) -> str:
    """Normalize column name for robust matching."""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)          # spaces -> underscore
    s = re.sub(r"[^a-z0-9_]", "", s)    # remove punctuation
    return s

def _norm_list(cols: Sequence[str]) -> list[str]:
    return [_norm(c) for c in cols]

def find_col(df_cols: Sequence[str], canonical: str, context: str = "single_raw") -> Optional[str]:
    """
    Find the best matching column name from df_cols for a canonical field,
    scoped by context to avoid cross-table collisions.

    context options:
      - "accounts"
      - "single_raw"
      - "subscriptions"
      - "usage"
      - "tickets"
    """
    if context not in ALIASES:
        raise ValueError(f"Unknown context '{context}'. Valid: {list(ALIASES.keys())}")

    aliases = ALIASES[context].get(canonical, [])
    if not aliases:
        return None

    df_cols = list(df_cols)
    df_norm = _norm_list(df_cols)

    # 1) exact normalized match
    for cand in aliases:
        c = _norm(cand)
        if c in df_norm:
            return df_cols[df_norm.index(c)]

    # 2) fallback: substring match (very light, to help camelCase / small diffs)
    #    e.g. "accountid" vs "account_id"
    joined = {df_norm[i]: df_cols[i] for i in range(len(df_cols))}
    for cand in aliases:
        c = _norm(cand)
        for k, orig in joined.items():
            if c == k:
                return orig
            if c and (c in k or k in c):
                return orig

    return None
