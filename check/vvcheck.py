from pathlib import Path
import pandas as pd

# === label columns (try both) ===
LABELS = [
    "account_churn_event_in_next_quarter",
    "churn_label_q2_rolling",
]


def pick_quarter_col(df: pd.DataFrame) -> str:
    # 你的面板里更可能叫 quarter_end / acct_quarter 等，所以这里加一些候选
    for c in [
        "quarter_end",
        "quarter",
        "acct_quarter",
        "account_quarter",
        "qtr",
        "quarter_str",
    ]:
        if c in df.columns:
            return c
    raise ValueError(
        f"Cannot find quarter column. Available columns (first 50): {list(df.columns)[:50]}"
    )


def pick_account_id_col(df: pd.DataFrame) -> str:
    for c in ["account_id", "acct_id", "id", "account", "account_name"]:
        if c in df.columns:
            return
