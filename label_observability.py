from __future__ import annotations

import pandas as pd


def add_label_availability(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "quarter" in df.columns and df["quarter"].notna().any():
        quarter_period = pd.PeriodIndex(df["quarter"], freq="Q")
    else:
        quarter_period = pd.PeriodIndex(pd.to_datetime(df["quarter_end"]), freq="Q")
    max_quarter = quarter_period.max()
    # Exclude the final quarter due to label observability, not business behavior.
    df["label_available_q1"] = (quarter_period + 1) <= max_quarter
    df["label_available_q2"] = (quarter_period + 2) <= max_quarter
    return df
