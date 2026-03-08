from __future__ import annotations

from pathlib import Path
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _find_base_dir() -> Path:
    """
    Try to locate project root in a robust way.
    Assumes this file lives under something like:
      <root>/backend_code/.../add_contract_flags.py
    and outputs are under <root>/backend_code/output OR <root>/output depending on your setup.
    """
    here = Path(__file__).resolve()
    # Option A: treat current folder as backend_code root (your original)
    base_a = here.parent

    # Option B: if repo has backend_code/ and output is under backend_code/output, keep base_a
    # Option C: if repo root has /output and scripts sit in backend_code/, go one level up
    # We detect by checking which input path exists.
    cand_a = base_a / "output" / "account_quarter_panel_qoq_processed_with_missing_flags.csv"
    cand_b = here.parents[1] / "output" / "account_quarter_panel_qoq_processed_with_missing_flags.csv"

    if cand_a.exists():
        return base_a
    if cand_b.exists():
        return here.parents[1]

    # fallback to original behavior
    return base_a


def _to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def main() -> None:
    base_dir = _find_base_dir()

    input_path = base_dir / "output" / "account_quarter_panel_qoq_processed_with_missing_flags.csv"
    output_path = base_dir / "output" / "account_quarter_panel_qoq_processed_with_contract_flags.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    df = pd.read_csv(input_path)
    row_count_before = len(df)

    # -----------------------------
    # Ensure rolling churn labels exist (do NOT overwrite if already present)
    # -----------------------------
    if "churn_label_q1_rolling" not in df.columns:
        if "account_churn_event_in_next_quarter" in df.columns:
            df["churn_label_q1_rolling"] = pd.to_numeric(
                df["account_churn_event_in_next_quarter"], errors="coerce"
            ).astype("Int64")
        else:
            print(
                "Warning: churn_label_q1_rolling not created. "
                "Missing columns: ['account_churn_event_in_next_quarter']"
            )

    if "churn_label_q2_rolling" not in df.columns:
        required_label_cols = [
            "account_churn_event_in_next_quarter",
            "account_churn_event_in_second_next_quarter",
        ]
        if all(col in df.columns for col in required_label_cols):
            df["churn_label_q2_rolling"] = (
                df["account_churn_event_in_next_quarter"].astype(bool)
                | df["account_churn_event_in_second_next_quarter"].astype(bool)
            ).astype("Int64")
        else:
            missing = [col for col in required_label_cols if col not in df.columns]
            print(
                "Warning: churn_label_q2_rolling not created. "
                f"Missing columns: {missing}"
            )

    # -----------------------------
    # Load subscription fact (used to derive contract end dates)
    # -----------------------------
    sub_fact_path = base_dir / "output" / "subscription_fact.csv"
    if not sub_fact_path.exists():
        raise FileNotFoundError(f"subscription_fact.csv not found: {sub_fact_path}")

    sub_fact = pd.read_csv(sub_fact_path)

    # normalize subscription_fact dates
    if "end_date" in sub_fact.columns:
        sub_fact["end_date"] = _to_datetime(sub_fact["end_date"])
    if "end_date_inferred" in sub_fact.columns:
        sub_fact["end_date_inferred"] = _to_datetime(sub_fact["end_date_inferred"])

    # normalize df date columns if they exist
    if "end_date_inferred" in df.columns:
        df["end_date_inferred"] = _to_datetime(df["end_date_inferred"])
    if "quarter_end" in df.columns:
        df["quarter_end"] = _to_datetime(df["quarter_end"])

    # -----------------------------
    # Build effective_end_date
    # Priority:
    #   1) observed end_date from subscription_fact (best)
    #   2) inferred end_date from df.end_date_inferred (if exists)
    #   3) inferred end_date from subscription_fact.end_date_inferred
    # -----------------------------
    sub_end_date = pd.Series(pd.NaT, index=df.index)
    sub_end_date_inferred = pd.Series(pd.NaT, index=df.index)

    if "subscription_id_asof_qend" in df.columns and "subscription_id" in sub_fact.columns:
        sub_fact_idx = sub_fact.set_index("subscription_id", drop=False)

        if "end_date" in sub_fact_idx.columns:
            mapped = df["subscription_id_asof_qend"].map(sub_fact_idx["end_date"])
            sub_end_date = _to_datetime(mapped)

        if "end_date_inferred" in sub_fact_idx.columns:
            mapped_inf = df["subscription_id_asof_qend"].map(sub_fact_idx["end_date_inferred"])
            sub_end_date_inferred = _to_datetime(mapped_inf)
    else:
        print("Warning: cannot map subscription end dates (missing subscription_id_asof_qend or subscription_id).")

    df_inferred = df["end_date_inferred"] if "end_date_inferred" in df.columns else pd.Series(pd.NaT, index=df.index)

    # observed wins; else use df inferred; else use subscription_fact inferred
    effective_end_date = sub_end_date.copy()
    effective_end_date = effective_end_date.where(effective_end_date.notna(), df_inferred)
    effective_end_date = effective_end_date.where(effective_end_date.notna(), sub_end_date_inferred)

    df["effective_end_date"] = effective_end_date

    # flags: based on what actually populated effective_end_date
    df["contract_end_observed_flag"] = sub_end_date.notna().astype(int)
    df["contract_end_inferred_flag"] = (
        (df["effective_end_date"].notna()) & (df["contract_end_observed_flag"] == 0)
    ).astype(int)

    # sanity checks
    row_count_after = len(df)
    print(f"Row count before: {row_count_before}")
    print(f"Row count after:  {row_count_after}")

    print("\nValue counts for contract_end_inferred_flag:")
    print(df["contract_end_inferred_flag"].value_counts(dropna=False).sort_index().to_string())

    print("\nValue counts for contract_end_observed_flag:")
    print(df["contract_end_observed_flag"].value_counts(dropna=False).sort_index().to_string())

    both_one = (df["contract_end_inferred_flag"] == 1) & (df["contract_end_observed_flag"] == 1)
    print(f"\nRows with both flags == 1: {int(both_one.sum())} (should be 0)")

    null_effective = df["effective_end_date"].isna().sum()
    print(f"Rows with effective_end_date is null: {int(null_effective)}")

    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()

