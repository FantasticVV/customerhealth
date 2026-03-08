from __future__ import annotations

from pathlib import Path
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _find_base_dir() -> Path:
    """
    Robustly locate the directory that contains /output.
    This prevents different scripts from accidentally writing to different output folders.
    """
    here = Path(__file__).resolve()
    base_a = here.parent

    cand_a = base_a / "output" / "account_quarter_panel_qoq_processed.csv"
    cand_b = here.parents[1] / "output" / "account_quarter_panel_qoq_processed.csv"

    if cand_a.exists():
        return base_a
    if cand_b.exists():
        return here.parents[1]

    return base_a


def main() -> None:
    base_dir = _find_base_dir()

    input_path = base_dir / "output" / "account_quarter_panel_qoq_processed.csv"
    output_path = base_dir / "output" / "account_quarter_panel_qoq_processed_with_missing_flags.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    df = pd.read_csv(input_path)
    row_count_before = len(df)

    # -----------------------------
    # Satisfaction missing flag
    # -----------------------------
    if "satisfaction_missing_flag" not in df.columns:
        if "avg_satisfaction" in df.columns:
            # coerce helps with empty strings / weird values
            avg_sat = pd.to_numeric(df["avg_satisfaction"], errors="coerce")
            df["satisfaction_missing_flag"] = avg_sat.isna().astype(int)
        else:
            print("Warning: avg_satisfaction missing; creating satisfaction_missing_flag=1 for all rows.")
            df["satisfaction_missing_flag"] = 1

    # -----------------------------
    # Contract missing flag
    # -----------------------------
    if "contract_missing_flag" not in df.columns:
        if "days_to_contract_end_asof_qend" in df.columns:
            d = pd.to_numeric(df["days_to_contract_end_asof_qend"], errors="coerce")
            df["contract_missing_flag"] = d.isna().astype(int)
        else:
            print("Warning: days_to_contract_end_asof_qend missing; creating contract_missing_flag=1 for all rows.")
            df["contract_missing_flag"] = 1

    row_count_after = len(df)

    print(f"Row count before: {row_count_before}")
    print(f"Row count after:  {row_count_after}")

    for col in ["satisfaction_missing_flag", "contract_missing_flag"]:
        if col in df.columns:
            print(f"\nValue counts for {col}:")
            print(df[col].value_counts(dropna=False).sort_index().to_string())
            try:
                print(f"% {col} == 1: {(df[col] == 1).mean():.4f}")
            except Exception:
                pass

    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
