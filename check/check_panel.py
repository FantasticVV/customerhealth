import pandas as pd
from pathlib import Path


def find_panel_file(project_root: Path) -> Path:
    """Find the most likely quarter panel CSV under output/."""
    out_dir = project_root / "output"
    exact = out_dir / "account_quarter_panel_qoq_processed_with_contract_flags.csv"
    if exact.exists():
        return exact

    if not out_dir.exists():
        raise FileNotFoundError(f"output/ folder not found at: {out_dir}")

    # Prefer the most final panel
    candidates = sorted(out_dir.glob("account_quarter_panel*qoq*with_contract_flags*.csv"))
    if candidates:
        return candidates[0]

    candidates2 = sorted(out_dir.glob("account_quarter_panel_qoq_processed*.csv"))
    if candidates2:
        return candidates2[0]

    # Last resort: any account_quarter_panel*
    candidates3 = sorted(out_dir.glob("account_quarter_panel*.csv"))
    if candidates3:
        return candidates3[0]

    raise FileNotFoundError("Cannot find any account_quarter_panel*.csv under output/.")


def pick_quarter_col(df: pd.DataFrame) -> str:
    for c in ["quarter_end", "quarter", "acct_quarter", "account_quarter", "qtr"]:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find a quarter column. Available columns (first 50): {list(df.columns)[:50]}")


def overall_rate_safely(df: pd.DataFrame, label_col: str) -> tuple[float | None, int]:
    """
    Overall churn rate = mean(label) on usable rows only.
    Usable rows = label not null. (Avoid right-censoring problems.)
    """
    if label_col not in df.columns:
        return None, 0
    usable = df.dropna(subset=[label_col])
    if len(usable) == 0:
        return None, 0
    return float(usable[label_col].mean()), int(len(usable))


def by_quarter_table(df: pd.DataFrame, qcol: str, label_col: str) -> pd.DataFrame | None:
    if label_col not in df.columns:
        return None
    usable = df.dropna(subset=[label_col]).copy()
    if usable.empty:
        return None
    out = (
        usable.groupby(qcol)[label_col]
        .agg(rate="mean", n="count")
        .sort_index()
    )
    return out


def main():
    # ✅ 项目根目录 = customerhealthfinal/
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    panel_path = find_panel_file(PROJECT_ROOT)

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("PANEL FILE:", panel_path)
    print("Exists?", panel_path.exists())

    df = pd.read_csv(panel_path)
    print("\nshape (rows, cols):", df.shape)

    # Basic columns
    if "account_id" in df.columns:
        print("unique accounts:", df["account_id"].nunique())
    else:
        print("WARNING: account_id not found. Available columns (first 30):", list(df.columns)[:30])

    qcol = pick_quarter_col(df)
    print("quarter column used:", qcol)
    print("unique quarters:", df[qcol].astype(str).nunique())

    # =========================
    # ✅ Overall churn rates (important)
    # =========================
    nextq_label = "account_churn_event_in_next_quarter"
    rolling2q_label = "churn_label_q2_rolling"

    nextq_rate, nextq_n = overall_rate_safely(df, nextq_label)
    roll2_rate, roll2_n = overall_rate_safely(df, rolling2q_label)

    print("\n==============================")
    print("OVERALL CHURN RATE CHECKS")
    print("==============================")

    if nextq_rate is None:
        print(f"[NEXT-Q] {nextq_label} not found or no usable rows.")
    else:
        print(f"[NEXT-Q] usable rows (label not null): n={nextq_n}")
        print(f"[NEXT-Q] overall churn rate (mean): {nextq_rate:.4f}  ({nextq_rate*100:.2f}%)")
        print(">>> Slide table 'Overall Churn Rate' (IF y = next-quarter) should use this number.")

    if roll2_rate is None:
        print(f"[ROLL2Q] {rolling2q_label} not found or no usable rows.")
    else:
        print(f"[ROLL2Q] usable rows (label not null): n={roll2_n}")
        print(f"[ROLL2Q] overall churn rate (mean): {roll2_rate:.4f}  ({roll2_rate*100:.2f}%)")
        print("Note: rolling 2Q base rate should usually be higher than next-quarter base rate.")

    # =========================
    # ✅ By-quarter tables (matches your plotted lines)
    # =========================
    print("\n==============================")
    print("BY-QUARTER RATES (usable rows only)")
    print("==============================")

    t1 = by_quarter_table(df, qcol, nextq_label)
    if t1 is None:
        print(f"[NEXT-Q] No by-quarter table for {nextq_label}.")
    else:
        print("\n[NEXT-Q] by quarter:")
        print(t1)

    t2 = by_quarter_table(df, qcol, rolling2q_label)
    if t2 is None:
        print(f"\n[ROLL2Q] No by-quarter table for {rolling2q_label}.")
    else:
        print("\n[ROLL2Q] by quarter:")
        print(t2)

    # =========================
    # Missing rates for core fields (keep your original checks)
    # =========================
    print("\n==============================")
    print("MISSING RATE CHECKS")
    print("==============================")
    for c in ["usage_count", "tickets_opened", "arr_amount", "seats"]:
        if c in df.columns:
            print(f"missing rate {c}: {df[c].isna().mean():.4f}")
        else:
            print(f"WARNING: {c} not found in columns")

    # Duplicates
    if {"account_id", qcol}.issubset(df.columns):
        dup = df.duplicated(["account_id", qcol]).sum()
        print(f"\nduplicate rows by (account_id, {qcol}): {dup}")
    else:
        print("\nWARNING: cannot compute duplicates because account_id/quarter column missing.")


if __name__ == "__main__":
    main()
