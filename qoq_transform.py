from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    base_dir = Path(__file__).resolve().parent  # ✅ Backend Code 目录
    input_path = base_dir / "output" / "account_quarter_panel.csv"
    output_path = base_dir / "output" / "account_quarter_panel_qoq_processed.csv"

    df = pd.read_csv(input_path)

    # Identify QoQ columns for processing (numeric only)
    qoq_cols = [
        c
        for c in df.columns
        if (("_qoq_delta" in c) or ("_qoq_change" in c)) and pd.api.types.is_numeric_dtype(df[c])
    ]
    print("QoQ columns detected:")
    for col in qoq_cols:
        print(f"- {col}")

    # Raw QoQ deltas are heavy-tailed; we clip extremes and standardize the clipped values.
    summary_rows = []
    for col in qoq_cols:
        series = df[col]

        # Use 1st/99th percentile clipping to stabilize variance while preserving signal.
        p1 = series.quantile(0.01)
        p99 = series.quantile(0.99)

        clipped_col = f"{col}_clipped"
        df[clipped_col] = series.clip(lower=p1, upper=p99)

        mean_clipped = df[clipped_col].mean()
        std_clipped = df[clipped_col].std(ddof=0)
        z_col = f"{col}_z"
        if std_clipped == 0 or np.isnan(std_clipped):
            df[z_col] = 0.0
        else:
            df[z_col] = (df[clipped_col] - mean_clipped) / std_clipped

        # Directional drop flag (captures sign only)
        drop_flag_col = f"{col}_drop_flag"
        df[drop_flag_col] = (series < 0).astype(int)

        pct_negative = (series < 0).mean()
        summary_rows.append(
            {
                "feature": col,
                "p1": p1,
                "p99": p99,
                "mean_clipped": mean_clipped,
                "std_clipped": std_clipped,
                "pct_negative": pct_negative,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    print("\nQoQ preprocessing summary:")
    print(summary_df.to_string(index=False))

    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
