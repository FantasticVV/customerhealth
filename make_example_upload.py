import pandas as pd
from pathlib import Path

base_dir = Path(__file__).resolve().parent
src = base_dir / "output" / "account_quarter_panel_qoq_processed_with_contract_flags.csv"
out = base_dir / "output" / "example_upload.csv"

df = pd.read_csv(src)

# 每个 account 取最新 quarter_end（更像“在线打分”）
df["quarter_end"] = pd.to_datetime(df["quarter_end"])
latest = df[df["quarter_end"] == df.groupby("account_id")["quarter_end"].transform("max")].copy()

# 你streamlit里会用到的展示列（有就保留）
extra_cols = [c for c in ["plan_tier", "company_size", "seats", "mrr_amount", "industry"] if c in latest.columns]

# 模型训练用的特征列（与你脚本 features 对齐）
feature_cols = [
    "usage_count_qoq_delta_z",
    "tickets_opened_qoq_delta_z",
    "usage_drop_flag",
    "days_to_contract_end_capped",   # 注意：如果你原表里没有这个列，需要先在build_feature_set后保存
    "arr_amount_qoq_delta_z",
    "seats_qoq_delta_z",
    "seats",
    "tenure",
    "avg_satisfaction",
    "satisfaction_missing_flag",
    "contract_missing_flag",
    "tenure_x_usage_drop_flag",
    "plan_tier",
]

keep = ["account_id"] + [c for c in feature_cols if c in latest.columns] + extra_cols

# 如果缺列，先提示（这样你知道为什么上传会报错）
missing = [c for c in feature_cols if c not in latest.columns]
if missing:
    print("WARNING: example_upload.csv is missing some feature columns:")
    for m in missing:
        print(" -", m)

latest[keep].to_csv(out, index=False)
print("Saved:", out)
