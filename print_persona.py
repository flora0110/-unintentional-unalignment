import os
import json
import pandas as pd
import re

# -------------------------------
# 📁 資料夾設定
# -------------------------------
BASE_DIR = "outputs/persona"

GROUPS = {
    "No_Never": os.path.join(BASE_DIR, "olmo1b_post_sft_dpo_no_vs_never"),
    "Yes_No": os.path.join(BASE_DIR, "olmo1b_post_sft_dpo_yes_vs_no"),
}

# -------------------------------
# 📦 讀取單一 summary.json
# -------------------------------
def load_summary_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            # print(f)
            data = json.load(f)

        # 提取 last_tracked_values
        values = data.get("last_tracked_values", {})
        flat = {k: v.get("value", None) for k, v in values.items()}
        flat["num_model_parameters"] = data.get("num_model_parameters")
        flat["last_epoch"] = data.get("last_epoch")
        flat["num_train_samples"] = data.get("num_train_samples")

        # 提取資料
        d = data.get("data", {})
        flat["input_text"] = d.get("inputs", [""])[0].strip()
        flat["preferred_output"] = d.get("preferred_output", [""])[0]
        flat["dispreferred_output"] = d.get("dispreferred_output", [""])[0]

        # 從資料夾名稱中提取 seed 與 timestamp
        folder_name = os.path.basename(os.path.dirname(path))
        seed_match = re.search(r"seed_(\d+)", folder_name)
        ts_match = re.search(r"(\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2})", folder_name)
        flat["seed"] = seed_match.group(1) if seed_match else ""
        flat["timestamp"] = ts_match.group(1) if ts_match else ""
        flat["exp_name"] = seed_match.group(1) if seed_match else folder_name

        return flat
    except Exception as e:
        print(f"⚠️ Failed to read {path}: {e}")
        return None


# -------------------------------
# 🔍 搜尋所有 summary.json
# -------------------------------
def collect_summaries(base_dir):
    summaries = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == "summary.json":
                full_path = os.path.join(root, file)
                data = load_summary_json(full_path)
                if data:
                    summaries.append(data)
    return pd.DataFrame(summaries)


# -------------------------------
# 🧮 主程式
# -------------------------------
df_no_never = collect_summaries(GROUPS["No_Never"])
print("=------")
df_yes_no = collect_summaries(GROUPS["Yes_No"])

# 標記組別
if not df_no_never.empty:
    df_no_never["group"] = "No vs Never"
if not df_yes_no.empty:
    df_yes_no["group"] = "Yes vs No"

# 合併成一張表
df_all = pd.concat([df_no_never, df_yes_no], ignore_index=True)

# 重新排列欄位
columns_order = [
    "group",
    "preferred logprob change", "dispreferred logprob change",
    "preferred logit", "dispreferred logit",
    "preferred prob", "dispreferred prob",
    
    "train loss",
    "seed", "timestamp", "last_epoch" 
]
df_all = df_all[[c for c in columns_order if c in df_all.columns]]

# -------------------------------
# 💾 儲存成 CSV
# -------------------------------
output_csv = "persona_dpo_summary_table.csv"
df_all.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"✅ Saved summary table to {output_csv}")
print(df_all)

# -------------------------------
# 🧾 輸出 HackMD 友善表格 (不含長文字欄)
# -------------------------------
# -------------------------------
# 🧾 分組輸出 HackMD 友善表格
# -------------------------------
if not df_all.empty:
    for group_name, group_df in df_all.groupby("group"):
        subset_df = group_df.drop(columns=["group"]).reset_index(drop=True)
        print(f"\n🧮 HackMD Table for {group_name} (copy below):\n")
        markdown_table = subset_df.to_markdown(index=False, tablefmt="pipe", floatfmt=".4f")
        print(markdown_table)
else:
    print("⚠️ No data to display.")
