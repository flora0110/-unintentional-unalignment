# corr_prefsim.py
import argparse, json, math
from pathlib import Path
import pandas as pd
import numpy as np

METRICS = ["ches_score", "ln_ches_score", "last_hidden_embedding_inner_prod"]

def is_num(x):
    return isinstance(x, (int, float)) and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))

def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON 需為 list 格式")
    rows = []
    for rec in data:
        if all(m in rec and is_num(rec[m]) for m in METRICS):
            rows.append({m: float(rec[m]) for m in METRICS})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="train_with_prefsim.json")
    args = ap.parse_args()

    df = load_records(args.path)
    n_total = sum(1 for _ in json.load(open(args.path, "r", encoding="utf-8")))
    print(f"總筆數: {n_total} | 可用筆數(三指標皆有效): {len(df)}")

    # Pearson correlation
    pearson = df.corr(method="pearson").loc[METRICS, METRICS]
    # Spearman correlation (rank-based)
    spearman = df.corr(method="spearman").loc[METRICS, METRICS]

    # 只列出兩兩上三角結果
    pairs = [("ches_score","ln_ches_score"),
             ("ches_score","last_hidden_embedding_inner_prod"),
             ("ln_ches_score","last_hidden_embedding_inner_prod")]

    def to_rows(mat, name):
        return [{"pair": f"{a} ~ {b}", name: float(mat.loc[a,b])} for (a,b) in pairs]

    table = pd.DataFrame(to_rows(pearson, "pearson_r")).merge(
        pd.DataFrame(to_rows(spearman, "spearman_rho")),
        on="pair"
    ).sort_values("pair")

    # 印表
    pd.set_option("display.precision", 6)
    print("\nPairwise correlations (higher=更線性一致; Spearman 看單調關係):")
    print(table.to_string(index=False))

    # 存檔
    out_dir = Path(args.path).parent
    table.to_csv(out_dir / "prefsim_pairwise_correlations.csv", index=False)
    pearson.to_csv(out_dir / "prefsim_pearson_matrix.csv")
    spearman.to_csv(out_dir / "prefsim_spearman_matrix.csv")
    print(f"\n已輸出：\n- {out_dir/'prefsim_pairwise_correlations.csv'}"
          f"\n- {out_dir/'prefsim_pearson_matrix.csv'}"
          f"\n- {out_dir/'prefsim_spearman_matrix.csv'}")

if __name__ == "__main__":
    main()
