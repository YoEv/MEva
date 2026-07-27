import argparse
import os
from typing import List

import pandas as pd

from lib.repro.metrics import plot_scatter, safe_pearson_spearman


def expected_experiments(split_tag: str) -> List[str]:
    # CNN-only pipeline: 7 experiment families x {cnn} per split.
    # (Transformer variants were removed.)
    return [
        f"f01_loss_only_cnn_{split_tag}",
        f"f02_entropy_only_cnn_{split_tag}",
        f"f03_sae_only_cnn_{split_tag}",
        f"f04_loss_entropy_cnn_{split_tag}",
        f"f05_entropy_sae_cnn_{split_tag}",
        f"f06_loss_sae_cnn_{split_tag}",
        f"f07_loss_entropy_sae_cnn_{split_tag}",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=str, default="outputs/reports")
    parser.add_argument("--split-tag", type=str, default="clean", choices=["clean"])
    args = parser.parse_args()

    os.makedirs(args.reports_dir, exist_ok=True)
    plot_dir = os.path.join(args.reports_dir, "plots")
    table_dir = os.path.join(args.reports_dir, "tables")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)
    rows = []
    for exp_name in expected_experiments(args.split_tag):
        csv_path = os.path.join(args.reports_dir, f"{exp_name}_test_scores.csv")
        if not os.path.isfile(csv_path):
            rows.append(
                {
                    "experiment": exp_name,
                    "status": "missing_csv",
                    "pearson": float("nan"),
                    "spearman": float("nan"),
                    "n": 0,
                }
            )
            continue

        df = pd.read_csv(csv_path)
        if "score" not in df.columns or "pred" not in df.columns:
            rows.append(
                {
                    "experiment": exp_name,
                    "status": "invalid_csv_columns",
                    "pearson": float("nan"),
                    "spearman": float("nan"),
                    "n": int(len(df)),
                }
            )
            continue
        y_true = df["score"].values.astype(float)
        y_pred = df["pred"].values.astype(float)
        r, rho = safe_pearson_spearman(y_true, y_pred)
        scatter_path = os.path.join(plot_dir, f"scatter_{exp_name}.png")
        plot_scatter(y_true, y_pred, out_path=scatter_path, title=exp_name)
        rows.append(
            {
                "experiment": exp_name,
                "status": "ok",
                "pearson": round(r, 6),
                "spearman": round(rho, 6),
                "n": int(len(df)),
                "csv_path": csv_path,
                "scatter_path": scatter_path,
            }
        )

    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(table_dir, f"aggregate_table_14_experiments_{args.split_tag}.csv")
    out_md = os.path.join(table_dir, f"aggregate_table_14_experiments_{args.split_tag}.md")
    out_df.to_csv(out_csv, index=False)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("| experiment | status | pearson | spearman | n |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for _, row in out_df.iterrows():
            f.write(f"| {row['experiment']} | {row['status']} | {row['pearson']} | {row['spearman']} | {row['n']} |\n")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
