import argparse
import os

import pandas as pd

from lib.repro.data_paths import load_exp11_splits
from lib.repro.metrics import plot_scatter, safe_pearson_spearman
from lib.repro.scaling import RESCALE_METHODS, apply_rescale

AXES = ["CE", "CU", "PC", "PQ"]


def _build_pred_paths(root: str, label_mode: str):
    return {
        "aesthetics_val": os.path.join(root, "aesthetics_scores_val.csv"),
        "aesthetics_test": os.path.join(root, "aesthetics_scores_test.csv"),
        "mean_loss_val": os.path.join(root, "mean_loss_val.csv"),
        "mean_loss_test": os.path.join(root, "mean_loss_test.csv"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--methods", type=str, nargs="+", default=list(RESCALE_METHODS.keys()))
    parser.add_argument("--label_mode", type=str, default="clean", choices=["clean"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plot_dir = os.path.join(args.out_dir, "plots")
    table_dir = os.path.join(args.out_dir, "tables")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)
    label_modes = [args.label_mode]

    for label_mode in label_modes:
        pred_paths = _build_pred_paths(args.out_dir, label_mode)
        split_map = load_exp11_splits(args.config, splits=label_mode)
        human_val = split_map["val"]["score"].values.astype(float)
        human_test = split_map["test"]["score"].values.astype(float)
        rows = []

        if os.path.exists(pred_paths["aesthetics_test"]) and os.path.exists(pred_paths["aesthetics_val"]):
            aes_val = pd.read_csv(pred_paths["aesthetics_val"])
            aes_test = pd.read_csv(pred_paths["aesthetics_test"])
            for ax in AXES:
                if ax not in aes_val.columns or ax not in aes_test.columns:
                    continue
                p_val = aes_val[ax].values.astype(float)
                p_test = aes_test[ax].values.astype(float)
                for method in args.methods:
                    if method not in RESCALE_METHODS:
                        continue
                    p_test_r = apply_rescale(method, p_val, human_val, p_test)
                    r, rho = safe_pearson_spearman(human_test, p_test_r)
                    rows.append(
                        {
                            "LabelMode": label_mode,
                            "Method": f"Aesthetics-{ax}",
                            "RescaleMethod": method,
                            "Pearson": round(r, 4),
                            "Spearman": round(rho, 4),
                        }
                    )
                    out_path = os.path.join(plot_dir, f"scatter_aesthetics_{ax}_rescaled_{method}_{label_mode}.png")
                    plot_scatter(human_test, p_test_r, out_path=out_path, title=f"[{label_mode}] Aesthetics-{ax} ({method})")

        if os.path.exists(pred_paths["mean_loss_test"]) and os.path.exists(pred_paths["mean_loss_val"]):
            ml_val = pd.read_csv(pred_paths["mean_loss_val"])
            ml_test = pd.read_csv(pred_paths["mean_loss_test"])
            p_val = ml_val["mean_loss"].values.astype(float)
            p_test = ml_test["mean_loss"].values.astype(float)
            for method in args.methods:
                if method not in RESCALE_METHODS:
                    continue
                p_test_r = apply_rescale(method, p_val, human_val, p_test)
                r, rho = safe_pearson_spearman(human_test, p_test_r)
                rows.append(
                    {
                        "LabelMode": label_mode,
                        "Method": "Mean Loss",
                        "RescaleMethod": method,
                        "Pearson": round(r, 4),
                        "Spearman": round(rho, 4),
                    }
                )
                out_path = os.path.join(plot_dir, f"scatter_mean_loss_rescaled_{method}_{label_mode}.png")
                plot_scatter(human_test, p_test_r, out_path=out_path, title=f"[{label_mode}] Mean Loss ({method})")

        if not rows:
            print(f"[{label_mode}] no rows generated")
            continue
        table_df = pd.DataFrame(rows)
        csv_path = os.path.join(table_dir, f"aggregate_table_rescaled_{label_mode}.csv")
        md_path = os.path.join(table_dir, f"aggregate_table_rescaled_{label_mode}.md")
        table_df.to_csv(csv_path, index=False)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("| LabelMode | Method | RescaleMethod | Pearson | Spearman |\n")
            f.write("|-----------|--------|---------------|--------:|--------:|\n")
            for _, row in table_df.iterrows():
                f.write(
                    f"| {row['LabelMode']} | {row['Method']} | {row['RescaleMethod']} | {row['Pearson']} | {row['Spearman']} |\n"
                )
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
