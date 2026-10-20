#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Long-only experiment summarizer.

Outputs only LONG summary files:
  1) auxiliary_metrics_long_summary.csv
  2) main_task_metrics_long_summary.csv
  3) scenario_weather_metrics_long_summary.csv

Usage:
  python summarize_experiments.py --root /home/cyx/FusionSF/MTL/PV_QuadTask_Project/experiments
"""
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd


def parse_experiment_info(file_path: Path):
    parts = list(file_path.parts)
    dataset, model, seed = "unknown_dataset", "unknown_model", None
    for i, p in enumerate(parts):
        m = re.match(r"seed[_-]?(\d+)", p, flags=re.IGNORECASE)
        if m:
            seed = int(m.group(1))
            if i - 1 >= 0:
                model = parts[i - 1]
            if i - 2 >= 0:
                dataset = parts[i - 2]
            break
    if seed is None:
        m = re.search(r"seed[_-]?(\d+)", file_path.name, flags=re.IGNORECASE)
        if m:
            seed = int(m.group(1))
    return dataset, model, seed


def should_skip_file(file_path: Path) -> bool:
    lower_parts = [p.lower() for p in file_path.parts]
    name = file_path.name.lower()
    if "summary" in lower_parts or "summary_long" in lower_parts:
        return True
    if name.startswith("summary_") or "long_summary" in name or "raw_long" in name:
        return True
    return not name.endswith(".csv")


def find_metric_files(root_dir: Path):
    found = {"auxiliary": [], "power": [], "weather": [], "stress": []}
    for f in root_dir.rglob("*.csv"):
        if should_skip_file(f):
            continue
        name = f.name.lower()
        if "auxiliary_metrics" in name:
            found["auxiliary"].append(f)
        elif "power_metrics" in name:
            found["power"].append(f)
        elif "weather_metrics" in name:
            found["weather"].append(f)
        elif "stress_metrics" in name:
            found["stress"].append(f)
    return {k: sorted(v) for k, v in found.items()}


def safe_read_csv(file_path: Path):
    try:
        df = pd.read_csv(file_path)
        if df is None or df.empty:
            print(f"[WARN] Empty CSV: {file_path}")
            return None
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception as e:
        print(f"[WARN] Failed to read {file_path}: {e}")
        return None


def lower_col_map(df):
    return {str(c).strip().lower(): c for c in df.columns}


def to_number(x):
    return pd.to_numeric(x, errors="coerce")


def is_numeric_col(df, col):
    return to_number(df[col]).notna().sum() > 0


def detect_group_col(df, table_type):
    cols = lower_col_map(df)
    if "scenario" in cols:
        return "stress", cols["scenario"]
    if "weather" in cols:
        return "weather", cols["weather"]
    if "weather_name" in cols:
        return "weather", cols["weather_name"]
    if "task" in cols:
        return "auxiliary", cols["task"]
    if "aux_task" in cols:
        return "auxiliary", cols["aux_task"]
    if table_type == "power":
        return "global", None
    return table_type, None


def detect_samples_col(df):
    cols = lower_col_map(df)
    for key in ["samples", "sample_count", "num_samples", "n", "hours", "test_points", "points", "days"]:
        if key in cols:
            return cols[key]
    return None


def normalize_metric_file(file_path: Path, table_type: str):
    df = safe_read_csv(file_path)
    if df is None:
        return pd.DataFrame()

    dataset, model, seed = parse_experiment_info(file_path)
    cols_lower = lower_col_map(df)
    records = []

    # Case A: long table with metric/value
    if "metric" in cols_lower and "value" in cols_lower:
        metric_col = cols_lower["metric"]
        value_col = cols_lower["value"]
        group_type, group_col = detect_group_col(df, table_type)
        samples_col = detect_samples_col(df)
        for _, row in df.iterrows():
            val = to_number(row[value_col])
            if pd.isna(val):
                continue
            group = str(row[group_col]) if group_col is not None else "global"
            samples = to_number(row[samples_col]) if samples_col is not None else np.nan
            records.append({
                "dataset": dataset, "model": model, "seed": seed,
                "group_type": group_type, "group": group,
                "metric": str(row[metric_col]), "value": float(val),
                "samples": samples, "source_file": str(file_path)
            })
        return pd.DataFrame(records)

    # Case B: two-column metric table
    if df.shape[1] == 2:
        c0, c1 = df.columns[0], df.columns[1]
        if is_numeric_col(df, c1) and not is_numeric_col(df, c0):
            group_type, _ = detect_group_col(df, table_type)
            for _, row in df.iterrows():
                val = to_number(row[c1])
                if pd.isna(val):
                    continue
                records.append({
                    "dataset": dataset, "model": model, "seed": seed,
                    "group_type": group_type, "group": "global",
                    "metric": str(row[c0]), "value": float(val),
                    "samples": np.nan, "source_file": str(file_path)
                })
            return pd.DataFrame(records)

    # Case C: wide table
    group_type, group_col = detect_group_col(df, table_type)
    samples_col = detect_samples_col(df)
    id_cols = set()
    if group_col is not None:
        id_cols.add(group_col)
    if samples_col is not None:
        id_cols.add(samples_col)
    for c in df.columns:
        if c.lower() in {"dataset", "model", "seed", "source_file", "file", "path", "run_name", "experiment"}:
            id_cols.add(c)

    numeric_metric_cols = [c for c in df.columns if c not in id_cols and is_numeric_col(df, c)]
    if not numeric_metric_cols:
        print(f"[WARN] No numeric metric columns found in: {file_path}")
        print(f"       Columns: {list(df.columns)}")
        return pd.DataFrame()

    for _, row in df.iterrows():
        group = str(row[group_col]) if group_col is not None else "global"
        samples = to_number(row[samples_col]) if samples_col is not None else np.nan
        for c in numeric_metric_cols:
            val = to_number(row[c])
            if pd.isna(val):
                continue
            records.append({
                "dataset": dataset, "model": model, "seed": seed,
                "group_type": group_type, "group": group,
                "metric": str(c), "value": float(val),
                "samples": samples, "source_file": str(file_path)
            })
    return pd.DataFrame(records)


def aggregate_long(raw_long):
    cols = ["dataset", "model", "group_type", "group", "metric",
            "seed_count", "mean", "std", "var", "min", "max", "samples", "mean±std"]
    if raw_long is None or raw_long.empty:
        return pd.DataFrame(columns=cols)

    group_cols_seed = ["dataset", "model", "group_type", "group", "metric", "seed"]
    seed_level = (raw_long.groupby(group_cols_seed, dropna=False)
                  .agg(value=("value", "mean"), samples=("samples", "max"))
                  .reset_index())

    group_cols = ["dataset", "model", "group_type", "group", "metric"]
    summary = (seed_level.groupby(group_cols, dropna=False)
               .agg(seed_count=("seed", lambda x: x.nunique(dropna=True)),
                    mean=("value", "mean"), std=("value", "std"), var=("value", "var"),
                    min=("value", "min"), max=("value", "max"), samples=("samples", "max"))
               .reset_index())

    # If seed is absent in paths, use number of rows as fallback seed_count
    bad = summary["seed_count"] == 0
    if bad.any():
        fallback = seed_level.groupby(group_cols, dropna=False).size().reset_index(name="_count")
        summary = summary.merge(fallback, on=group_cols, how="left")
        summary.loc[bad, "seed_count"] = summary.loc[bad, "_count"]
        summary = summary.drop(columns=["_count"])

    summary["std"] = summary["std"].fillna(0.0)
    summary["var"] = summary["var"].fillna(0.0)
    summary["samples"] = summary["samples"].fillna("")
    summary["mean±std"] = summary.apply(lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1)
    return summary.sort_values(["dataset", "model", "group_type", "group", "metric"], kind="mergesort").reset_index(drop=True)


def build_summary(files, table_type):
    parts = []
    for f in files:
        one = normalize_metric_file(f, table_type)
        if one is not None and not one.empty:
            parts.append(one)
    raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return raw, aggregate_long(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Experiment root, e.g. /home/cyx/FusionSF/MTL/PV_QuadTask_Project/experiments")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory. Default: {root}/summary_long")
    parser.add_argument("--save-raw", action="store_true",
                        help="Also save raw long records before seed aggregation.")
    args = parser.parse_args()

    root_dir = Path(args.root).resolve()
    out_dir = Path(args.out).resolve() if args.out else root_dir / "summary_long"
    out_dir.mkdir(parents=True, exist_ok=True)

    found = find_metric_files(root_dir)
    print(f"[INFO] Found auxiliary metric files: {len(found['auxiliary'])}")
    print(f"[INFO] Found power metric files: {len(found['power'])}")
    print(f"[INFO] Found weather metric files: {len(found['weather'])}")
    print(f"[INFO] Found stress metric files: {len(found['stress'])}")

    aux_raw, aux_summary = build_summary(found["auxiliary"], "auxiliary")
    power_raw, power_summary = build_summary(found["power"], "power")
    weather_raw, weather_summary = build_summary(found["weather"], "weather")
    stress_raw, stress_summary = build_summary(found["stress"], "stress")

    scenario_summary = pd.concat([weather_summary, stress_summary], ignore_index=True)
    if not scenario_summary.empty:
        scenario_summary = scenario_summary.sort_values(
            ["dataset", "model", "group_type", "group", "metric"], kind="mergesort"
        ).reset_index(drop=True)

    out_aux = out_dir / "auxiliary_metrics_long_summary.csv"
    out_power = out_dir / "main_task_metrics_long_summary.csv"
    out_scenario = out_dir / "scenario_weather_metrics_long_summary.csv"

    aux_summary.to_csv(out_aux, index=False, encoding="utf-8-sig")
    power_summary.to_csv(out_power, index=False, encoding="utf-8-sig")
    scenario_summary.to_csv(out_scenario, index=False, encoding="utf-8-sig")

    print("\n[完成] LONG 汇总结果已保存：")
    print(f"  1) {out_aux}  rows={len(aux_summary)}")
    print(f"  2) {out_power}  rows={len(power_summary)}")
    print(f"  3) {out_scenario}  rows={len(scenario_summary)}")

    if args.save_raw:
        raw_dir = out_dir / "raw_long"
        raw_dir.mkdir(parents=True, exist_ok=True)
        aux_raw.to_csv(raw_dir / "auxiliary_metrics_raw_long.csv", index=False, encoding="utf-8-sig")
        power_raw.to_csv(raw_dir / "main_task_metrics_raw_long.csv", index=False, encoding="utf-8-sig")
        weather_raw.to_csv(raw_dir / "weather_metrics_raw_long.csv", index=False, encoding="utf-8-sig")
        stress_raw.to_csv(raw_dir / "stress_metrics_raw_long.csv", index=False, encoding="utf-8-sig")
        print(f"\n[完成] Raw long records saved to: {raw_dir}")

    if len(aux_summary) == 0 and len(found["auxiliary"]) > 0:
        print("\n[诊断] 找到了 auxiliary_metrics 文件，但汇总为空。")
        print("       请用 --save-raw 或打开一个 auxiliary_metrics CSV 检查列名。")
        print("       常见原因：CSV 里没有数值列，或被错误保存为空。")


if __name__ == "__main__":
    main()
