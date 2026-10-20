#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset-wise experiment summarizer.

Difference from the old long-only summarizer
-------------------------------------------
The old script searched the whole experiment root and wrote one global summary
folder. This script treats every dataset folder under --root independently:

  experiments/
    private/
      ... metric csv files ...
      summary_long/
        auxiliary_metrics_long_summary.csv
        main_task_metrics_long_summary.csv
        scenario_weather_metrics_long_summary.csv
        all_metrics_long_summary.csv
    public_60815/
      ... metric csv files ...
      summary_long/
        ...

It does NOT merge different datasets into one global analysis file.

Typical usage
-------------
python summarize_experiments_dataset.py \
  --root /home/cyx/FusionSF/MTL/PV_QuadTask_Project/experiments

Only process selected datasets:
python summarize_experiments_by_dataset.py \
  --root /home/cyx/FusionSF/MTL/PV_QuadTask_Project/experiments \
  --datasets private public_60815

If --root itself is one dataset folder:
python summarize_experiments_by_dataset.py \
  --root /home/cyx/FusionSF/MTL/PV_QuadTask_Project/experiments/private \
  --root-is-dataset
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


SUMMARY_DIR_NAMES = {"summary", "summary_long", "raw_long", "summaries", "aggregate", "aggregated"}
METRIC_KEYS = {
    "auxiliary": "auxiliary_metrics",
    "power": "power_metrics",
    "weather": "weather_metrics",
    "stress": "stress_metrics",
}


def is_summary_path(file_path: Path) -> bool:
    lower_parts = [p.lower() for p in file_path.parts]
    name = file_path.name.lower()
    if any(p in SUMMARY_DIR_NAMES for p in lower_parts):
        return True
    if name.startswith("summary_") or "long_summary" in name or "raw_long" in name:
        return True
    return False


def should_skip_file(file_path: Path) -> bool:
    if file_path.suffix.lower() != ".csv":
        return True
    return is_summary_path(file_path)


def has_metric_files(folder: Path) -> bool:
    for f in folder.rglob("*.csv"):
        if should_skip_file(f):
            continue
        name = f.name.lower()
        if any(key in name for key in METRIC_KEYS.values()):
            return True
    return False


def discover_dataset_dirs(root_dir: Path, root_is_dataset: bool = False, selected: Optional[List[str]] = None) -> List[Path]:
    """Return dataset folders to summarize.

    Default behavior: each immediate subdirectory under root is treated as one
    dataset if it contains metric files recursively. This avoids merging all
    datasets into a single global output.
    """
    root_dir = root_dir.resolve()
    selected_set = set(selected or [])

    if root_is_dataset:
        dataset_dirs = [root_dir]
    else:
        children = [d for d in root_dir.iterdir() if d.is_dir() and d.name.lower() not in SUMMARY_DIR_NAMES]
        dataset_dirs = [d for d in children if has_metric_files(d)]

        # If no child dataset folder is found, fall back to treating root as one dataset.
        if not dataset_dirs and has_metric_files(root_dir):
            dataset_dirs = [root_dir]

    if selected_set:
        dataset_dirs = [d for d in dataset_dirs if d.name in selected_set]

    return sorted(dataset_dirs, key=lambda p: p.name)


def parse_experiment_info(file_path: Path, dataset_root: Path, dataset_name: str) -> Tuple[str, str, Optional[int]]:
    """Infer dataset/model/seed from one metric file path.

    dataset is fixed by dataset_root. model is normally the parent directory of
    seed_x, e.g. dataset/model/seed_0/metrics/xxx.csv.
    """
    parts = list(file_path.parts)
    dataset, model, seed = dataset_name, "unknown_model", None

    for i, p in enumerate(parts):
        m = re.match(r"seed[_-]?(\d+)$", p, flags=re.IGNORECASE)
        if m:
            seed = int(m.group(1))
            if i - 1 >= 0:
                model = parts[i - 1]
            break

    if seed is None:
        m = re.search(r"seed[_-]?(\d+)", file_path.name, flags=re.IGNORECASE)
        if m:
            seed = int(m.group(1))

    if model == "unknown_model":
        # Fallback: choose the first path element under dataset_root as model.
        try:
            rel = file_path.relative_to(dataset_root)
            if len(rel.parts) >= 2:
                # If path is model/metrics/file.csv, rel.parts[0] is model.
                model = rel.parts[0]
            elif len(rel.parts) == 1:
                model = file_path.stem
        except ValueError:
            pass

    return dataset, model, seed


def find_metric_files(dataset_dir: Path) -> Dict[str, List[Path]]:
    found = {"auxiliary": [], "power": [], "weather": [], "stress": []}
    for f in dataset_dir.rglob("*.csv"):
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


def safe_read_csv(file_path: Path) -> Optional[pd.DataFrame]:
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


def lower_col_map(df: pd.DataFrame) -> Dict[str, str]:
    return {str(c).strip().lower(): c for c in df.columns}


def to_number(x):
    return pd.to_numeric(x, errors="coerce")


def is_numeric_col(df: pd.DataFrame, col: str) -> bool:
    return to_number(df[col]).notna().sum() > 0


def detect_group_col(df: pd.DataFrame, table_type: str) -> Tuple[str, Optional[str]]:
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


def detect_samples_col(df: pd.DataFrame) -> Optional[str]:
    cols = lower_col_map(df)
    for key in ["samples", "sample_count", "num_samples", "n", "hours", "test_points", "points", "days"]:
        if key in cols:
            return cols[key]
    return None


def normalize_metric_file(file_path: Path, table_type: str, dataset_root: Path, dataset_name: str) -> pd.DataFrame:
    df = safe_read_csv(file_path)
    if df is None:
        return pd.DataFrame()

    dataset, model, seed = parse_experiment_info(file_path, dataset_root, dataset_name)
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
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "table_type": table_type,
                "group_type": group_type,
                "group": group,
                "metric": str(row[metric_col]),
                "value": float(val),
                "samples": samples,
                "source_file": str(file_path),
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
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "table_type": table_type,
                    "group_type": group_type,
                    "group": "global",
                    "metric": str(row[c0]),
                    "value": float(val),
                    "samples": np.nan,
                    "source_file": str(file_path),
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
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "table_type": table_type,
                "group_type": group_type,
                "group": group,
                "metric": str(c),
                "value": float(val),
                "samples": samples,
                "source_file": str(file_path),
            })
    return pd.DataFrame(records)


def aggregate_long(raw_long: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset", "model", "table_type", "group_type", "group", "metric",
        "seed_count", "mean", "std", "var", "min", "max", "samples", "mean±std",
    ]
    if raw_long is None or raw_long.empty:
        return pd.DataFrame(columns=cols)

    group_cols_seed = ["dataset", "model", "table_type", "group_type", "group", "metric", "seed"]
    seed_level = (
        raw_long.groupby(group_cols_seed, dropna=False)
        .agg(value=("value", "mean"), samples=("samples", "max"))
        .reset_index()
    )

    group_cols = ["dataset", "model", "table_type", "group_type", "group", "metric"]
    summary = (
        seed_level.groupby(group_cols, dropna=False)
        .agg(
            seed_count=("seed", lambda x: x.nunique(dropna=True)),
            mean=("value", "mean"),
            std=("value", "std"),
            var=("value", "var"),
            min=("value", "min"),
            max=("value", "max"),
            samples=("samples", "max"),
        )
        .reset_index()
    )

    # If seed is absent in paths, use number of rows as fallback seed_count.
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

    return summary.sort_values(
        ["dataset", "model", "table_type", "group_type", "group", "metric"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_summary(files: Iterable[Path], table_type: str, dataset_root: Path, dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    for f in files:
        one = normalize_metric_file(f, table_type, dataset_root, dataset_name)
        if one is not None and not one.empty:
            parts.append(one)
    raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return raw, aggregate_long(raw)


def summarize_one_dataset(dataset_dir: Path, out_name: str, save_raw: bool = False) -> None:
    dataset_dir = dataset_dir.resolve()
    dataset_name = dataset_dir.name
    out_dir = dataset_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    found = find_metric_files(dataset_dir)
    print("\n" + "=" * 100)
    print(f"[DATASET] {dataset_name}")
    print(f"[PATH]    {dataset_dir}")
    print(f"[INFO] Found auxiliary metric files: {len(found['auxiliary'])}")
    print(f"[INFO] Found power metric files: {len(found['power'])}")
    print(f"[INFO] Found weather metric files: {len(found['weather'])}")
    print(f"[INFO] Found stress metric files: {len(found['stress'])}")

    aux_raw, aux_summary = build_summary(found["auxiliary"], "auxiliary", dataset_dir, dataset_name)
    power_raw, power_summary = build_summary(found["power"], "power", dataset_dir, dataset_name)
    weather_raw, weather_summary = build_summary(found["weather"], "weather", dataset_dir, dataset_name)
    stress_raw, stress_summary = build_summary(found["stress"], "stress", dataset_dir, dataset_name)

    scenario_summary = pd.concat([weather_summary, stress_summary], ignore_index=True)
    if not scenario_summary.empty:
        scenario_summary = scenario_summary.sort_values(
            ["dataset", "model", "table_type", "group_type", "group", "metric"],
            kind="mergesort",
        ).reset_index(drop=True)

    all_summary = pd.concat(
        [aux_summary, power_summary, scenario_summary], ignore_index=True
    )
    if not all_summary.empty:
        all_summary = all_summary.sort_values(
            ["dataset", "model", "table_type", "group_type", "group", "metric"],
            kind="mergesort",
        ).reset_index(drop=True)

    out_aux = out_dir / "auxiliary_metrics_long_summary.csv"
    out_power = out_dir / "main_task_metrics_long_summary.csv"
    out_scenario = out_dir / "scenario_weather_metrics_long_summary.csv"
    out_all = out_dir / "all_metrics_long_summary.csv"

    aux_summary.to_csv(out_aux, index=False, encoding="utf-8-sig")
    power_summary.to_csv(out_power, index=False, encoding="utf-8-sig")
    scenario_summary.to_csv(out_scenario, index=False, encoding="utf-8-sig")
    all_summary.to_csv(out_all, index=False, encoding="utf-8-sig")

    print("[DONE] Dataset-wise summary saved:")
    print(f"  1) {out_aux} rows={len(aux_summary)}")
    print(f"  2) {out_power} rows={len(power_summary)}")
    print(f"  3) {out_scenario} rows={len(scenario_summary)}")
    print(f"  4) {out_all} rows={len(all_summary)}")

    if save_raw:
        raw_dir = out_dir / "raw_long"
        raw_dir.mkdir(parents=True, exist_ok=True)
        aux_raw.to_csv(raw_dir / "auxiliary_metrics_raw_long.csv", index=False, encoding="utf-8-sig")
        power_raw.to_csv(raw_dir / "main_task_metrics_raw_long.csv", index=False, encoding="utf-8-sig")
        weather_raw.to_csv(raw_dir / "weather_metrics_raw_long.csv", index=False, encoding="utf-8-sig")
        stress_raw.to_csv(raw_dir / "stress_metrics_raw_long.csv", index=False, encoding="utf-8-sig")
        print(f"[DONE] Raw long records saved to: {raw_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Experiment root. Usually the parent folder that contains dataset folders, e.g. .../experiments",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default="summary_long",
        help="Output folder name created inside each dataset folder. Default: summary_long",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset folder names to process. Example: --datasets private public_60815",
    )
    parser.add_argument(
        "--root-is-dataset",
        action="store_true",
        help="Treat --root itself as one dataset folder instead of discovering immediate child dataset folders.",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Also save raw long records inside each dataset summary folder.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root).resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"Root path does not exist: {root_dir}")

    dataset_dirs = discover_dataset_dirs(root_dir, root_is_dataset=args.root_is_dataset, selected=args.datasets)
    if not dataset_dirs:
        raise RuntimeError(
            f"No dataset folders with metric CSV files found under {root_dir}. "
            f"If root itself is a dataset folder, rerun with --root-is-dataset."
        )

    print(f"[INFO] Root: {root_dir}")
    print(f"[INFO] Datasets to summarize: {[d.name for d in dataset_dirs]}")

    for dataset_dir in dataset_dirs:
        summarize_one_dataset(dataset_dir, out_name=args.out_name, save_raw=args.save_raw)

    print("\n[完成] 已按数据集分别汇总。没有生成跨数据集的总汇总文件。")


if __name__ == "__main__":
    main()
