#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust significance tests for PV forecasting experiments.

Directory expected:
experiments/{dataset}/{model}/seed_{seed}/predictions/*detailed_predictions*.csv

Example:
python significance_tests.py \
  --root /home/cyx/FusionSF/MTL/PV_QuadTask_Project/experiments \
  --ours OURS \
  --baselines Baseline Baseline_large
"""

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def parse_experiment_info(file_path: Path) -> Tuple[str, str, Optional[int]]:
    parts = list(file_path.parts)

    dataset = "unknown_dataset"
    model = "unknown_model"
    seed = None

    for i, p in enumerate(parts):
        m = re.match(r"seed[_-]?(\d+)", str(p), flags=re.IGNORECASE)
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

    return str(dataset), str(model), seed


def find_prediction_files(root: Path) -> List[Path]:
    files = []
    for f in root.rglob("*.csv"):
        name = f.name.lower()
        if "summary" in [p.lower() for p in f.parts]:
            continue
        if "detailed_predictions" in name:
            files.append(f)
    return sorted(files)


def normalize_col_map(df: pd.DataFrame) -> Dict[str, str]:
    return {str(c).strip().lower(): c for c in df.columns}


def pick_col(df: pd.DataFrame, candidates: List[str], required: bool = False) -> Optional[str]:
    cmap = normalize_col_map(df)
    for cand in candidates:
        if cand.lower() in cmap:
            return cmap[cand.lower()]
    if required:
        raise ValueError(f"Missing required column. Tried={candidates}; Available={list(df.columns)}")
    return None


def safe_bool_series(x, length: int):
    if x is None:
        return pd.Series([False] * length)
    if isinstance(x, pd.Series):
        if x.dtype == bool:
            return x
        return x.astype(str).str.lower().isin(["1", "true", "yes", "y", "t"])
    return pd.Series([False] * length)


def load_all_predictions(root: Path, debug: bool = True) -> pd.DataFrame:
    files = find_prediction_files(root)
    if debug:
        print(f"[DEBUG] detailed prediction files found: {len(files)}")
        for f in files[:10]:
            dataset, model, seed = parse_experiment_info(f)
            print(f"  - {f}")
            print(f"    parsed: dataset={dataset}, model={model}, seed={seed}")

    if not files:
        raise FileNotFoundError(f"No *detailed_predictions*.csv files found under: {root}")

    rows = []
    bad_files = []

    for f in files:
        dataset, model, seed = parse_experiment_info(f)

        try:
            df = pd.read_csv(f)
        except Exception as e:
            bad_files.append((str(f), f"read error: {e}"))
            continue

        if df.empty:
            bad_files.append((str(f), "empty csv"))
            continue

        df.columns = [str(c).strip() for c in df.columns]

        try:
            y_true_col = pick_col(df, [
                "y_true_MW", "Ground_Truth", "Ground_Truth_MW",
                "y_true", "true", "Power_true", "Truth", "Actual"
            ], required=True)

            y_pred_col = pick_col(df, [
                "y_pred_MW", "Prediction", "Prediction_MW",
                "y_pred", "pred", "Power_pred", "Forecast"
            ], required=True)
        except Exception as e:
            bad_files.append((str(f), f"column error: {e}"))
            continue

        target_time_col = pick_col(df, ["target_time", "Time", "time", "datetime", "date_time"])
        init_time_col = pick_col(df, ["init_time", "forecast_time", "start_time"])
        horizon_col = pick_col(df, ["horizon", "step", "lead_time"])
        sample_id_col = pick_col(df, ["sample_id", "idx", "index"])

        weather_true_col = pick_col(df, ["weather_true_name", "Weather", "weather_true", "weather"])
        weather_pred_col = pick_col(df, ["weather_pred_name", "weather_pred"])
        dhi_true_col = pick_col(df, ["dhi_true", "DHI_true", "dfi_true", "DFI_true"])
        ramp_true_col = pick_col(df, ["ramp_true_id", "ramp_true", "ramp_label", "target_ramp"])
        ramp_pred_col = pick_col(df, ["ramp_pred_id", "ramp_pred"])
        snow_col = pick_col(df, ["snowfall", "snowfall_true", "snow"])
        precip_col = pick_col(df, ["precipitation", "precipitation_true", "rain"])
        high_dhi_col = pick_col(df, ["high_dhi_mask", "high_DHI_mask"])
        high_ramp_col = pick_col(df, ["high_ramp_mask", "high_Ramp_mask"])

        # IMPORTANT:
        # Do not assign scalar columns to an empty DataFrame before assigning array columns.
        # In pandas, scalar assignment to an empty DataFrame creates a zero-length column;
        # after later array assignments, these scalar columns become NaN.
        # Therefore we first create the correct number of rows, then assign metadata.
        out = pd.DataFrame(index=np.arange(len(df)))
        out["dataset"] = dataset
        out["model"] = model
        out["seed"] = seed

        if sample_id_col:
            out["sample_id"] = df[sample_id_col].values
        else:
            out["sample_id"] = np.arange(len(df))

        if target_time_col:
            out["target_time"] = pd.to_datetime(df[target_time_col], errors="coerce")
        else:
            out["target_time"] = pd.NaT

        if init_time_col:
            out["init_time"] = pd.to_datetime(df[init_time_col], errors="coerce")
        else:
            out["init_time"] = pd.NaT

        if horizon_col:
            out["horizon"] = pd.to_numeric(df[horizon_col], errors="coerce").fillna(-1).astype(int)
        else:
            out["horizon"] = -1

        out["y_true"] = pd.to_numeric(df[y_true_col], errors="coerce")
        out["y_pred"] = pd.to_numeric(df[y_pred_col], errors="coerce")

        valid_rows = out[["y_true", "y_pred"]].dropna().shape[0]
        if valid_rows == 0:
            bad_files.append((str(f), f"all y_true/y_pred are NaN; columns used: {y_true_col}, {y_pred_col}"))
            continue

        out["abs_error"] = (out["y_true"] - out["y_pred"]).abs()
        out["sq_error"] = (out["y_true"] - out["y_pred"]) ** 2

        out["weather_true"] = df[weather_true_col].astype(str).values if weather_true_col else ""
        out["weather_pred"] = df[weather_pred_col].astype(str).values if weather_pred_col else ""

        out["dhi_true"] = pd.to_numeric(df[dhi_true_col], errors="coerce") if dhi_true_col else np.nan
        out["ramp_true"] = pd.to_numeric(df[ramp_true_col], errors="coerce") if ramp_true_col else np.nan
        out["ramp_pred"] = pd.to_numeric(df[ramp_pred_col], errors="coerce") if ramp_pred_col else np.nan
        out["snowfall"] = pd.to_numeric(df[snow_col], errors="coerce") if snow_col else np.nan
        out["precipitation"] = pd.to_numeric(df[precip_col], errors="coerce") if precip_col else np.nan
        out["high_dhi_mask"] = safe_bool_series(df[high_dhi_col], len(df)).values if high_dhi_col else False
        out["high_ramp_mask"] = safe_bool_series(df[high_ramp_col], len(df)).values if high_ramp_col else False

        out = out.dropna(subset=["y_true", "y_pred"]).reset_index(drop=True)
        rows.append(out)

    if bad_files:
        print("\n[WARN] Some files were skipped:")
        for p, reason in bad_files[:20]:
            print(f"  - {p}")
            print(f"    reason: {reason}")
        if len(bad_files) > 20:
            print(f"  ... and {len(bad_files) - 20} more")

    if not rows:
        raise RuntimeError(
            "No valid prediction rows loaded. Check column names in detailed_predictions CSV."
        )

    all_df = pd.concat(rows, ignore_index=True)

    if debug:
        print("\n[DEBUG] Loaded prediction rows:", len(all_df))
        print("[DEBUG] Datasets:", sorted(all_df["dataset"].unique().tolist()))
        print("[DEBUG] Models:", sorted(all_df["model"].unique().tolist()))
        print("[DEBUG] Seeds:", sorted(all_df["seed"].dropna().unique().tolist()))

        print("\n[DEBUG] Rows by dataset/model/seed:")
        print(
            all_df.groupby(["dataset", "model", "seed"])
            .size()
            .reset_index(name="rows")
            .to_string(index=False)
        )

    return all_df


def contains_any(series: pd.Series, keywords: List[str]) -> pd.Series:
    s = series.fillna("").astype(str).str.lower()
    mask = pd.Series(False, index=series.index)
    for kw in keywords:
        mask = mask | s.str.contains(kw.lower(), regex=False)
    return mask


def add_dynamic_scenario_masks(df_one_ref: pd.DataFrame) -> pd.DataFrame:
    """
    Create scenario masks from aligned prediction dataframe.

    This version does not require sample_id. It uses an internal _row_id
    to avoid KeyError after pandas merge creates sample_id_x/sample_id_y.
    """
    df = df_one_ref.copy().reset_index(drop=True)
    df["_row_id"] = np.arange(len(df))

    df["mask_global"] = True

    # -----------------------------
    # Weather masks
    # -----------------------------
    if "weather_true" in df.columns:
        df["mask_snow"] = contains_any(df["weather_true"], ["降雪", "snow"])
        df["mask_rain"] = contains_any(df["weather_true"], ["降雨", "rain"])
        df["mask_cloudy"] = contains_any(df["weather_true"], ["多云", "cloud"])
        df["mask_sunny"] = contains_any(df["weather_true"], ["晴", "sun"])
    else:
        df["mask_snow"] = False
        df["mask_rain"] = False
        df["mask_cloudy"] = False
        df["mask_sunny"] = False

    # If explicit snow/precipitation columns exist, use them as supplements
    if "snowfall" in df.columns and df["snowfall"].notna().sum() > 0:
        df["mask_snow"] = df["mask_snow"] | (df["snowfall"].fillna(0) > 0)

    if "precipitation" in df.columns and df["precipitation"].notna().sum() > 0:
        df["mask_rain"] = df["mask_rain"] | (df["precipitation"].fillna(0) > 0)

    # -----------------------------
    # High DHI mask
    # -----------------------------
    if "high_dhi_mask" in df.columns and df["high_dhi_mask"].sum() > 0:
        df["mask_high_DHI_top10"] = df["high_dhi_mask"].astype(bool)
    elif "dhi_true" in df.columns and df["dhi_true"].notna().sum() > 10:
        dhi_thr = df["dhi_true"].quantile(0.90)
        df["mask_high_DHI_top10"] = df["dhi_true"] >= dhi_thr
    else:
        df["mask_high_DHI_top10"] = False

    # -----------------------------
    # High Ramp mask
    # -----------------------------
    if "high_ramp_mask" in df.columns and df["high_ramp_mask"].sum() > 0:
        df["mask_high_Ramp_top5"] = df["high_ramp_mask"].astype(bool)
    else:
        # Use available columns for chronological sorting
        sort_cols = []
        if "target_time" in df.columns and df["target_time"].notna().sum() > 0:
            sort_cols.append("target_time")
        if "horizon" in df.columns:
            sort_cols.append("horizon")

        # Always include internal row id as a stable fallback
        sort_cols.append("_row_id")

        tmp = df.sort_values(sort_cols).copy()
        tmp["_abs_dp"] = tmp["y_true"].diff().abs()

        if tmp["_abs_dp"].notna().sum() > 10:
            ramp_thr = tmp["_abs_dp"].quantile(0.95)
            tmp["mask_high_Ramp_top5"] = tmp["_abs_dp"] >= ramp_thr
        else:
            tmp["mask_high_Ramp_top5"] = False

        # Restore original order by _row_id
        tmp = tmp[["_row_id", "mask_high_Ramp_top5"]]
        df = df.merge(tmp, on="_row_id", how="left")
        df["mask_high_Ramp_top5"] = df["mask_high_Ramp_top5"].fillna(False).astype(bool)

    # -----------------------------
    # Top 5% true power mask
    # -----------------------------
    if "y_true" in df.columns and df["y_true"].notna().sum() > 10:
        power_thr = df["y_true"].quantile(0.95)
        df["mask_top5_true_power"] = df["y_true"] >= power_thr
    else:
        df["mask_top5_true_power"] = False

    # -----------------------------
    # Composite mask
    # -----------------------------
    df["mask_snow_high_Ramp"] = df["mask_snow"] & df["mask_high_Ramp_top5"]

    return df.drop(columns=["_row_id"], errors="ignore")


def get_scenario_masks(df_ref: pd.DataFrame, min_samples: int = 20) -> Dict[str, pd.Series]:
    df = add_dynamic_scenario_masks(df_ref)
    scenarios = {
        "global": df["mask_global"],
        "snow": df["mask_snow"],
        "rain": df["mask_rain"],
        "cloudy": df["mask_cloudy"],
        "sunny": df["mask_sunny"],
        "high_DHI_top10": df["mask_high_DHI_top10"],
        "high_Ramp_top5": df["mask_high_Ramp_top5"],
        "top5_true_power": df["mask_top5_true_power"],
        "snow_high_Ramp": df["mask_snow_high_Ramp"],
    }

    valid = {}
    for k, m in scenarios.items():
        m = m.fillna(False).astype(bool)
        if int(m.sum()) >= min_samples:
            valid[k] = m
    return valid


def normal_two_sided_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def newey_west_variance(d: np.ndarray, lag: int) -> float:
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n <= 1:
        return np.nan

    x = d - d.mean()
    gamma0 = np.dot(x, x) / n
    lrv = gamma0

    max_lag = min(lag, n - 1)
    for k in range(1, max_lag + 1):
        gamma_k = np.dot(x[k:], x[:-k]) / n
        weight = 1.0 - k / (max_lag + 1.0)
        lrv += 2.0 * weight * gamma_k

    return lrv


def dm_test(loss_diff: np.ndarray, lag: int = 24):
    d = np.asarray(loss_diff, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 5:
        return np.nan, np.nan, np.nan

    mean_d = float(np.mean(d))
    lrv = newey_west_variance(d, lag=lag)

    if not np.isfinite(lrv) or lrv <= 1e-12:
        return mean_d, np.nan, np.nan

    dm_stat = mean_d / math.sqrt(lrv / n)
    p = normal_two_sided_p(dm_stat)
    return mean_d, float(dm_stat), float(p)


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_true - y_pred
    ae = np.abs(err)
    se = err ** 2
    out = {
        "MAE": float(np.mean(ae)),
        "RMSE": float(np.sqrt(np.mean(se))),
        "P90_AE": float(np.quantile(ae, 0.90)),
        "P95_AE": float(np.quantile(ae, 0.95)),
    }
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    out["R2"] = float(1.0 - np.sum(se) / denom) if denom > 1e-12 else np.nan
    return out


def align_pair(df_baseline: pd.DataFrame, df_ours: pd.DataFrame) -> pd.DataFrame:
    use_time = (
        df_baseline["target_time"].notna().sum() > 0
        and df_ours["target_time"].notna().sum() > 0
    )

    if use_time:
        keys = ["target_time", "horizon"]
    else:
        keys = ["sample_id", "horizon"]

    keep_base = [
        "sample_id", "target_time", "horizon", "y_true", "y_pred",
        "weather_true", "dhi_true", "ramp_true", "snowfall", "precipitation",
        "high_dhi_mask", "high_ramp_mask"
    ]
    keep_base = [c for c in keep_base if c in df_baseline.columns]

    base = df_baseline[keep_base].copy().rename(columns={"y_pred": "pred_baseline"})
    ours = df_ours[[c for c in ["sample_id", "target_time", "horizon", "y_pred"] if c in df_ours.columns]].copy()
    ours = ours.rename(columns={"y_pred": "pred_ours"})

    merged = base.merge(ours, on=keys, how="inner")

    if "sample_id" not in merged.columns:
        if "sample_id_x" in merged.columns:
            merged["sample_id"] = merged["sample_id_x"]
        elif "sample_id_y" in merged.columns:
            merged["sample_id"] = merged["sample_id_y"]
        else:
            merged["sample_id"] = np.arange(len(merged))

    merged = merged.dropna(subset=["y_true", "pred_baseline", "pred_ours"]).reset_index(drop=True)
    return merged


def run_pair_tests(all_df, dataset, ours_model, baseline_model, min_samples, dm_lag):
    df_dataset = all_df[all_df["dataset"] == dataset].copy()

    seeds_ours = set(df_dataset[df_dataset["model"] == ours_model]["seed"].dropna().astype(int).unique())
    seeds_base = set(df_dataset[df_dataset["model"] == baseline_model]["seed"].dropna().astype(int).unique())
    common_seeds = sorted(seeds_ours & seeds_base)

    if not common_seeds:
        print(f"[WARN] No common seeds: dataset={dataset}, ours={ours_model}, baseline={baseline_model}")
        return pd.DataFrame(), pd.DataFrame()

    dm_rows = []
    metric_rows = []

    for seed in common_seeds:
        df_o = df_dataset[(df_dataset["model"] == ours_model) & (df_dataset["seed"] == seed)].copy()
        df_b = df_dataset[(df_dataset["model"] == baseline_model) & (df_dataset["seed"] == seed)].copy()

        aligned = align_pair(df_b, df_o)
        if aligned.empty:
            print(f"[WARN] Empty alignment: dataset={dataset}, baseline={baseline_model}, seed={seed}")
            continue

        masks = get_scenario_masks(aligned, min_samples=min_samples)

        for scenario, mask in masks.items():
            sub = aligned[mask.values].copy()
            n = len(sub)
            if n < min_samples:
                continue

            y = sub["y_true"].values.astype(float)
            pred_b = sub["pred_baseline"].values.astype(float)
            pred_o = sub["pred_ours"].values.astype(float)

            ae_b = np.abs(y - pred_b)
            ae_o = np.abs(y - pred_o)
            se_b = (y - pred_b) ** 2
            se_o = (y - pred_o) ** 2

            for loss_type, diff in [("AE", ae_b - ae_o), ("SE", se_b - se_o)]:
                mean_d, stat, p = dm_test(diff, lag=dm_lag)
                dm_rows.append({
                    "dataset": dataset,
                    "ours_model": ours_model,
                    "baseline_model": baseline_model,
                    "seed": seed,
                    "scenario": scenario,
                    "n": n,
                    "loss_type": loss_type,
                    "mean_loss_diff_baseline_minus_ours": mean_d,
                    "dm_stat": stat,
                    "p_value_two_sided": p,
                    "significant_0.05": bool(np.isfinite(p) and p < 0.05),
                    "ours_better_mean_loss": bool(np.isfinite(mean_d) and mean_d > 0),
                })

            mb = metric_values(y, pred_b)
            mo = metric_values(y, pred_o)

            for metric in ["MAE", "RMSE", "P90_AE", "P95_AE", "R2"]:
                base_val = mb[metric]
                ours_val = mo[metric]

                if metric == "R2":
                    diff = ours_val - base_val
                    improvement_percent = np.nan
                else:
                    diff = base_val - ours_val
                    improvement_percent = diff / base_val * 100 if abs(base_val) > 1e-12 else np.nan

                metric_rows.append({
                    "dataset": dataset,
                    "ours_model": ours_model,
                    "baseline_model": baseline_model,
                    "seed": seed,
                    "scenario": scenario,
                    "n": n,
                    "metric": metric,
                    "baseline_value": base_val,
                    "ours_value": ours_val,
                    "diff_positive_means_ours_better": diff,
                    "improvement_percent": improvement_percent,
                })

    return pd.DataFrame(dm_rows), pd.DataFrame(metric_rows)


def seed_level_tests(metric_df: pd.DataFrame) -> pd.DataFrame:
    if metric_df.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["dataset", "ours_model", "baseline_model", "scenario", "metric"]

    for keys, g in metric_df.groupby(group_cols):
        diffs = g["diff_positive_means_ours_better"].dropna().values.astype(float)
        imps = g["improvement_percent"].dropna().values.astype(float)

        seed_count = len(diffs)
        if seed_count == 0:
            continue

        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1)) if seed_count > 1 else 0.0
        mean_imp = float(np.mean(imps)) if len(imps) > 0 else np.nan
        std_imp = float(np.std(imps, ddof=1)) if len(imps) > 1 else 0.0

        t_stat = np.nan
        t_p = np.nan
        w_stat = np.nan
        w_p = np.nan

        if SCIPY_AVAILABLE and seed_count >= 2:
            try:
                t_res = stats.ttest_1samp(diffs, popmean=0.0, alternative="greater")
                t_stat = float(t_res.statistic)
                t_p = float(t_res.pvalue)
            except Exception:
                pass

            try:
                if np.any(np.abs(diffs) > 1e-12):
                    w_res = stats.wilcoxon(diffs, alternative="greater", zero_method="wilcox")
                    w_stat = float(w_res.statistic)
                    w_p = float(w_res.pvalue)
            except Exception:
                pass

        rows.append({
            "dataset": keys[0],
            "ours_model": keys[1],
            "baseline_model": keys[2],
            "scenario": keys[3],
            "metric": keys[4],
            "seed_count": seed_count,
            "mean_diff_positive_means_ours_better": mean_diff,
            "std_diff": std_diff,
            "mean_improvement_percent": mean_imp,
            "std_improvement_percent": std_imp,
            "paired_t_stat": t_stat,
            "paired_t_p_one_sided": t_p,
            "wilcoxon_stat": w_stat,
            "wilcoxon_p_one_sided": w_p,
            "all_seeds_ours_better": bool(np.all(diffs > 0)),
            "better_seed_count": int(np.sum(diffs > 0)),
        })

    return pd.DataFrame(rows)


def summarize_dm(dm_df: pd.DataFrame, seed_tests_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not dm_df.empty:
        group_cols = ["dataset", "ours_model", "baseline_model", "scenario", "loss_type"]
        for keys, g in dm_df.groupby(group_cols):
            rows.append({
                "dataset": keys[0],
                "ours_model": keys[1],
                "baseline_model": keys[2],
                "scenario": keys[3],
                "test_target": f"DM_{keys[4]}",
                "seed_count": int(len(g)),
                "better_seed_count": int(g["ours_better_mean_loss"].sum()),
                "passed_seed_count_p005": int((g["significant_0.05"] & g["ours_better_mean_loss"]).sum()),
                "mean_dm_stat": float(g["dm_stat"].mean()),
                "mean_p_value": float(g["p_value_two_sided"].mean()),
                "mean_loss_diff": float(g["mean_loss_diff_baseline_minus_ours"].mean()),
                "mean_improvement_percent": np.nan,
                "paired_t_p_one_sided": np.nan,
                "wilcoxon_p_one_sided": np.nan,
            })

    if seed_tests_df is not None and not seed_tests_df.empty:
        for _, r in seed_tests_df.iterrows():
            rows.append({
                "dataset": r["dataset"],
                "ours_model": r["ours_model"],
                "baseline_model": r["baseline_model"],
                "scenario": r["scenario"],
                "test_target": f"SeedPaired_{r['metric']}",
                "seed_count": r["seed_count"],
                "better_seed_count": r["better_seed_count"],
                "passed_seed_count_p005": np.nan,
                "mean_dm_stat": np.nan,
                "mean_p_value": np.nan,
                "mean_loss_diff": r["mean_diff_positive_means_ours_better"],
                "mean_improvement_percent": r["mean_improvement_percent"],
                "paired_t_p_one_sided": r["paired_t_p_one_sided"],
                "wilcoxon_p_one_sided": r["wilcoxon_p_one_sided"],
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--ours", type=str, default="OURS")
    parser.add_argument("--baselines", type=str, nargs="*", default=None)
    parser.add_argument("--datasets", type=str, nargs="*", default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--dm-lag", type=int, default=24)
    parser.add_argument("--no-debug", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out).resolve() if args.out else root / "significance"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_df = load_all_predictions(root, debug=(not args.no_debug))

    if args.datasets:
        print(f"[INFO] User dataset filter: {args.datasets}")
        print(f"[INFO] Datasets before filtering: {sorted(all_df['dataset'].unique().tolist())}")
        all_df = all_df[all_df["dataset"].isin(args.datasets)].copy()
        print(f"[INFO] Rows after dataset filtering: {len(all_df)}")

    datasets = sorted(all_df["dataset"].dropna().unique().tolist())
    if not datasets:
        raise RuntimeError(
            "No datasets found after filtering. "
            "Do not use --datasets first, or use exactly one of the dataset names printed above."
        )

    all_dm = []
    all_metric = []

    for dataset in datasets:
        models = sorted(all_df[all_df["dataset"] == dataset]["model"].dropna().unique().tolist())
        print(f"\n[INFO] Dataset={dataset}; available models={models}")

        if args.ours not in models:
            print(f"[WARN] Ours model '{args.ours}' not found in dataset={dataset}.")
            continue

        if args.baselines:
            baselines = [b for b in args.baselines if b in models]
            missing = [b for b in args.baselines if b not in models]
            if missing:
                print(f"[WARN] Missing baselines in dataset={dataset}: {missing}")
        else:
            baselines = [m for m in models if m != args.ours]

        for baseline in baselines:
            print(f"[INFO] Testing: {args.ours} vs {baseline}")
            dm_df, metric_df = run_pair_tests(
                all_df=all_df,
                dataset=dataset,
                ours_model=args.ours,
                baseline_model=baseline,
                min_samples=args.min_samples,
                dm_lag=args.dm_lag,
            )
            if not dm_df.empty:
                all_dm.append(dm_df)
            if not metric_df.empty:
                all_metric.append(metric_df)

    dm_all = pd.concat(all_dm, ignore_index=True) if all_dm else pd.DataFrame()
    metric_all = pd.concat(all_metric, ignore_index=True) if all_metric else pd.DataFrame()
    seed_tests = seed_level_tests(metric_all) if not metric_all.empty else pd.DataFrame()
    summary = summarize_dm(dm_all, seed_tests)

    dm_path = out_dir / "dm_test_by_seed.csv"
    metric_path = out_dir / "scenario_metrics_by_seed.csv"
    seed_path = out_dir / "seed_level_metric_tests.csv"
    summary_path = out_dir / "significance_summary.csv"

    dm_all.to_csv(dm_path, index=False, encoding="utf-8-sig")
    metric_all.to_csv(metric_path, index=False, encoding="utf-8-sig")
    seed_tests.to_csv(seed_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n[完成] 显著性检验结果已保存：")
    print(f"  1) {dm_path} rows={len(dm_all)}")
    print(f"  2) {metric_path} rows={len(metric_all)}")
    print(f"  3) {seed_path} rows={len(seed_tests)}")
    print(f"  4) {summary_path} rows={len(summary)}")


if __name__ == "__main__":
    main()
