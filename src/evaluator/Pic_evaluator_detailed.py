import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    confusion_matrix
)


# ==========================================
# 工具函数
# ==========================================
def _safe_inverse_transform(scaler, arr_2d):
    """arr_2d shape: [B, pred_len] or [N]. Return same flattened physical scale."""
    arr = np.asarray(arr_2d).reshape(-1, 1)
    return scaler.inverse_transform(arr).flatten()


def _safe_r2(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    try:
        return r2_score(y_true, y_pred)
    except Exception:
        return np.nan


def _classification_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def _unwrap_subset(ds):
    """
    返回 base_dataset 与当前 DataLoader 对应的 base_dataset 索引顺序。
    支持 Subset 嵌套。
    """
    indices = None
    while isinstance(ds, Subset):
        current_indices = np.array(list(ds.indices), dtype=int)
        if indices is None:
            indices = current_indices
        else:
            indices = indices[current_indices]
        ds = ds.dataset

    if indices is None:
        indices = np.arange(len(ds), dtype=int)

    return ds, indices


def _get_time_meta(base_dataset, dataset_idx, horizon_idx):
    """
    dataset_idx: base_dataset 中的窗口索引
    horizon_idx: 0-based horizon
    """
    x_begin = int(base_dataset.valid_x_begins[int(dataset_idx)])
    pred_begin = x_begin + int(base_dataset.seq_len)

    init_time = base_dataset.time_dt.iloc[pred_begin]
    target_time = base_dataset.time_dt.iloc[pred_begin + horizon_idx]

    return x_begin, pred_begin, init_time, target_time


def _weather_name(dataset, idx):
    try:
        return dataset.weather_names_mapping.get(int(idx), str(int(idx)))
    except Exception:
        return str(idx)


def _compute_basic_power_metrics(y_true, y_pred, max_p=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    r2 = _safe_r2(y_true, y_pred)

    if max_p is None:
        max_p = np.max(y_true) if len(y_true) > 0 else np.nan

    nmae = mae / max_p * 100 if max_p and max_p > 0 else np.nan
    nrmse = rmse / max_p * 100 if max_p and max_p > 0 else np.nan
    nmse = mse / (max_p ** 2) * 100 if max_p and max_p > 0 else np.nan

    return {
        "R2": r2,
        "MAE_MW": mae,
        "RMSE_MW": rmse,
        "MSE_MW2": mse,
        "NMAE_percent": nmae,
        "NRMSE_percent": nrmse,
        "NMSE_percent": nmse
    }


# ==========================================
# 可视化：按天气绘制曲线
# ==========================================
def plot_weather_specific_curves(
    all_p_true,
    all_p_pred,
    daily_weather_labels,
    dataset,
    file_name,
    output_dir,
    target_weathers=("多云", "降雨", "降雪")
):
    fig, axes = plt.subplots(len(target_weathers), 1, figsize=(14, 4 * len(target_weathers)), dpi=150)
    if len(target_weathers) == 1:
        axes = [axes]

    print("\n[绘图模块] 正在生成极端天气拟合曲线图...")

    weather_name_mapping_vis = {"多云": "Cloudy", "晴天": "Sunny", "降雨": "Rain", "降雪": "Snow"}

    for ax, w_name_cn in zip(axes, target_weathers):
        w_idx = -1
        for k, v in dataset.weather_names_mapping.items():
            if w_name_cn and w_name_cn.strip() in str(v).strip():
                w_idx = k
                break

        if w_idx == -1:
            ax.set_title(f"No weather label found for {w_name_cn}")
            continue

        indices = np.where(daily_weather_labels == w_idx)[0]
        if len(indices) == 0:
            ax.set_title(f"No samples for {w_name_cn}")
            continue

        plot_len = min(72, len(indices))
        plot_indices = indices[:plot_len]

        y_true = all_p_true[plot_indices]
        y_pred = all_p_pred[plot_indices]

        ax.plot(y_true, label="Ground Truth (Power)", linewidth=2.2, marker="o", markersize=4)
        ax.plot(y_pred, label="Prediction", linewidth=2.2, linestyle="--", marker="x", markersize=4)

        w_name_en = weather_name_mapping_vis.get(w_name_cn.strip(), w_name_cn)
        ax.set_title(f"Prediction Curve under {w_name_en} Weather", fontsize=14, fontweight="bold")
        ax.set_ylabel("PV Power (MW)")
        ax.set_xlabel("Time Step (Hours)")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.fill_between(range(plot_len), y_true, y_pred, alpha=0.12)

    plt.tight_layout()
    out_dir = Path(output_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"{file_name}_weather_curves.png"
    plt.savefig(save_path, bbox_inches="tight")
    print(f"  - 曲线图已保存: {save_path}")
    plt.close(fig)


# ==========================================
# 生成压力测试指标
# ==========================================
def _compute_stress_metrics(df_pred, max_p):
    rows = []

    def add_metric(scenario, samples, metric, value, threshold=None):
        rows.append({
            "scenario": scenario,
            "samples": int(samples) if pd.notna(samples) else samples,
            "metric": metric,
            "value": value,
            "threshold": threshold
        })

    # Global
    global_metrics = _compute_basic_power_metrics(df_pred["y_true_MW"], df_pred["y_pred_MW"], max_p=max_p)
    for k, v in global_metrics.items():
        add_metric("global", len(df_pred), k, v)

    # P90 / P95 AE：真正的误差分布分位数
    add_metric("global", len(df_pred), "P90_AE_MW", np.quantile(df_pred["abs_error_MW"], 0.90))
    add_metric("global", len(df_pred), "P95_AE_MW", np.quantile(df_pred["abs_error_MW"], 0.95))

    # Daily cumulative generation error
    temp = df_pred.copy()
    temp["date"] = pd.to_datetime(temp["target_time"]).dt.date
    daily = temp.groupby("date").agg(
        true_MWh=("y_true_MW", "sum"),
        pred_MWh=("y_pred_MW", "sum")
    ).reset_index()
    daily["abs_error_MWh"] = np.abs(daily["true_MWh"] - daily["pred_MWh"])
    valid = daily["true_MWh"] > 1e-6
    daily_mape = np.mean(daily.loc[valid, "abs_error_MWh"] / daily.loc[valid, "true_MWh"]) * 100 if valid.any() else np.nan
    add_metric("daily_cumulative_generation", len(daily), "MAE_MWh_per_day", daily["abs_error_MWh"].mean())
    add_metric("daily_cumulative_generation", len(daily), "MAPE_percent", daily_mape)

    # High DHI: top 10% DHI
    if "dhi_true" in df_pred.columns and df_pred["dhi_true"].notna().sum() > 0:
        dhi_thr = df_pred["dhi_true"].quantile(0.90)
        mask = df_pred["dhi_true"] >= dhi_thr
        if mask.sum() > 0:
            add_metric("high_DHI_top10", mask.sum(), "MAE_MW", mean_absolute_error(df_pred.loc[mask, "y_true_MW"], df_pred.loc[mask, "y_pred_MW"]), threshold=dhi_thr)
            add_metric("high_DHI_top10", mask.sum(), "RMSE_MW", np.sqrt(mean_squared_error(df_pred.loc[mask, "y_true_MW"], df_pred.loc[mask, "y_pred_MW"])), threshold=dhi_thr)

    # High Ramp: actual |ΔP| top 5%
    df_sorted = df_pred.sort_values("target_time").copy()
    # 若存在重复 target_time，只保留第一次，避免重叠窗口导致 ramp 计算重复
    df_unique = df_sorted.drop_duplicates(subset=["target_time"]).copy()
    df_unique["abs_ramp_MW_per_h"] = df_unique["y_true_MW"].diff().abs()
    ramp_valid = df_unique["abs_ramp_MW_per_h"].dropna()
    if len(ramp_valid) > 0:
        ramp_thr = ramp_valid.quantile(0.95)
        high_ramp_times = set(df_unique.loc[df_unique["abs_ramp_MW_per_h"] >= ramp_thr, "target_time"].astype(str))
        mask = df_pred["target_time"].astype(str).isin(high_ramp_times)
        if mask.sum() > 0:
            add_metric("high_Ramp_top5", mask.sum(), "MAE_MW", mean_absolute_error(df_pred.loc[mask, "y_true_MW"], df_pred.loc[mask, "y_pred_MW"]), threshold=ramp_thr)
            add_metric("high_Ramp_top5", mask.sum(), "RMSE_MW", np.sqrt(mean_squared_error(df_pred.loc[mask, "y_true_MW"], df_pred.loc[mask, "y_pred_MW"])), threshold=ramp_thr)

    # Top 5% true power scenario：注意这不是 P95_AE
    power_thr = df_pred["y_true_MW"].quantile(0.95)
    mask = df_pred["y_true_MW"] >= power_thr
    if mask.sum() > 0:
        add_metric("top5_true_power", mask.sum(), "MAE_MW", mean_absolute_error(df_pred.loc[mask, "y_true_MW"], df_pred.loc[mask, "y_pred_MW"]), threshold=power_thr)
        add_metric("top5_true_power", mask.sum(), "RMSE_MW", np.sqrt(mean_squared_error(df_pred.loc[mask, "y_true_MW"], df_pred.loc[mask, "y_pred_MW"])), threshold=power_thr)

    # Snow + High Ramp combined, if available
    snow_mask = df_pred["weather_true_name"].astype(str).str.contains("雪|snow|Snow", regex=True, na=False)
    if "high_ramp_times" in locals():
        high_ramp_mask = df_pred["target_time"].astype(str).isin(high_ramp_times)
        combined = snow_mask & high_ramp_mask
        if combined.sum() > 0:
            add_metric("snow_high_Ramp", combined.sum(), "MAE_MW", mean_absolute_error(df_pred.loc[combined, "y_true_MW"], df_pred.loc[combined, "y_pred_MW"]))
            add_metric("snow_high_Ramp", combined.sum(), "RMSE_MW", np.sqrt(mean_squared_error(df_pred.loc[combined, "y_true_MW"], df_pred.loc[combined, "y_pred_MW"])))
        else:
            add_metric("snow_high_Ramp", 0, "MAE_MW", np.nan)

    return pd.DataFrame(rows), daily


# ==========================================
# 主评估函数：保存逐样本预测 + 辅助任务 + 压力测试
# ==========================================
def evaluate_model_comprehensive(
    model,
    test_loader,
    dataset,
    file_name,
    device="cpu",
    output_dir=None,
    dataset_name=None,
    model_name=None,
    seed=None,
    make_plots=False,
    compute_aux=True
):
    """
    输出文件：
    1) predictions/{file_name}_detailed_predictions.csv
       用于 DM 检验、P95/P90、High-DHI、High-Ramp、case study。
    2) metrics/{file_name}_power_metrics.csv
       全局功率指标。
    3) metrics/{file_name}_weather_metrics.csv
       分天气功率指标。
    4) metrics/{file_name}_auxiliary_metrics.csv
       DHI / Weather / Ramp 辅助任务指标。
    5) metrics/{file_name}_stress_metrics.csv
       复杂情景与边界压力指标。
    """

    if output_dir is None:
        output_dir = "/home/cyx/FusionSF/MTL/PV_QuadTask_Project/result"

    output_dir = Path(output_dir)
    pred_dir = output_dir / "predictions"
    metrics_dir = output_dir / "metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.to(device)

    base_dataset, loader_indices = _unwrap_subset(test_loader.dataset)

    rows = []
    window_aux_rows = []

    all_p_pred_flat = []
    all_p_true_flat = []
    all_dhi_pred_flat = []
    all_dhi_true_flat = []
    all_w_pred = []
    all_w_true = []
    all_r_pred_flat = []
    all_r_true_flat = []
    daily_weather_labels_flat = []

    sample_cursor = 0

    print("\n[评估系统启动] 正在生成逐样本预测文件与统计指标...")

    with torch.no_grad():
        for batch in test_loader:
            ts = batch["ts_input"].to(device)
            nwp = batch["nwp_input"].to(device)

            outputs = model(ts, nwp)
            if isinstance(outputs, (tuple, list)):
                p_out = outputs[0]
                dhi_out = outputs[1] if len(outputs) > 1 else torch.zeros_like(p_out)
                w_logits = outputs[2] if len(outputs) > 2 else torch.zeros((p_out.shape[0], getattr(base_dataset, "num_classes", 1)), device=device)
                ramp_logits = outputs[3] if len(outputs) > 3 else torch.zeros((p_out.shape[0], p_out.shape[1], 1), device=device)

                # 某些模型返回 p_sh_dict, w_w, w_ramp
                w_w = outputs[5] if len(outputs) > 5 and torch.is_tensor(outputs[5]) else None
                w_ramp_gate = outputs[6] if len(outputs) > 6 and torch.is_tensor(outputs[6]) else None
            else:
                p_out = outputs
                dhi_out = torch.zeros_like(p_out)
                w_logits = torch.zeros((p_out.shape[0], getattr(base_dataset, "num_classes", 1)), device=device)
                ramp_logits = torch.zeros((p_out.shape[0], p_out.shape[1], 1), device=device)
                w_w = None
                w_ramp_gate = None

            B = p_out.shape[0]
            pred_len = p_out.shape[1]

            p_pred = np.maximum(_safe_inverse_transform(base_dataset.scaler_p, p_out.cpu().numpy()), 0)
            p_true = _safe_inverse_transform(base_dataset.scaler_p, batch["target_p"].cpu().numpy())

            dhi_pred = np.maximum(_safe_inverse_transform(base_dataset.scaler_dfi, dhi_out.cpu().numpy()), 0)
            dhi_true = _safe_inverse_transform(base_dataset.scaler_dfi, batch["target_dfi"].cpu().numpy())

            p_pred_2d = p_pred.reshape(B, pred_len)
            p_true_2d = p_true.reshape(B, pred_len)
            dhi_pred_2d = dhi_pred.reshape(B, pred_len)
            dhi_true_2d = dhi_true.reshape(B, pred_len)

            weather_prob = torch.softmax(w_logits, dim=-1).cpu().numpy()
            weather_pred = np.argmax(weather_prob, axis=-1)
            weather_true = batch["daily_weather"].cpu().numpy()

            ramp_prob = torch.softmax(ramp_logits, dim=-1).cpu().numpy()
            ramp_pred = np.argmax(ramp_prob, axis=-1)
            ramp_true = batch["target_ramp"].cpu().numpy()

            if w_w is not None:
                weather_shared = w_w[:, 0].detach().cpu().numpy()
                weather_private = w_w[:, 1].detach().cpu().numpy() if w_w.shape[1] > 1 else np.full(B, np.nan)
            else:
                weather_shared = np.full(B, np.nan)
                weather_private = np.full(B, np.nan)

            if w_ramp_gate is not None:
                ramp_shared = w_ramp_gate[:, 0].detach().cpu().numpy()
                ramp_private = w_ramp_gate[:, 1].detach().cpu().numpy() if w_ramp_gate.shape[1] > 1 else np.full(B, np.nan)
            else:
                ramp_shared = np.full(B, np.nan)
                ramp_private = np.full(B, np.nan)

            for b in range(B):
                dataset_idx = int(loader_indices[sample_cursor + b])
                _, pred_begin, init_time, _ = _get_time_meta(base_dataset, dataset_idx, 0)

                w_true_id = int(weather_true[b])
                w_pred_id = int(weather_pred[b])
                w_true_name = _weather_name(base_dataset, w_true_id)
                w_pred_name = _weather_name(base_dataset, w_pred_id)

                # window-level auxiliary classification outputs
                aux_row = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "seed": seed,
                    "sample_order": sample_cursor + b,
                    "dataset_window_index": dataset_idx,
                    "init_time": str(init_time),
                    "weather_true_id": w_true_id,
                    "weather_true_name": w_true_name,
                    "weather_pred_id": w_pred_id,
                    "weather_pred_name": w_pred_name,
                    "weather_shared_weight": weather_shared[b],
                    "weather_private_weight": weather_private[b],
                    "ramp_shared_weight": ramp_shared[b],
                    "ramp_private_weight": ramp_private[b],
                }
                for c in range(weather_prob.shape[1]):
                    aux_row[f"weather_prob_{c}"] = weather_prob[b, c]
                window_aux_rows.append(aux_row)

                for h in range(pred_len):
                    _, _, init_time, target_time = _get_time_meta(base_dataset, dataset_idx, h)

                    y_t = float(p_true_2d[b, h])
                    y_p = float(p_pred_2d[b, h])
                    d_t = float(dhi_true_2d[b, h])
                    d_p = float(dhi_pred_2d[b, h])

                    r_true_id = int(ramp_true[b, h])
                    r_pred_id = int(ramp_pred[b, h])

                    row = {
                        "dataset": dataset_name,
                        "model": model_name,
                        "seed": seed,
                        "sample_order": sample_cursor + b,
                        "dataset_window_index": dataset_idx,
                        "target_time": str(target_time),
                        "init_time": str(init_time),
                        "horizon": h + 1,
                        "y_true_MW": y_t,
                        "y_pred_MW": y_p,
                        "abs_error_MW": abs(y_t - y_p),
                        "sq_error_MW2": (y_t - y_p) ** 2,
                        "dhi_true": d_t,
                        "dhi_pred": d_p,
                        "dhi_abs_error": abs(d_t - d_p),
                        "weather_true_id": w_true_id,
                        "weather_true_name": w_true_name,
                        "weather_pred_id": w_pred_id,
                        "weather_pred_name": w_pred_name,
                        "ramp_true_id": r_true_id,
                        "ramp_pred_id": r_pred_id,
                        "weather_shared_weight": weather_shared[b],
                        "weather_private_weight": weather_private[b],
                        "ramp_shared_weight": ramp_shared[b],
                        "ramp_private_weight": ramp_private[b],
                    }

                    for c in range(weather_prob.shape[1]):
                        row[f"weather_prob_{c}"] = weather_prob[b, c]
                    for c in range(ramp_prob.shape[-1]):
                        row[f"ramp_prob_{c}"] = ramp_prob[b, h, c]

                    rows.append(row)

            all_p_pred_flat.extend(p_pred)
            all_p_true_flat.extend(p_true)
            all_dhi_pred_flat.extend(dhi_pred)
            all_dhi_true_flat.extend(dhi_true)
            all_w_pred.extend(weather_pred)
            all_w_true.extend(weather_true)
            all_r_pred_flat.extend(ramp_pred.reshape(-1))
            all_r_true_flat.extend(ramp_true.reshape(-1))
            daily_weather_labels_flat.extend(np.repeat(weather_true, pred_len))

            sample_cursor += B

    df_pred = pd.DataFrame(rows)
    df_aux_window = pd.DataFrame(window_aux_rows)

    # 保存逐样本预测：这是 DM 检验、压力测试和 case study 的核心文件
    pred_path = pred_dir / f"{file_name}_detailed_predictions.csv"
    df_pred.to_csv(pred_path, index=False, encoding="utf-8-sig")
    aux_window_path = pred_dir / f"{file_name}_window_aux_outputs.csv"
    df_aux_window.to_csv(aux_window_path, index=False, encoding="utf-8-sig")

    all_p_pred = np.asarray(all_p_pred_flat)
    all_p_true = np.asarray(all_p_true_flat)
    all_dhi_pred = np.asarray(all_dhi_pred_flat)
    all_dhi_true = np.asarray(all_dhi_true_flat)
    all_w_pred = np.asarray(all_w_pred)
    all_w_true = np.asarray(all_w_true)
    all_r_pred = np.asarray(all_r_pred_flat)
    all_r_true = np.asarray(all_r_true_flat)
    daily_weather_labels = np.asarray(daily_weather_labels_flat)

    max_p = np.max(all_p_true)

    # =============================
    # 报告 I：辅助任务
    # =============================
    aux_metrics = []
    if compute_aux:
        dhi_mae = mean_absolute_error(all_dhi_true, all_dhi_pred)
        dhi_rmse = np.sqrt(mean_squared_error(all_dhi_true, all_dhi_pred))

        aux_metrics.extend([
            {"task": "DHI", "metric": "MAE", "value": dhi_mae},
            {"task": "DHI", "metric": "RMSE", "value": dhi_rmse},
            {"task": "Weather", "metric": "Accuracy", "value": accuracy_score(all_w_true, all_w_pred)},
            {"task": "Weather", "metric": "Macro_F1", "value": _classification_f1(all_w_true, all_w_pred)},
            {"task": "Ramp", "metric": "Accuracy", "value": accuracy_score(all_r_true, all_r_pred)},
            {"task": "Ramp", "metric": "Macro_F1", "value": _classification_f1(all_r_true, all_r_pred)},
        ])

        # Confusion matrices
        weather_cm = pd.DataFrame(confusion_matrix(all_w_true, all_w_pred))
        ramp_cm = pd.DataFrame(confusion_matrix(all_r_true, all_r_pred))
        weather_cm.to_csv(metrics_dir / f"{file_name}_weather_confusion_matrix.csv", index=False)
        ramp_cm.to_csv(metrics_dir / f"{file_name}_ramp_confusion_matrix.csv", index=False)

    df_aux_metrics = pd.DataFrame(aux_metrics)
    df_aux_metrics.to_csv(metrics_dir / f"{file_name}_auxiliary_metrics.csv", index=False, encoding="utf-8-sig")

    # =============================
    # 报告 II：全局功率预测
    # =============================
    power_metrics_dict = _compute_basic_power_metrics(all_p_true, all_p_pred, max_p=max_p)
    df_power_metrics = pd.DataFrame([
        {"metric": k, "value": v} for k, v in power_metrics_dict.items()
    ])
    df_power_metrics.to_csv(metrics_dir / f"{file_name}_power_metrics.csv", index=False, encoding="utf-8-sig")

    # =============================
    # 报告 III：按天气切分
    # =============================
    weather_rows = []
    for w_idx, w_name in base_dataset.weather_names_mapping.items():
        indices = np.where(daily_weather_labels == int(w_idx))[0]
        if len(indices) == 0:
            continue

        sub_metrics = _compute_basic_power_metrics(all_p_true[indices], all_p_pred[indices], max_p=max_p)
        for metric, value in sub_metrics.items():
            weather_rows.append({
                "weather_id": int(w_idx),
                "weather_name": w_name,
                "samples": int(len(indices)),
                "metric": metric,
                "value": value
            })

    df_weather_metrics = pd.DataFrame(weather_rows)
    df_weather_metrics.to_csv(metrics_dir / f"{file_name}_weather_metrics.csv", index=False, encoding="utf-8-sig")

    # =============================
    # 报告 IV：复杂情景与边界压力测试
    # =============================
    df_stress, df_daily = _compute_stress_metrics(df_pred, max_p=max_p)
    df_stress.to_csv(metrics_dir / f"{file_name}_stress_metrics.csv", index=False, encoding="utf-8-sig")
    df_daily.to_csv(metrics_dir / f"{file_name}_daily_generation_errors.csv", index=False, encoding="utf-8-sig")

    # 保存一个总 summary json
    summary = {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "prediction_file": str(pred_path),
        "window_aux_file": str(aux_window_path),
        "power_metrics": power_metrics_dict,
        "num_test_points": int(len(df_pred)),
        "num_test_windows": int(df_pred["sample_order"].nunique())
    }
    with open(metrics_dir / f"{file_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # =============================
    # 控制台简报
    # =============================
    print("\n" + "=" * 60 + "\n >>> 报告 I: 辅助任务成绩单\n" + "=" * 60)
    if compute_aux and not df_aux_metrics.empty:
        for _, r in df_aux_metrics.iterrows():
            print(f"[{r['task']}] {r['metric']}: {r['value']:.6f}")

    print("\n" + "=" * 60 + "\n >>> 报告 II: 最终发电预测能力\n" + "=" * 60)
    print(f"R²: {power_metrics_dict['R2']:.4f}")
    print(f"MAE: {power_metrics_dict['MAE_MW']:.4f} MW | NMAE: {power_metrics_dict['NMAE_percent']:.2f}%")
    print(f"RMSE: {power_metrics_dict['RMSE_MW']:.4f} MW | NRMSE: {power_metrics_dict['NRMSE_percent']:.2f}%")

    print("\n" + "=" * 60 + "\n >>> 数据落盘\n" + "=" * 60)
    print(f"逐样本预测 CSV: {pred_path}")
    print(f"窗口级辅助输出 CSV: {aux_window_path}")
    print(f"压力测试指标 CSV: {metrics_dir / f'{file_name}_stress_metrics.csv'}")

    if make_plots:
        plot_weather_specific_curves(
            all_p_true,
            all_p_pred,
            daily_weather_labels,
            base_dataset,
            file_name=file_name,
            output_dir=output_dir,
            target_weathers=("多云", "降雨", "降雪")
        )

    return {
        "predictions": df_pred,
        "power_metrics": df_power_metrics,
        "weather_metrics": df_weather_metrics,
        "auxiliary_metrics": df_aux_metrics,
        "stress_metrics": df_stress,
        "daily_generation": df_daily
    }
