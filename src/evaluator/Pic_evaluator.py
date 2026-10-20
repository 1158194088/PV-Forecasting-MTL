import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score

# ==========================================
# ★ Visualization of prediction curves under specific weather conditions
# ==========================================
def plot_weather_specific_curves(all_p_true, all_p_pred, daily_weather_labels, dataset, file_name ,target_weathers=['Cloudy', 'Rain', 'Snow']):
    fig, axes = plt.subplots(len(target_weathers), 1, figsize=(14, 4 * len(target_weathers)), dpi=150)
    if len(target_weathers) == 1: axes = [axes]
    print("\n[PLOT MODULE START] Generating and saving weather‑specific fitting curves...")
    weather_name_mapping_vis = {'Cloudy': 'Cloudy', 'Sunny': 'Sunny', 'Rain': 'Rain', 'Snow': 'Snow'}
    for ax, w_name_en in zip(axes, target_weathers):
        w_idx = -1
        for k, v in dataset.weather_names_mapping.items():
            if w_name_en and w_name_en.strip() in v.strip():
                w_idx = k
                break

        if w_idx == -1: continue
        indices = np.where(daily_weather_labels == w_idx)[0]
        if len(indices) == 0: continue
        plot_len = min(72, len(indices))
        plot_indices = indices[:plot_len]
        y_true = all_p_true[plot_indices]
        y_pred = all_p_pred[plot_indices]
        ax.plot(y_true, label='Ground Truth (Power)', color='#2c3e50', linewidth=2.5, marker='o', markersize=5)
        ax.plot(y_pred, label='Heterogeneous Quad‑Task Pred', color='#e74c3c', linewidth=2.5, linestyle='--', marker='x', markersize=5)

        w_name_en_title = weather_name_mapping_vis.get(w_name_en.strip(), w_name_en)
        ax.set_title(f'Heterogeneous Experts Fitting: {w_name_en_title} (First {plot_len} Hours)', fontsize=15, fontweight='bold', pad=10)
        ax.set_ylabel('PV Power (MW)', fontsize=12)
        ax.set_xlabel('Time Step (Hours)', fontsize=12)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.6)

        ax.fill_between(range(plot_len), y_true, y_pred, color='red', alpha=0.1)
    plt.tight_layout()
    save_path = '/home/cyx/FusionSF/MTL/PV_QuadTask_Project/result/' + file_name + '.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"  - Curve comparison figure generated and saved to: {save_path}")
    plt.close(fig)

# ==========================================
# 3. Comprehensive multi‑task evaluation system
# ==========================================
def evaluate_model_comprehensive(model, test_loader, dataset, file_name, device='cpu'):
    model.eval()
    model.to(device)
    all_p_pred, all_p_true, all_dfi_pred, all_dfi_true, all_w_pred, all_w_true, all_r_pred, all_r_true, daily_weather_labels = [], [], [], [], [], [], [], [], []

    print("\n[EVALUATION SYSTEM START] Running heterogeneous Quad‑Task multi‑dimensional inference...")
    with torch.no_grad():
        for batch in test_loader:
            # ★ Updated to receive 5 return values
            p_out, dfi_out, w_logits, ramp_logits, *_ = model(batch['ts_input'].to(device), batch['nwp_input'].to(device))

            all_p_pred.extend(np.maximum(dataset.scaler_p.inverse_transform(p_out.cpu().numpy().reshape(-1, 1)).flatten(), 0))
            all_p_true.extend(dataset.scaler_p.inverse_transform(batch['target_p'].cpu().numpy().reshape(-1, 1)).flatten())
            all_dfi_pred.extend(np.maximum(dataset.scaler_dfi.inverse_transform(dfi_out.cpu().numpy().reshape(-1, 1)).flatten(), 0))
            all_dfi_true.extend(dataset.scaler_dfi.inverse_transform(batch['target_dfi'].cpu().numpy().reshape(-1, 1)).flatten())
            all_w_pred.extend(torch.argmax(w_logits, dim=-1).cpu().numpy())
            all_w_true.extend(batch['daily_weather'].cpu().numpy())

            all_r_pred.extend(torch.argmax(ramp_logits, dim=-1).cpu().numpy().flatten())
            all_r_true.extend(batch['target_ramp'].cpu().numpy().flatten())

            daily_weather_labels.extend(np.repeat(batch['daily_weather'].numpy(), 24))
    all_p_pred, all_p_true = np.array(all_p_pred), np.array(all_p_true)
    max_p = np.max(all_p_true)
    daily_weather_labels = np.array(daily_weather_labels)

    print("\n" + "="*60 + "\n >>> Report I: Physical and macro constraint extraction performance (Auxiliary task metrics)\n" + "="*60)
    print(f"[Micro Expert A] Diffuse Irradiance (DFI) MAE: {mean_absolute_error(all_dfi_true, all_dfi_pred):.4f} W/m²")
    print(f"[Micro Expert B] 24‑hour ramp‑rate classification Acc: {accuracy_score(all_r_true, all_r_pred)*100:.2f}% | F1 Macro: {f1_score(all_r_true, all_r_pred, average='macro'):.4f}")
    print(f"[Macro Expert C] Daily weather classification Acc: {accuracy_score(all_w_true, all_w_pred)*100:.2f}% | F1 Macro: {f1_score(all_w_true, all_w_pred, average='macro'):.4f}")

    print("\n" + "="*60 + "\n >>> Report II: Final power forecasting performance (Main task metrics)\n" + "="*60)

    # Compute MSE and RMSE
    r2 = r2_score(all_p_true, all_p_pred)
    mae = mean_absolute_error(all_p_true, all_p_pred)
    mse = mean_squared_error(all_p_true, all_p_pred)
    rmse = np.sqrt(mse)

    # Compute normalized metrics (NMAE, NRMSE, NHMSE)
    nmae = (mae / max_p * 100) if max_p > 0 else 0
    nrmse = (rmse / max_p * 100) if max_p > 0 else 0
    nhmse = (mse / (max_p ** 2) * 100) if max_p > 0 else 0

    print(f"R‑squared: {r2:.4f}")
    print(f"MAE: {mae:.4f} MW | Normalized NMAE: {nmae:.2f}%")
    print(f"RMSE: {rmse:.4f} MW | Normalized NRMSE: {nrmse:.2f}%")
    print(f"MSE: {mse:.4f} MW² | Normalized NHMSE (NMSE): {nhmse:.2f}%")

    print("\n" + "="*60 + "\n >>> Report III: Robustness evaluation under specific weather conditions (grouped by daily ground‑truth labels)\n" + "="*60)
    for w_idx, w_name in dataset.weather_names_mapping.items():
        indices = np.where(daily_weather_labels == w_idx)[0]
        if len(indices) == 0: continue

        sub_mae = mean_absolute_error(all_p_true[indices], all_p_pred[indices])
        sub_mse = mean_squared_error(all_p_true[indices], all_p_pred[indices])
        sub_rmse = np.sqrt(sub_mse)

        sub_nmae = (sub_mae / max_p * 100) if max_p > 0 else 0
        sub_nrmse = (sub_rmse / max_p * 100) if max_p > 0 else 0

        sub_r2 = r2_score(all_p_true[indices], all_p_pred[indices]) if len(indices) > 1 else 0

        print(f"[{w_name.ljust(15)}] Total {len(indices):>5} hourly samples | R²: {sub_r2:7.4f}")
        print(f"                     | NMAE: {sub_nmae:5.2f}% | MAE: {sub_mae:6.4f} MW")
        print(f"                     | NRMSE: {sub_nrmse:5.2f}% | RMSE: {sub_rmse:6.4f} MW")

    # Save prediction results to CSV file
    weather_str_labels = [dataset.weather_names_mapping[w] for w in daily_weather_labels]
    df_save = pd.DataFrame({
        'Ground_Truth': all_p_true,
        'Prediction': all_p_pred,
        'Weather': weather_str_labels
    })

    save_filename = '/home/cyx/FusionSF/MTL/PV_QuadTask_Project/result/' + file_name + '.csv'
    df_save.to_csv(save_filename, index=False)
    print(f"\n[SAVE RESULTS] Predictions extracted and saved to: {save_filename}")
    plot_weather_specific_curves(all_p_true, all_p_pred, daily_weather_labels, dataset, file_name, target_weathers=['Cloudy', 'Rain', 'Snow'])
