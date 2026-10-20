"""
CA-FiLM structure ablation runner for QuadTask PV forecasting.

Ablation design (consistent with the provided table):

V0: None            | CGC original bottom, no cross-task interaction
V1: Simple Addition | F(main) + F(aux)
V2: Concatenation   | Concat(F(main), F(aux))
V3: CA-FiLM         | Shift-only, beta modulation
V4: CA-FiLM         | Scale-only, gamma modulation
OURS: CA-FiLM       | Scale(gamma) + Shift(beta)

Control rule:
- The backbone, CGC(Hetero), auxiliary heads, RouteReg, losses, data split,
  evaluator, scheduler, and Nash-MTL optimizer are kept unchanged.
- Only the cross-task feature interaction mechanism before the power head changes.

Run:
    python run_cafilm_structure_ablation_quadtask_5seeds.py

Before full training, it is recommended to test with:
    SEEDS = [0]
    MAX_EPOCHS = 3
"""

import os
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, Subset

from src.dataset.dataset_QuadTask import PVQuadTaskDataset
from src.evaluator.Pic_evaluator import evaluate_model_comprehensive
from src.optimizer.NashOptimizer import NashMTLOptimizer


# =============================================================================
# 1. Reproducibility and checkpoint utilities
# =============================================================================
def seed_everything(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed, workers=True)


class HardcoreCheckpoint(pl.Callback):
    """Save best checkpoint by validation MAE under the current run directory."""

    def __init__(self, dirpath: Path, save_last: bool = False):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.save_last = save_last
        self.best_mae = float("inf")
        self.best_model_path = ""

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "val_mae" not in metrics:
            return

        current_mae = metrics["val_mae"].item()
        if math.isnan(current_mae) or math.isinf(current_mae):
            print(f"\n[WARN] val_mae is abnormal ({current_mae}); skip checkpoint saving.")
            return

        if self.save_last:
            trainer.save_checkpoint(str(self.dirpath / "last.ckpt"))

        if current_mae < self.best_mae:
            self.best_mae = current_mae
            self.best_model_path = str(self.dirpath / "best.ckpt")
            trainer.save_checkpoint(self.best_model_path)
            print(f"\n[SAVE] best val_mae={current_mae:.6f}: {self.best_model_path}")


# =============================================================================
# 2. Model definition: fixed backbone + switchable CA-FiLM interaction structure
# =============================================================================
class TokenEmbedding(nn.Module):
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.value_embedding = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.value_embedding(x))


class PVTransformer_QuadTask_CAFiLMAblation(pl.LightningModule):
    """
    Fixed QuadTask architecture with CA-FiLM structure ablation.

    The model is based on the latest All-Aux Temporal CA-FiLM version:
    - CGC(Hetero) expert pool;
    - task-aware heads: DFI step-wise regression, Weather pooled classification,
      Ramp step-wise classification;
    - RouteReg on Weather/Ramp gates;
    - Nash-MTL for four tasks;
    - only the interaction mechanism before the power head is changed.
    """

    VALID_INTERACTIONS = {"none", "add", "concat", "shift", "scale", "film"}

    def __init__(
        self,
        nwp_dim: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        patch_len: int = 8,
        stride: int = 4,
        pred_len: int = 24,
        interaction_type: str = "film",
        route_threshold: float = 0.40,
        route_lambda: float = 3.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        if interaction_type not in self.VALID_INTERACTIONS:
            raise ValueError(
                f"interaction_type must be one of {sorted(self.VALID_INTERACTIONS)}, "
                f"got {interaction_type!r}."
            )

        self.nwp_dim = nwp_dim
        self.num_classes = num_classes
        self.d_model = d_model
        self.pred_len = pred_len
        self.interaction_type = interaction_type
        self.route_threshold = route_threshold
        self.route_lambda = route_lambda
        self.lr = lr
        self.weight_decay = weight_decay

        # ------------------------------------------------------------------
        # Backbone: identical to latest full model.
        # ------------------------------------------------------------------
        self.ts_patcher = TokenEmbedding(input_dim=7, d_model=d_model)
        self.nwp_patcher = TokenEmbedding(input_dim=nwp_dim + 6, d_model=d_model)
        self.pos_enc_ts = nn.Parameter(torch.randn(1, 100, d_model) * 0.1)
        self.pos_enc_nwp = nn.Parameter(torch.randn(1, 100, d_model) * 0.1)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=0.1,
            batch_first=True,
        )
        self.fusion_norm = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.2,
            norm_first=True,
        )
        self.shared_bottom = nn.TransformerEncoder(layer, num_layers=2)

        # ------------------------------------------------------------------
        # CGC(Hetero) experts: keep heterogeneous capacity unchanged.
        # ------------------------------------------------------------------
        self.expert_shared = nn.Sequential(
            nn.Linear(d_model, d_model * 3),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model * 3, d_model),
        )
        self.expert_main = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model, d_model),
        )
        self.expert_aux_ramp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model * 2, d_model),
        )
        self.expert_aux_dfi = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model // 2, d_model),
        )
        self.expert_aux_w = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model // 2, d_model),
        )

        self.gate_main = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2),
        )
        self.gate_aux_dfi = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2),
        )
        self.gate_aux_w = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2),
        )
        self.gate_aux_ramp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2),
        )

        # ------------------------------------------------------------------
        # Task-aware auxiliary heads: keep unchanged.
        # ------------------------------------------------------------------
        self.head_dfi = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1),
        )
        self.head_weather = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes),
        )
        self.head_ramp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 5),
        )

        # ------------------------------------------------------------------
        # Interaction modules. They are always instantiated to keep initialization
        # order consistent; the interaction_type controls which path is used.
        # ------------------------------------------------------------------
        input_phys_step_dim = 1 + num_classes + 5
        self.phys_projector = nn.Sequential(
            nn.Linear(input_phys_step_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Used by V2: Concat(F_main, F_aux) -> F_interact.
        self.concat_projector = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Used by V3/V4/OURS: FiLM generator.
        self.film_generator = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model * 2),
        )
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)

        # ------------------------------------------------------------------
        # Power head: keep unchanged.
        # ------------------------------------------------------------------
        flat_dim = d_model * pred_len
        self.head_power = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, pred_len),
        )

    @staticmethod
    def _gate_fuse(gate, gate_input, shared_feat, task_feat):
        weights = F.softmax(gate(gate_input), dim=-1)
        fused = weights[:, 0:1, None] * shared_feat + weights[:, 1:2, None] * task_feat
        return fused, weights

    def _build_aux_context(self, dfi_pred, w_logits, ramp_logits):
        """Build detached hourly auxiliary context: [B, P, 1 + C + 5]."""
        with torch.no_grad():
            dfi_context = dfi_pred.detach().unsqueeze(-1)  # [B, P, 1]
            weather_prob = F.softmax(w_logits.detach(), dim=-1)
            weather_context = weather_prob.unsqueeze(1).expand(-1, self.pred_len, -1)
            ramp_context = F.softmax(ramp_logits.detach(), dim=-1)
            phys_context_step = torch.cat(
                [dfi_context, weather_context, ramp_context],
                dim=-1,
            )
        return phys_context_step

    def _apply_interaction(self, fused_main, dfi_pred, w_logits, ramp_logits):
        """Apply the selected cross-task interaction mechanism."""
        if self.interaction_type == "none":
            # V0: CGC original bottom, no auxiliary-to-main interaction.
            return fused_main

        phys_context_step = self._build_aux_context(dfi_pred, w_logits, ramp_logits)
        aux_feat = self.phys_projector(phys_context_step)  # F(aux): [B, P, D]

        if self.interaction_type == "add":
            # V1: Simple Addition, F(main) + F(aux).
            return fused_main + aux_feat

        if self.interaction_type == "concat":
            # V2: Concatenation, Concat(F(main), F(aux)).
            return self.concat_projector(torch.cat([fused_main, aux_feat], dim=-1))

        film_input = torch.cat([fused_main, aux_feat], dim=-1)
        film_params = self.film_generator(film_input)
        gamma_raw, beta_raw = torch.chunk(film_params, 2, dim=-1)

        if self.interaction_type == "shift":
            # V3: beta-only CA-FiLM.
            return fused_main + beta_raw

        if self.interaction_type == "scale":
            # V4: gamma-only CA-FiLM.
            return fused_main * (1.0 + gamma_raw)

        if self.interaction_type == "film":
            # OURS: full CA-FiLM, scale + shift.
            return fused_main * (1.0 + gamma_raw) + beta_raw

        raise RuntimeError(f"Unexpected interaction_type={self.interaction_type!r}")

    def forward(self, ts, nwp):
        # 1. Dual-modal feature extraction.
        p_ts = self.ts_patcher(ts)
        p_nwp = self.nwp_patcher(nwp)
        p_ts = p_ts + self.pos_enc_ts[:, :p_ts.size(1), :]
        p_nwp = p_nwp + self.pos_enc_nwp[:, :p_nwp.size(1), :]

        attn_out, _ = self.cross_attention(query=p_nwp, key=p_ts, value=p_ts)
        fused_feat = self.fusion_norm(p_nwp + attn_out)
        bottom_out = self.shared_bottom(fused_feat)

        # 2. CGC expert routing.
        out_shared = self.expert_shared(bottom_out)
        out_main = self.expert_main(bottom_out)
        out_dfi = self.expert_aux_dfi(bottom_out)
        out_w = self.expert_aux_w(bottom_out)
        out_ramp = self.expert_aux_ramp(bottom_out)

        gate_input = torch.mean(bottom_out, dim=1)
        fused_main, w_main = self._gate_fuse(self.gate_main, gate_input, out_shared, out_main)
        fused_dfi, w_dfi = self._gate_fuse(self.gate_aux_dfi, gate_input, out_shared, out_dfi)
        fused_w, w_w = self._gate_fuse(self.gate_aux_w, gate_input, out_shared, out_w)
        fused_ramp, w_ramp = self._gate_fuse(self.gate_aux_ramp, gate_input, out_shared, out_ramp)

        # 3. Auxiliary predictions.
        dfi_pred = self.head_dfi(fused_dfi).squeeze(-1)       # [B, P]
        weather_context = torch.mean(fused_w, dim=1)          # [B, D]
        w_logits = self.head_weather(weather_context)         # [B, C]
        ramp_logits = self.head_ramp(fused_ramp)              # [B, P, 5]

        # 4. Cross-task interaction ablation before the power head.
        fused_main_refined = self._apply_interaction(fused_main, dfi_pred, w_logits, ramp_logits)
        p_pred = self.head_power(fused_main_refined)

        p_sh_dict = {
            "m": w_main[:, 0].mean().detach().item(),
            "d": w_dfi[:, 0].mean().detach().item(),
            "w": w_w[:, 0].mean().detach().item(),
            "r": w_ramp[:, 0].mean().detach().item(),
        }
        return p_pred, dfi_pred, w_logits, ramp_logits, p_sh_dict, w_w, w_ramp

    def configure_optimizers(self):
        base_opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        shared_params = (
            list(self.ts_patcher.parameters())
            + list(self.nwp_patcher.parameters())
            + [self.pos_enc_ts, self.pos_enc_nwp]
            + list(self.cross_attention.parameters())
            + list(self.fusion_norm.parameters())
            + list(self.expert_shared.parameters())
            + list(self.shared_bottom.parameters())
        )

        # Include all possible interaction modules in the main-task group.
        # Unused modules receive zero/no gradients for variants that do not use them.
        main_params = (
            list(self.expert_main.parameters())
            + list(self.gate_main.parameters())
            + list(self.phys_projector.parameters())
            + list(self.concat_projector.parameters())
            + list(self.film_generator.parameters())
            + list(self.head_power.parameters())
        )
        dfi_params = (
            list(self.expert_aux_dfi.parameters())
            + list(self.gate_aux_dfi.parameters())
            + list(self.head_dfi.parameters())
        )
        w_params = (
            list(self.expert_aux_w.parameters())
            + list(self.gate_aux_w.parameters())
            + list(self.head_weather.parameters())
        )
        ramp_params = (
            list(self.expert_aux_ramp.parameters())
            + list(self.gate_aux_ramp.parameters())
            + list(self.head_ramp.parameters())
        )

        self.nash_optimizer = NashMTLOptimizer(
            base_opt,
            shared_params,
            main_params,
            dfi_params,
            w_params,
            ramp_params,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            base_opt,
            T_0=15,
            T_mult=2,
            eta_min=1e-6,
        )
        return [base_opt], [scheduler]

    def training_step(self, batch, batch_idx):
        opt = self.nash_optimizer
        p_pred, dfi_pred, w_logits, ramp_logits, p_sh_dict, w_w, w_r = self(
            batch["ts_input"],
            batch["nwp_input"],
        )

        loss_p = F.huber_loss(p_pred, batch["target_p"], delta=0.5)
        loss_dfi = F.mse_loss(dfi_pred, batch["target_dfi"])
        loss_w_raw = F.cross_entropy(w_logits, batch["daily_weather"])
        loss_ramp_raw = F.cross_entropy(
            ramp_logits.reshape(-1, 5),
            batch["target_ramp"].reshape(-1),
        )

        # RouteReg is kept unchanged across all CA-FiLM structure ablations.
        penalty_w = F.relu(w_w[:, 0] - self.route_threshold).mean() * self.route_lambda
        penalty_r = F.relu(w_r[:, 0] - self.route_threshold).mean() * self.route_lambda
        loss_w = loss_w_raw + penalty_w
        loss_ramp = loss_ramp_raw + penalty_r

        metrics = opt.backward(loss_p, loss_dfi, loss_w, loss_ramp)

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        opt.step()
        if self.trainer.is_last_batch:
            self.lr_schedulers().step()

        self.log("train_loss_p", loss_p, prog_bar=True)
        self.log("1_Loss/Aux_DFI_Raw", loss_dfi, prog_bar=False)
        self.log("1_Loss/Aux_Weather_Raw", loss_w, prog_bar=False)
        self.log("1_Loss/Aux_Ramp_Raw", loss_ramp, prog_bar=False)
        self.log("1_Loss/RouteReg_Weather", penalty_w, prog_bar=False)
        self.log("1_Loss/RouteReg_Ramp", penalty_r, prog_bar=False)

        self.log("6_Gate/Main_use_Shared", p_sh_dict["m"], prog_bar=False)
        self.log("6_Gate/DFI_use_Shared", p_sh_dict["d"], prog_bar=False)
        self.log("6_Gate/Weather_use_Shared", p_sh_dict["w"], prog_bar=False)
        self.log("6_Gate/Ramp_use_Shared", p_sh_dict["r"], prog_bar=False)

        self.log("3_Nash/CosTheta_Overall_Combined", metrics["cos_combined"], prog_bar=True)
        self.log("7_Gradient_Angle/Cos_Main_vs_DFI", metrics["cos_md"], prog_bar=False)
        self.log("7_Gradient_Angle/Cos_Main_vs_Weather", metrics["cos_mw"], prog_bar=False)
        self.log("7_Gradient_Angle/Cos_Main_vs_Ramp", metrics["cos_mr"], prog_bar=False)
        return loss_p

    def validation_step(self, batch, batch_idx):
        p_pred, _, _, _, _, _, _ = self(batch["ts_input"], batch["nwp_input"])
        mae = F.l1_loss(p_pred, batch["target_p"])
        self.log("val_mae", mae, on_epoch=True, prog_bar=True)
        return mae


# =============================================================================
# 3. Ablation configs and result aggregation
# =============================================================================
CAFILM_STRUCTURE_CONFIGS = [
    {
        "id": "V0",
        "name": "None_CGCOriginalBottom",
        "interaction_type": "none",
        "interaction_label": "None",
        "variant": "CGC original bottom",
    },
    {
        "id": "V1",
        "name": "SimpleAddition",
        "interaction_type": "add",
        "interaction_label": "Simple Addition",
        "variant": "F(main)+F(aux)",
    },
    {
        "id": "V2",
        "name": "Concatenation",
        "interaction_type": "concat",
        "interaction_label": "Concatenation",
        "variant": "Concat(F(main),F(aux))",
    },
    {
        "id": "V3",
        "name": "CAFiLM_ShiftOnly",
        "interaction_type": "shift",
        "interaction_label": "CA-FiLM",
        "variant": "Shift-only beta",
    },
    {
        "id": "V4",
        "name": "CAFiLM_ScaleOnly",
        "interaction_type": "scale",
        "interaction_label": "CA-FiLM",
        "variant": "Scale-only gamma",
    },
    {
        "id": "OURS",
        "name": "CAFiLM_ScaleShift",
        "interaction_type": "film",
        "interaction_label": "CA-FiLM",
        "variant": "Scale gamma + Shift beta",
    },
]


def _metric_file_to_long(path: Path, keys: Dict) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    for k, v in keys.items():
        df[k] = v
    return df


def summarize_cafilm_structure_results(exp_root: Path, dataset_name: str, seeds: List[int]) -> None:
    summary_dir = exp_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    power_all = []
    weather_all = []
    aux_all = []
    stress_all = []

    for cfg in CAFILM_STRUCTURE_CONFIGS:
        exp_name = f"{cfg['id']}_{cfg['name']}"
        run_root = exp_root / exp_name
        for seed in seeds:
            run_dir = run_root / f"seed_{seed}"
            file_name = f"{dataset_name}_{exp_name}_seed{seed}"
            metrics_dir = run_dir / "metrics"
            keys = {
                "exp_id": cfg["id"],
                "exp_name": cfg["name"],
                "interaction": cfg["interaction_label"],
                "variant": cfg["variant"],
                "seed": seed,
            }
            power_all.append(_metric_file_to_long(metrics_dir / f"{file_name}_power_metrics.csv", keys))
            weather_all.append(_metric_file_to_long(metrics_dir / f"{file_name}_weather_metrics.csv", keys))
            aux_all.append(_metric_file_to_long(metrics_dir / f"{file_name}_auxiliary_metrics.csv", keys))
            stress_all.append(_metric_file_to_long(metrics_dir / f"{file_name}_stress_metrics.csv", keys))

    def _concat(frames):
        frames = [x for x in frames if x is not None and not x.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    power_df = _concat(power_all)
    weather_df = _concat(weather_all)
    aux_df = _concat(aux_all)
    stress_df = _concat(stress_all)

    if not power_df.empty:
        power_df.to_csv(summary_dir / "power_metrics_all_seeds_long.csv", index=False, encoding="utf-8-sig")
        power_summary = (
            power_df.groupby(["exp_id", "exp_name", "interaction", "variant", "metric"], as_index=False)
            .agg(mean=("value", "mean"), std=("value", "std"), seeds=("seed", "nunique"))
        )
        power_summary.to_csv(summary_dir / "power_metrics_mean_std.csv", index=False, encoding="utf-8-sig")

        target_metrics = ["R2", "MAE_MW", "RMSE_MW", "NMAE_percent", "NRMSE_percent"]
        compact = power_summary[power_summary["metric"].isin(target_metrics)].copy()
        compact["mean_std"] = compact.apply(
            lambda r: f"{r['mean']:.6f} ± {0.0 if pd.isna(r['std']) else r['std']:.6f}",
            axis=1,
        )
        compact = compact.pivot_table(
            index=["exp_id", "exp_name", "interaction", "variant"],
            columns="metric",
            values="mean_std",
            aggfunc="first",
        ).reset_index()
        compact.to_csv(summary_dir / "cafilm_structure_power_summary.csv", index=False, encoding="utf-8-sig")

    if not weather_df.empty:
        weather_df.to_csv(summary_dir / "weather_metrics_all_seeds_long.csv", index=False, encoding="utf-8-sig")
        weather_summary = (
            weather_df.groupby(["exp_id", "exp_name", "interaction", "variant", "weather_id", "weather_name", "metric"], as_index=False)
            .agg(mean=("value", "mean"), std=("value", "std"), seeds=("seed", "nunique"))
        )
        weather_summary.to_csv(summary_dir / "weather_metrics_mean_std.csv", index=False, encoding="utf-8-sig")

    if not aux_df.empty:
        aux_df.to_csv(summary_dir / "auxiliary_metrics_all_seeds_long.csv", index=False, encoding="utf-8-sig")
        aux_summary = (
            aux_df.groupby(["exp_id", "exp_name", "interaction", "variant", "task", "metric"], as_index=False)
            .agg(mean=("value", "mean"), std=("value", "std"), seeds=("seed", "nunique"))
        )
        aux_summary.to_csv(summary_dir / "auxiliary_metrics_mean_std.csv", index=False, encoding="utf-8-sig")

    if not stress_df.empty:
        stress_df.to_csv(summary_dir / "stress_metrics_all_seeds_long.csv", index=False, encoding="utf-8-sig")
        stress_summary = (
            stress_df.groupby(["exp_id", "exp_name", "interaction", "variant", "scenario", "metric"], as_index=False)
            .agg(mean=("value", "mean"), std=("value", "std"), seeds=("seed", "nunique"))
        )
        stress_summary.to_csv(summary_dir / "stress_metrics_mean_std.csv", index=False, encoding="utf-8-sig")

    print(f"\n[SUMMARY] CA-FiLM structure ablation summaries saved to: {summary_dir}")


# =============================================================================
# 4. Main runner
# =============================================================================
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # CONFIG: edit these paths and settings for your machine.
    # ------------------------------------------------------------------
    DATASET_NAME = "private"
    SEEDS = [0, 1, 2, 3, 4]

    BASE_DIR = "/home/cyx/FusionSF/MTL/PV_QuadTask_Project/data"
    MTL_FILE = os.path.join(BASE_DIR, "MTL_Dataset_pro.csv")
    NWP_FILE = os.path.join(BASE_DIR, "nwp_filtered_clean_1H.csv")
    DAILY_LABEL_FILE = os.path.join(BASE_DIR, "历史天级别-处理.csv")

    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.1
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    MAX_EPOCHS = 1000
    PATIENCE = 20

    D_MODEL = 128
    NHEAD = 4
    PRED_LEN = 24
    LR = 1e-4
    WEIGHT_DECAY = 1e-2
    ROUTE_THRESHOLD = 0.40
    ROUTE_LAMBDA = 3.0

    PROJECT_DIR = Path("/home/cyx/FusionSF/MTL/PV_QuadTask_Project")
    EXP_ROOT = PROJECT_DIR / "experiments" / DATASET_NAME / "CAFiLM_Structure_Ablation"
    EXP_ROOT.mkdir(parents=True, exist_ok=True)

    for f in [MTL_FILE, NWP_FILE, DAILY_LABEL_FILE]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing required file: {f}")

    for cfg in CAFILM_STRUCTURE_CONFIGS:
        exp_name = f"{cfg['id']}_{cfg['name']}"
        exp_dir = EXP_ROOT / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "#" * 100)
        print(f"[CA-FILM ABLATION] {exp_name} | interaction={cfg['interaction_type']} | {cfg['variant']}")
        print("#" * 100)

        for seed in SEEDS:
            print("\n" + "=" * 90)
            print(f"[RUN] Dataset={DATASET_NAME} | Experiment={exp_name} | Seed={seed}")
            print("=" * 90)

            seed_everything(seed)

            run_dir = exp_dir / f"seed_{seed}"
            ckpt_dir = run_dir / "checkpoints"
            for d in [ckpt_dir, run_dir / "predictions", run_dir / "metrics"]:
                d.mkdir(parents=True, exist_ok=True)

            dataset = PVQuadTaskDataset(
                MTL_FILE,
                NWP_FILE,
                DAILY_LABEL_FILE,
                train_ratio=TRAIN_RATIO,
                pred_len=PRED_LEN,
            )

            total_len = len(dataset)
            train_len = int(TRAIN_RATIO * total_len)
            val_len = int(VAL_RATIO * total_len)
            test_start = train_len + val_len

            train_indices = list(range(0, train_len))
            val_indices = list(range(train_len, test_start))
            test_indices = list(range(test_start, total_len))

            split_info = {
                "total_windows": total_len,
                "train_len": train_len,
                "val_len": val_len,
                "test_len": len(test_indices),
                "train_ratio": TRAIN_RATIO,
                "val_ratio": VAL_RATIO,
                "seq_len": getattr(dataset, "seq_len", None),
                "pred_len": getattr(dataset, "pred_len", None),
                "train_indices": train_indices,
                "val_indices": val_indices,
                "test_indices": test_indices,
            }
            with open(run_dir / "split_indices.json", "w", encoding="utf-8") as f:
                json.dump(split_info, f, ensure_ascii=False, indent=2)

            train_loader = DataLoader(
                Subset(dataset, train_indices),
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
            )
            val_loader = DataLoader(
                Subset(dataset, val_indices),
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
            )
            test_loader = DataLoader(
                Subset(dataset, test_indices),
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
            )

            model = PVTransformer_QuadTask_CAFiLMAblation(
                nwp_dim=dataset.nwp_dim,
                num_classes=dataset.num_classes,
                d_model=D_MODEL,
                nhead=NHEAD,
                pred_len=PRED_LEN,
                interaction_type=cfg["interaction_type"],
                route_threshold=ROUTE_THRESHOLD,
                route_lambda=ROUTE_LAMBDA,
                lr=LR,
                weight_decay=WEIGHT_DECAY,
            )

            config = {
                "dataset_name": DATASET_NAME,
                "exp_name": exp_name,
                "seed": seed,
                "cafilm_structure_ablation": cfg,
                "mtl_file": MTL_FILE,
                "nwp_file": NWP_FILE,
                "daily_label_file": DAILY_LABEL_FILE,
                "train_ratio": TRAIN_RATIO,
                "val_ratio": VAL_RATIO,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "model_params": {
                    "nwp_dim": dataset.nwp_dim,
                    "num_classes": dataset.num_classes,
                    "d_model": D_MODEL,
                    "nhead": NHEAD,
                    "pred_len": PRED_LEN,
                    "interaction_type": cfg["interaction_type"],
                    "route_threshold": ROUTE_THRESHOLD,
                    "route_lambda": ROUTE_LAMBDA,
                    "lr": LR,
                    "weight_decay": WEIGHT_DECAY,
                },
            }
            with open(run_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

            early_stop = EarlyStopping(monitor="val_mae", patience=PATIENCE, mode="min")
            my_checkpoint = HardcoreCheckpoint(dirpath=ckpt_dir, save_last=False)

            trainer = pl.Trainer(
                max_epochs=MAX_EPOCHS,
                callbacks=[early_stop, my_checkpoint],
                accelerator="auto",
                devices=1,
                log_every_n_steps=5,
                num_sanity_val_steps=0,
                default_root_dir=str(run_dir / "lightning_logs"),
            )

            print(f"Start training: {exp_name}, seed={seed}")
            trainer.fit(model, train_loader, val_loader)

            best_model_path = my_checkpoint.best_model_path
            if best_model_path and os.path.exists(best_model_path):
                print(f"\nLoading best validation checkpoint: {best_model_path}")
                best_model = PVTransformer_QuadTask_CAFiLMAblation.load_from_checkpoint(
                    checkpoint_path=best_model_path,
                    nwp_dim=dataset.nwp_dim,
                    num_classes=dataset.num_classes,
                    d_model=D_MODEL,
                    nhead=NHEAD,
                    pred_len=PRED_LEN,
                    interaction_type=cfg["interaction_type"],
                    route_threshold=ROUTE_THRESHOLD,
                    route_lambda=ROUTE_LAMBDA,
                    lr=LR,
                    weight_decay=WEIGHT_DECAY,
                )
            else:
                print("\n[WARN] best.ckpt not found; using in-memory final model for testing.")
                best_model = model

            eval_device = "cuda" if torch.cuda.is_available() else "cpu"
            run_name = f"{DATASET_NAME}_{exp_name}_seed{seed}"

            evaluate_model_comprehensive(
                best_model,
                test_loader,
                dataset,
                file_name=run_name,
                device=eval_device,
                output_dir=str(run_dir),
                dataset_name=DATASET_NAME,
                model_name=exp_name,
                seed=seed,
                make_plots=(seed == SEEDS[0] and cfg["id"] == "OURS"),
                compute_aux=True,
            )

            print(f"[DONE] {run_name}: outputs saved to {run_dir}")

    summarize_cafilm_structure_results(EXP_ROOT, DATASET_NAME, SEEDS)
