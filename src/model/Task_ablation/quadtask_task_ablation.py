"""
Fixed-full-structure QuadTask auxiliary-task ablation runner (8 variants x 5 seeds).

This script implements the "fixed complete architecture + training switches" protocol
for the All-Aux Temporal CA-FiLM model:

    Backbone + CGC + DFI/Weather/Ramp heads + All-Aux Temporal CA-FiLM + Nash-MTL

Important design choice
-----------------------
All modules are ALWAYS instantiated in the same order as the full model. The switches
only control whether a task:
  1) contributes its auxiliary loss;
  2) participates in Nash-MTL;
  3) enters the Temporal CA-FiLM physical context.

When a task is disabled, its corresponding CA-FiLM context channel is replaced by zeros
so that the phys_projector input dimension remains fixed: DFI(1) + Weather(C) + Ramp(5).
This keeps architecture/initialization order aligned across ablation rows while preserving
ordinary auxiliary-task ablation semantics.

Run:
    python run_quadtask_task_ablation_fixed_allaux_temporal_5seeds.py

Outputs:
experiments/<DATASET_NAME>/QuadTask_TaskAblation_FixedFull_AllAuxTemporal/
  ablation_01_MainOnly/seed_0/...
  ...
  summary/
    power_metrics_all_seeds_long.csv
    power_metrics_mean_std.csv
    ablation_power_summary.csv
    weather_metrics_mean_std.csv
    auxiliary_metrics_mean_std.csv
    stress_metrics_mean_std.csv
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
from src.evaluator.Pic_evaluator_detailed import evaluate_model_comprehensive


# =============================================================================
# 1. Reproducibility and checkpointing
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
# 2. Fixed-order flexible Nash optimizer
# =============================================================================
class FixedOrderFlexibleNashMTLOptimizer:
    """Nash-MTL optimizer with fixed task order but variable active tasks.

    Fixed order: main -> dfi -> weather -> ramp.
    Disabled tasks are simply absent from the Nash game; they are not represented by
    zero losses, so they do not act as fake players.
    """

    TASK_ORDER = ["main", "dfi", "weather", "ramp"]

    def __init__(self, optimizer, shared_params, task_param_groups: Dict[str, List[nn.Parameter]]):
        self._optim = optimizer
        self.shared_params = list(shared_params)
        self.task_param_groups = {k: list(v) for k, v in task_param_groups.items()}

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        return self._optim.zero_grad()

    def step(self):
        return self._optim.step()

    def backward(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        active_names = [name for name in self.TASK_ORDER if name in losses and losses[name] is not None]
        if "main" not in active_names:
            raise ValueError("FixedOrderFlexibleNashMTLOptimizer requires a 'main' loss.")

        # Single-task fallback: ordinary backward.
        if len(active_names) == 1:
            self.zero_grad()
            losses["main"].backward()
            return {
                "cos_combined": 1.0,
                "cos_main_vs_dfi": 0.0,
                "cos_main_vs_weather": 0.0,
                "cos_main_vs_ramp": 0.0,
                "num_active_tasks": 1.0,
            }

        shared_grads = {}
        task_specific_grads = {}

        for i, name in enumerate(active_names):
            self.zero_grad()
            retain = i < len(active_names) - 1
            losses[name].backward(retain_graph=retain)
            shared_grads[name] = self._get_grads(self.shared_params)
            task_specific_grads[name] = self._get_grads(self.task_param_groups.get(name, []))

        flat = {name: torch.cat([g.flatten() for g in grads]) for name, grads in shared_grads.items()}
        norms = {name: torch.norm(flat[name]) + 1e-8 for name in active_names}
        unit = {name: flat[name] / norms[name] for name in active_names}

        n = len(active_names)
        C = torch.empty((n, n), device=flat[active_names[0]].device)
        for i, ni in enumerate(active_names):
            for j, nj in enumerate(active_names):
                C[i, j] = torch.dot(unit[ni], unit[nj])
        C = C + torch.eye(n, device=C.device) * 1e-4

        alpha = torch.ones(n, device=C.device)
        for _ in range(10):
            grad_alpha = torch.matmul(C, alpha) - 1.0 / (alpha + 1e-8)
            hessian = C + torch.diag(1.0 / (alpha ** 2 + 1e-8))
            try:
                delta = torch.linalg.solve(hessian, grad_alpha)
            except RuntimeError:
                delta = grad_alpha * 0.1
            alpha = torch.clamp(alpha - delta, min=1e-4)

        alpha_norm = alpha / torch.sum(alpha)
        g_shared_direction = sum(alpha_norm[i] * unit[name] for i, name in enumerate(active_names))
        avg_norm = sum(norms[name] for name in active_names) / float(n)
        g_shared_final_flat = g_shared_direction * avg_norm

        offset = 0
        g_shared_final = []
        for p in self.shared_params:
            numel = p.numel()
            g_shared_final.append(g_shared_final_flat[offset:offset + numel].view(p.shape))
            offset += numel

        self.zero_grad()
        for p, g in zip(self.shared_params, g_shared_final):
            p.grad = g
        for name in active_names:
            for p, g in zip(self.task_param_groups.get(name, []), task_specific_grads[name]):
                p.grad = g

        main_norm = norms["main"]
        combined_norm = torch.norm(g_shared_final_flat) + 1e-8
        cos_combined = torch.dot(flat["main"], g_shared_final_flat) / (main_norm * combined_norm + 1e-8)

        def _cos_with_main(task_name: str) -> float:
            if task_name not in active_names:
                return 0.0
            value = torch.dot(unit["main"], unit[task_name]).item()
            return value if math.isfinite(value) else 0.0

        cos_combined_value = float(cos_combined.item())
        if not math.isfinite(cos_combined_value):
            cos_combined_value = 0.0

        return {
            "cos_combined": cos_combined_value,
            "cos_main_vs_dfi": _cos_with_main("dfi"),
            "cos_main_vs_weather": _cos_with_main("weather"),
            "cos_main_vs_ramp": _cos_with_main("ramp"),
            "num_active_tasks": float(n),
        }

    @staticmethod
    def _get_grads(params: List[nn.Parameter]) -> List[torch.Tensor]:
        return [torch.zeros_like(p).to(p.device) if p.grad is None else p.grad.clone() for p in params]


# =============================================================================
# 3. Fixed-full-structure switchable model
# =============================================================================
class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.value_embedding = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.value_embedding(x))


class PVTransformer_QuadTask_FixedFullAblation(pl.LightningModule):
    """Fixed complete architecture + training switches.

    This class keeps the same module list/order as the full All-Aux Temporal CA-FiLM model:
      - DFI, Weather, Ramp branches are always instantiated;
      - the All-Aux Temporal CA-FiLM projector is always instantiated with input dimension 1+C+5;
      - task switches control loss/Nash/CA-FiLM contribution only.
    """

    def __init__(
        self,
        nwp_dim: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        patch_len: int = 8,
        stride: int = 4,
        pred_len: int = 24,
        use_dfi: bool = True,
        use_weather: bool = True,
        use_ramp: bool = True,
        gate_penalty_threshold: float = 0.40,
        gate_penalty_weight: float = 3.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.nwp_dim = nwp_dim
        self.num_classes = num_classes
        self.d_model = d_model
        self.pred_len = pred_len
        self.use_dfi = bool(use_dfi)
        self.use_weather = bool(use_weather)
        self.use_ramp = bool(use_ramp)
        self.gate_penalty_threshold = gate_penalty_threshold
        self.gate_penalty_weight = gate_penalty_weight
        self.lr = lr
        self.weight_decay = weight_decay

        # ------------------------------------------------------------------
        # 1. Backbone: same as full model.
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
        # 2. CGC experts: instantiated in the same order as the full model.
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

        flat_dim = d_model * pred_len

        # ------------------------------------------------------------------
        # 3. Task-aware auxiliary heads: always instantiated.
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
        # 4. All-Aux Temporal CA-FiLM: always fixed as DFI(1)+Weather(C)+Ramp(5).
        # Disabled tasks contribute zero context.
        # ------------------------------------------------------------------
        input_phys_step_dim = 1 + num_classes + 5
        self.phys_projector = nn.Sequential(
            nn.Linear(input_phys_step_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.film_generator = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model * 2),
        )
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)

        # ------------------------------------------------------------------
        # 5. Main prediction head.
        # ------------------------------------------------------------------
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

    def forward(self, ts, nwp):
        B = ts.shape[0]
        device = ts.device
        dtype = ts.dtype

        # Backbone.
        p_ts = self.ts_patcher(ts)
        p_nwp = self.nwp_patcher(nwp)
        p_ts = p_ts + self.pos_enc_ts[:, :p_ts.size(1), :]
        p_nwp = p_nwp + self.pos_enc_nwp[:, :p_nwp.size(1), :]

        attn_out, _ = self.cross_attention(query=p_nwp, key=p_ts, value=p_ts)
        fused_feat = self.fusion_norm(p_nwp + attn_out)
        bottom_out = self.shared_bottom(fused_feat)

        # CGC expert routing: compute all branches to preserve the complete architecture path.
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

        # Auxiliary predictions are always computed, but only enabled tasks are supervised/used as CA-FiLM contexts.
        dfi_pred_raw = self.head_dfi(fused_dfi).squeeze(-1)              # [B, P]
        weather_context_raw = torch.mean(fused_w, dim=1)                # [B, D]
        w_logits_raw = self.head_weather(weather_context_raw)           # [B, C]
        ramp_logits_raw = self.head_ramp(fused_ramp)                    # [B, P, 5]

        # Return dummy outputs for disabled tasks, so auxiliary metrics are not interpreted as valid.
        dfi_pred = dfi_pred_raw if self.use_dfi else torch.zeros(B, self.pred_len, device=device, dtype=dtype)
        w_logits = w_logits_raw if self.use_weather else torch.zeros(B, self.num_classes, device=device, dtype=dtype)
        ramp_logits = ramp_logits_raw if self.use_ramp else torch.zeros(B, self.pred_len, 5, device=device, dtype=dtype)

        # Fixed-dimension Temporal CA-FiLM context.
        with torch.no_grad():
            if self.use_dfi:
                dfi_context = dfi_pred_raw.detach().unsqueeze(-1)
            else:
                dfi_context = torch.zeros(B, self.pred_len, 1, device=device, dtype=dtype)

            if self.use_weather:
                weather_prob = F.softmax(w_logits_raw.detach(), dim=-1)
                weather_context = weather_prob.unsqueeze(1).expand(-1, self.pred_len, -1)
            else:
                weather_context = torch.zeros(B, self.pred_len, self.num_classes, device=device, dtype=dtype)

            if self.use_ramp:
                ramp_context = F.softmax(ramp_logits_raw.detach(), dim=-1)
            else:
                ramp_context = torch.zeros(B, self.pred_len, 5, device=device, dtype=dtype)

            phys_context_step = torch.cat([dfi_context, weather_context, ramp_context], dim=-1)

        phys_embedding_step = self.phys_projector(phys_context_step)
        combined_info = torch.cat([fused_main, phys_embedding_step], dim=-1)
        film_params = self.film_generator(combined_info)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        fused_main_refined = fused_main * (1.0 + gamma) + beta

        p_pred = self.head_power(fused_main_refined)

        p_sh_dict = {
            "m": w_main[:, 0].mean().detach().item(),
            "d": w_dfi[:, 0].mean().detach().item() if self.use_dfi else 0.0,
            "w": w_w[:, 0].mean().detach().item() if self.use_weather else 0.0,
            "r": w_ramp[:, 0].mean().detach().item() if self.use_ramp else 0.0,
        }

        return p_pred, dfi_pred, w_logits, ramp_logits, p_sh_dict, w_w, w_ramp

    def configure_optimizers(self):
        base_opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        shared_params = (
            list(self.ts_patcher.parameters()) +
            list(self.nwp_patcher.parameters()) +
            [self.pos_enc_ts, self.pos_enc_nwp] +
            list(self.cross_attention.parameters()) +
            list(self.fusion_norm.parameters()) +
            list(self.expert_shared.parameters()) +
            list(self.shared_bottom.parameters())
        )

        main_params = (
            list(self.expert_main.parameters()) +
            list(self.gate_main.parameters()) +
            list(self.phys_projector.parameters()) +
            list(self.film_generator.parameters()) +
            list(self.head_power.parameters())
        )
        dfi_params = (
            list(self.expert_aux_dfi.parameters()) +
            list(self.gate_aux_dfi.parameters()) +
            list(self.head_dfi.parameters())
        )
        weather_params = (
            list(self.expert_aux_w.parameters()) +
            list(self.gate_aux_w.parameters()) +
            list(self.head_weather.parameters())
        )
        ramp_params = (
            list(self.expert_aux_ramp.parameters()) +
            list(self.gate_aux_ramp.parameters()) +
            list(self.head_ramp.parameters())
        )

        task_param_groups = {"main": list(main_params)}
        if self.use_dfi:
            task_param_groups["dfi"] = list(dfi_params)
        if self.use_weather:
            task_param_groups["weather"] = list(weather_params)
        if self.use_ramp:
            task_param_groups["ramp"] = list(ramp_params)

        self.nash_optimizer = FixedOrderFlexibleNashMTLOptimizer(base_opt, shared_params, task_param_groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            base_opt, T_0=15, T_mult=2, eta_min=1e-6
        )
        return [base_opt], [scheduler]

    def training_step(self, batch, batch_idx):
        opt = self.nash_optimizer
        p_pred, dfi_pred, w_logits, ramp_logits, p_sh_dict, w_w, w_r = self(
            batch["ts_input"], batch["nwp_input"]
        )

        loss_p = F.huber_loss(p_pred, batch["target_p"], delta=0.5)
        losses = {"main": loss_p}

        loss_dfi = torch.tensor(0.0, device=self.device)
        loss_w = torch.tensor(0.0, device=self.device)
        loss_ramp = torch.tensor(0.0, device=self.device)
        penalty_w = torch.tensor(0.0, device=self.device)
        penalty_r = torch.tensor(0.0, device=self.device)

        if self.use_dfi:
            loss_dfi = F.mse_loss(dfi_pred, batch["target_dfi"])
            losses["dfi"] = loss_dfi

        if self.use_weather:
            loss_w = F.cross_entropy(w_logits, batch["daily_weather"])
            penalty_w = F.relu(w_w[:, 0] - self.gate_penalty_threshold).mean() * self.gate_penalty_weight
            loss_w = loss_w + penalty_w
            losses["weather"] = loss_w

        if self.use_ramp:
            loss_ramp = F.cross_entropy(ramp_logits.reshape(-1, 5), batch["target_ramp"].reshape(-1))
            penalty_r = F.relu(w_r[:, 0] - self.gate_penalty_threshold).mean() * self.gate_penalty_weight
            loss_ramp = loss_ramp + penalty_r
            losses["ramp"] = loss_ramp

        for loss_name, loss_value in losses.items():
            if not torch.isfinite(loss_value):
                raise RuntimeError(f"[NaN/Inf] {loss_name} loss is invalid: {loss_value.item()}")

        metrics = opt.backward(losses)

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        opt.step()
        if self.trainer.is_last_batch:
            self.lr_schedulers().step()

        self.log("train_loss_p", loss_p, prog_bar=True)
        self.log("1_Loss/Aux_DFI", loss_dfi, prog_bar=False)
        self.log("1_Loss/Aux_Weather", loss_w, prog_bar=False)
        self.log("1_Loss/Aux_Ramp", loss_ramp, prog_bar=False)
        self.log("1_Loss/Penalty_Weather", penalty_w, prog_bar=False)
        self.log("1_Loss/Penalty_Ramp", penalty_r, prog_bar=False)

        self.log("6_Gate/Main_use_Shared", p_sh_dict["m"], prog_bar=False)
        self.log("6_Gate/DFI_use_Shared", p_sh_dict["d"], prog_bar=False)
        self.log("6_Gate/Weather_use_Shared", p_sh_dict["w"], prog_bar=False)
        self.log("6_Gate/Ramp_use_Shared", p_sh_dict["r"], prog_bar=False)

        self.log("3_Nash/CosTheta_Overall_Combined", metrics["cos_combined"], prog_bar=True)
        self.log("7_Gradient_Angle/Cos_Main_vs_DFI", metrics["cos_main_vs_dfi"], prog_bar=False)
        self.log("7_Gradient_Angle/Cos_Main_vs_Weather", metrics["cos_main_vs_weather"], prog_bar=False)
        self.log("7_Gradient_Angle/Cos_Main_vs_Ramp", metrics["cos_main_vs_ramp"], prog_bar=False)
        self.log("3_Nash/Num_Active_Tasks", metrics["num_active_tasks"], prog_bar=False)

        return loss_p

    def validation_step(self, batch, batch_idx):
        p_pred, _, _, _, _, _, _ = self(batch["ts_input"], batch["nwp_input"])
        mae = F.l1_loss(p_pred, batch["target_p"])
        self.log("val_mae", mae, on_epoch=True, prog_bar=True)
        return mae


# =============================================================================
# 4. Ablation configs and result aggregation
# =============================================================================
ABLATION_CONFIGS = [
    {"id": 1, "name": "MainOnly",             "use_dfi": False, "use_weather": False, "use_ramp": False},
    {"id": 2, "name": "DFI",                  "use_dfi": True,  "use_weather": False, "use_ramp": False},
    {"id": 3, "name": "Weather",              "use_dfi": False, "use_weather": True,  "use_ramp": False},
    {"id": 4, "name": "Ramp",                 "use_dfi": False, "use_weather": False, "use_ramp": True},
    {"id": 5, "name": "DFI_Weather",          "use_dfi": True,  "use_weather": True,  "use_ramp": False},
    {"id": 6, "name": "DFI_Ramp",             "use_dfi": True,  "use_weather": False, "use_ramp": True},
    {"id": 7, "name": "Weather_Ramp",         "use_dfi": False, "use_weather": True,  "use_ramp": True},
    {"id": 8, "name": "DFI_Weather_Ramp",     "use_dfi": True,  "use_weather": True,  "use_ramp": True},
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


def summarize_ablation_results(exp_root: Path, dataset_name: str, seeds: List[int]) -> None:
    summary_dir = exp_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    power_all, weather_all, aux_all, stress_all = [], [], [], []

    for cfg in ABLATION_CONFIGS:
        ablation_dir = exp_root / f"ablation_{cfg['id']:02d}_{cfg['name']}"
        model_name = f"A{cfg['id']:02d}_{cfg['name']}"
        for seed in seeds:
            run_dir = ablation_dir / f"seed_{seed}"
            file_name = f"{dataset_name}_{model_name}_seed{seed}"
            metrics_dir = run_dir / "metrics"
            keys = {
                "ablation_id": cfg["id"],
                "ablation_name": cfg["name"],
                "model": model_name,
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
            power_df.groupby(["ablation_id", "ablation_name", "metric"], as_index=False)
            .agg(mean=("value", "mean"), std=("value", "std"), seeds=("seed", "nunique"))
        )
        power_summary.to_csv(summary_dir / "power_metrics_mean_std.csv", index=False, encoding="utf-8-sig")

        target_metrics = ["R2", "MAE_MW", "RMSE_MW", "NMAE_percent", "NRMSE_percent"]
        compact = power_summary[power_summary["metric"].isin(target_metrics)].copy()
        compact["mean_std"] = compact.apply(
            lambda r: f"{r['mean']:.6f} ± {0.0 if pd.isna(r['std']) else r['std']:.6f}", axis=1
        )
        compact = compact.pivot_table(
            index=["ablation_id", "ablation_name"],
            columns="metric",
            values="mean_std",
            aggfunc="first",
        ).reset_index()
        compact.to_csv(summary_dir / "ablation_power_summary.csv", index=False, encoding="utf-8-sig")

    if not weather_df.empty:
        weather_df.to_csv(summary_dir / "weather_metrics_all_seeds_long.csv", index=False, encoding="utf-8-sig")
        weather_summary = (
            weather_df.groupby(["ablation_id", "ablation_name", "weather_id", "weather_name", "metric"], as_index=False)
            .agg(mean=("value", "mean"), std=("value", "std"), seeds=("seed", "nunique"))
        )
        weather_summary.to_csv(summary_dir / "weather_metrics_mean_std.csv", index=False, encoding="utf-8-sig")

    if not aux_df.empty:
        aux_df.to_csv(summary_dir / "auxiliary_metrics_all_seeds_long.csv", index=False, encoding="utf-8-sig")
        aux_summary = (
            aux_df.groupby(["ablation_id", "ablation_name", "task", "metric"], as_index=False)
            .agg(mean=("value", "mean"), std=("value", "std"), seeds=("seed", "nunique"))
        )
        aux_summary.to_csv(summary_dir / "auxiliary_metrics_mean_std.csv", index=False, encoding="utf-8-sig")

    if not stress_df.empty:
        stress_df.to_csv(summary_dir / "stress_metrics_all_seeds_long.csv", index=False, encoding="utf-8-sig")
        stress_summary = (
            stress_df.groupby(["ablation_id", "ablation_name", "scenario", "metric"], as_index=False)
            .agg(mean=("value", "mean"), std=("value", "std"), seeds=("seed", "nunique"))
        )
        stress_summary.to_csv(summary_dir / "stress_metrics_mean_std.csv", index=False, encoding="utf-8-sig")

    print(f"\n[SUMMARY] Aggregated result files saved to: {summary_dir}")


# =============================================================================
# 5. Main runner: 8 ablations x 5 seeds
# =============================================================================
if __name__ == "__main__":
    DATASET_NAME = "private"
    SEEDS = [0, 1, 2, 3, 4]

    BASE_DIR = "/home/cyx/FusionSF/MTL/PV_QuadTask_Project/data"
    MTL_FILE = os.path.join(BASE_DIR, "MTL_Dataset_pro.csv")
    NWP_FILE = os.path.join(BASE_DIR, "nwp_filtered_clean_1H_new.csv")
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

    PROJECT_DIR = Path("/home/cyx/FusionSF/MTL/PV_QuadTask_Project")
    EXP_ROOT = PROJECT_DIR / "experiments" / DATASET_NAME / "QuadTask_TaskAblation"
    EXP_ROOT.mkdir(parents=True, exist_ok=True)

    for f in [MTL_FILE, NWP_FILE, DAILY_LABEL_FILE]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing required file: {f}")

    for cfg in ABLATION_CONFIGS:
        model_name = f"A{cfg['id']:02d}_{cfg['name']}"
        ablation_dir = EXP_ROOT / f"ablation_{cfg['id']:02d}_{cfg['name']}"
        ablation_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "#" * 100)
        print(f"[ABLATION] {model_name} | use_dfi={cfg['use_dfi']} | use_weather={cfg['use_weather']} | use_ramp={cfg['use_ramp']}")
        print("#" * 100)

        for seed in SEEDS:
            print("\n" + "=" * 90)
            print(f"[RUN] Dataset={DATASET_NAME} | Model={model_name} | Seed={seed}")
            print("=" * 90)

            seed_everything(seed)

            run_dir = ablation_dir / f"seed_{seed}"
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

            model = PVTransformer_QuadTask_FixedFullAblation(
                nwp_dim=dataset.nwp_dim,
                num_classes=dataset.num_classes,
                d_model=D_MODEL,
                nhead=NHEAD,
                pred_len=PRED_LEN,
                use_dfi=cfg["use_dfi"],
                use_weather=cfg["use_weather"],
                use_ramp=cfg["use_ramp"],
                lr=LR,
                weight_decay=WEIGHT_DECAY,
            )

            config = {
                "dataset_name": DATASET_NAME,
                "model_name": model_name,
                "seed": seed,
                "ablation": cfg,
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
                    "use_dfi": cfg["use_dfi"],
                    "use_weather": cfg["use_weather"],
                    "use_ramp": cfg["use_ramp"],
                    "cafilm": "AllAux_Temporal_FixedInput_WithZeroMaskedDisabledTasks",
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

            print(f"Start training: {model_name}, seed={seed}")
            trainer.fit(model, train_loader, val_loader)

            best_model_path = my_checkpoint.best_model_path
            if best_model_path and os.path.exists(best_model_path):
                print(f"\nLoading best validation checkpoint: {best_model_path}")
                best_model = PVTransformer_QuadTask_FixedFullAblation.load_from_checkpoint(
                    checkpoint_path=best_model_path,
                    nwp_dim=dataset.nwp_dim,
                    num_classes=dataset.num_classes,
                    d_model=D_MODEL,
                    nhead=NHEAD,
                    pred_len=PRED_LEN,
                    use_dfi=cfg["use_dfi"],
                    use_weather=cfg["use_weather"],
                    use_ramp=cfg["use_ramp"],
                    lr=LR,
                    weight_decay=WEIGHT_DECAY,
                )
            else:
                print("\n[WARN] best.ckpt not found; using in-memory final model for testing.")
                best_model = model

            eval_device = "cuda" if torch.cuda.is_available() else "cpu"
            run_name = f"{DATASET_NAME}_{model_name}_seed{seed}"

            evaluate_model_comprehensive(
                best_model,
                test_loader,
                dataset,
                file_name=run_name,
                device=eval_device,
                output_dir=str(run_dir),
                dataset_name=DATASET_NAME,
                model_name=model_name,
                seed=seed,
                make_plots=(seed == SEEDS[0] and cfg["id"] == 8),
                compute_aux=True,
            )

            print(f"[DONE] {run_name}: outputs saved to {run_dir}")

    summarize_ablation_results(EXP_ROOT, DATASET_NAME, SEEDS)
