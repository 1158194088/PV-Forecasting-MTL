
"""
Optimizer ablation runner for QuadTask PV forecasting.

Ablation table:
    M1   CA-FiLM     Linear Weighting
    M2   CA-FiLM     PCGrad
    M3   CA-FiLM     CAGrad
    M4   CA-FiLM     IMTL
    M5   No-CAFiLM   Nash-MTL
    OURS CA-FiLM     Nash-MTL

Design principle
----------------
This file keeps the model architecture fixed around the latest improved version:
    CGC(Hetero) + RouteReg + task-aware auxiliary heads + optional Temporal CA-FiLM

Only the optimizer / feature-interaction switch changes:
    - M1-M4 and OURS use All-Aux Temporal CA-FiLM.
    - M5 disables CA-FiLM but keeps CGC(Hetero), RouteReg and Nash-MTL.
    - RouteReg is kept in all variants to isolate optimizer and CA-FiLM effects.

Run:
    python run_optimizer_ablation_quadtask_5seeds.py

Before full run, test with:
    SEEDS = [0]
    MAX_EPOCHS = 3
"""

import os
import json
import math
import random
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
from src.optimizer.NashOptimizer import NashMTLOptimizer

# CAGrad uses scipy.optimize. Install scipy if your environment does not have it.
import scipy.optimize as scipy_opt

from src.dataset.dataset_QuadTask import PVQuadTaskDataset
from src.evaluator.Pic_evaluator_detailed import evaluate_model_comprehensive


# =============================================================================
# 1. Reproducibility and checkpoint
# =============================================================================
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed, workers=True)


class HardcoreCheckpoint(pl.Callback):
    """Save one best checkpoint per current run directory."""

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
# 2. Multi-task optimizers
# =============================================================================
class _BaseMTLGradientOptimizer:
    """Shared utilities for four-task gradient manipulation optimizers."""

    def __init__(self, optimizer, shared_params, main_params, dfi_params, w_params, ramp_params):
        self._optim = optimizer
        self.shared_params = list(shared_params)
        self.main_params = list(main_params)
        self.dfi_params = list(dfi_params)
        self.w_params = list(w_params)
        self.ramp_params = list(ramp_params)

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        return self._optim.zero_grad()

    def step(self):
        return self._optim.step()

    @staticmethod
    def _get_grads(params: List[nn.Parameter]) -> List[torch.Tensor]:
        return [torch.zeros_like(p).to(p.device) if p.grad is None else p.grad.clone() for p in params]

    def _collect_task_grads(self, loss_main, loss_dfi, loss_w, loss_ramp):
        """Independently backpropagate four losses and collect shared/specific grads."""
        self.zero_grad()
        loss_main.backward(retain_graph=True)
        g_m_shared = self._get_grads(self.shared_params)
        g_m_main = self._get_grads(self.main_params)

        self.zero_grad()
        loss_dfi.backward(retain_graph=True)
        g_d_shared = self._get_grads(self.shared_params)
        g_d_dfi = self._get_grads(self.dfi_params)

        self.zero_grad()
        loss_w.backward(retain_graph=True)
        g_w_shared = self._get_grads(self.shared_params)
        g_w_w = self._get_grads(self.w_params)

        self.zero_grad()
        loss_ramp.backward(retain_graph=False)
        g_r_shared = self._get_grads(self.shared_params)
        g_r_ramp = self._get_grads(self.ramp_params)

        flat = {
            "main": torch.cat([g.flatten() for g in g_m_shared]),
            "dfi": torch.cat([g.flatten() for g in g_d_shared]),
            "weather": torch.cat([g.flatten() for g in g_w_shared]),
            "ramp": torch.cat([g.flatten() for g in g_r_shared]),
        }

        specific = {
            "main": g_m_main,
            "dfi": g_d_dfi,
            "weather": g_w_w,
            "ramp": g_r_ramp,
        }
        return flat, specific

    def _assign_grads(self, g_shared_final_flat: torch.Tensor, specific: Dict[str, List[torch.Tensor]]):
        """Write negotiated shared gradient and task-specific gradients back."""
        offset = 0
        g_shared_final = []
        for p in self.shared_params:
            numel = p.numel()
            g_shared_final.append(g_shared_final_flat[offset:offset + numel].view(p.shape))
            offset += numel

        self.zero_grad()
        for p, g in zip(self.shared_params, g_shared_final):
            p.grad = g
        for p, g in zip(self.main_params, specific["main"]):
            p.grad = g
        for p, g in zip(self.dfi_params, specific["dfi"]):
            p.grad = g
        for p, g in zip(self.w_params, specific["weather"]):
            p.grad = g
        for p, g in zip(self.ramp_params, specific["ramp"]):
            p.grad = g

    @staticmethod
    def _metrics(flat: Dict[str, torch.Tensor], g_shared_final_flat: torch.Tensor) -> Dict[str, float]:
        eps = 1e-8
        norm_m = torch.norm(flat["main"]) + eps
        norm_d = torch.norm(flat["dfi"]) + eps
        norm_w = torch.norm(flat["weather"]) + eps
        norm_r = torch.norm(flat["ramp"]) + eps

        u_m = flat["main"] / norm_m
        u_d = flat["dfi"] / norm_d
        u_w = flat["weather"] / norm_w
        u_r = flat["ramp"] / norm_r

        norm_final = torch.norm(g_shared_final_flat) + eps
        cos_combined = torch.dot(flat["main"], g_shared_final_flat) / (norm_m * norm_final + eps)

        def safe_item(x):
            v = float(x.item() if torch.is_tensor(x) else x)
            return v if math.isfinite(v) else 0.0

        return {
            "norm_main": safe_item(norm_m),
            "norm_aux_aligned": safe_item(norm_final),
            "cos_combined": safe_item(cos_combined),
            "cos_md": safe_item(torch.dot(u_m, u_d)),
            "cos_mw": safe_item(torch.dot(u_m, u_w)),
            "cos_mr": safe_item(torch.dot(u_m, u_r)),
            "cos_dw": safe_item(torch.dot(u_d, u_w)),
            "cos_dr": safe_item(torch.dot(u_d, u_r)),
            "cos_wr": safe_item(torch.dot(u_w, u_r)),
        }


class PCGradOptimizer(_BaseMTLGradientOptimizer):
    """PCGrad optimizer for shared parameters; task-specific groups use their own gradients."""

    def backward(self, loss_main, loss_dfi, loss_w, loss_ramp):
        flat, specific = self._collect_task_grads(loss_main, loss_dfi, loss_w, loss_ramp)
        grads = [flat["main"], flat["dfi"], flat["weather"], flat["ramp"]]
        pc_grads = []

        for i in range(4):
            g_i = grads[i].clone()
            other_indices = list(range(4))
            other_indices.remove(i)
            random.shuffle(other_indices)
            for j in other_indices:
                g_j = grads[j]
                dot = torch.dot(g_i, g_j)
                if dot < 0:
                    g_i = g_i - (dot / (torch.dot(g_j, g_j) + 1e-8)) * g_j
            pc_grads.append(g_i)

        # Keep the user's original PCGrad spirit: sum projected gradients.
        g_shared_final_flat = sum(pc_grads)
        self._assign_grads(g_shared_final_flat, specific)
        return self._metrics(flat, g_shared_final_flat)


class CAGradOptimizer(_BaseMTLGradientOptimizer):
    """CAGrad optimizer for shared parameters."""

    def __init__(self, optimizer, shared_params, main_params, dfi_params, w_params, ramp_params, c: float = 0.4):
        super().__init__(optimizer, shared_params, main_params, dfi_params, w_params, ramp_params)
        self.c = c

    def backward(self, loss_main, loss_dfi, loss_w, loss_ramp):
        flat, specific = self._collect_task_grads(loss_main, loss_dfi, loss_w, loss_ramp)

        G = torch.stack([flat["main"], flat["dfi"], flat["weather"], flat["ramp"]], dim=1)  # [dim, 4]
        g0 = torch.mean(G, dim=1)

        GG_np = torch.matmul(G.T, G).detach().cpu().numpy()
        g0G_np = torch.matmul(g0.unsqueeze(0), G).squeeze().detach().cpu().numpy()

        def obj_fn(w):
            return self.c * np.sqrt(np.dot(w, np.dot(GG_np, w)) + 1e-8) + np.dot(g0G_np, w)

        bounds = [(0.0, 1.0) for _ in range(4)]
        cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        try:
            res = scipy_opt.minimize(
                obj_fn,
                np.ones(4) / 4.0,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter": 50, "ftol": 1e-8, "disp": False},
            )
            if not res.success or np.any(~np.isfinite(res.x)):
                w_np = np.ones(4) / 4.0
            else:
                w_np = res.x
        except Exception:
            w_np = np.ones(4) / 4.0

        w_star = torch.tensor(w_np, dtype=G.dtype, device=G.device)
        gw = torch.matmul(G, w_star)
        gw_norm = torch.norm(gw) + 1e-8
        g0_norm = torch.norm(g0) + 1e-8

        g_shared_final_flat = g0 + self.c * g0_norm * gw / gw_norm
        self._assign_grads(g_shared_final_flat, specific)
        return self._metrics(flat, g_shared_final_flat)


class IMTLOptimizer(_BaseMTLGradientOptimizer):
    """IMTL-style equal-projection direction optimizer."""

    def backward(self, loss_main, loss_dfi, loss_w, loss_ramp):
        flat, specific = self._collect_task_grads(loss_main, loss_dfi, loss_w, loss_ramp)

        G = torch.stack([flat["main"], flat["dfi"], flat["weather"], flat["ramp"]], dim=1)
        D = torch.matmul(G.T, G) + 1e-4 * torch.eye(4, device=G.device)

        try:
            ones = torch.ones(4, device=G.device, dtype=G.dtype)
            alpha = torch.linalg.solve(D, ones)
            alpha_sum = torch.sum(alpha)
            if torch.abs(alpha_sum) < 1e-8 or not torch.isfinite(alpha_sum):
                alpha = torch.ones(4, device=G.device, dtype=G.dtype) / 4.0
            else:
                alpha = alpha / alpha_sum
        except RuntimeError:
            alpha = torch.ones(4, device=G.device, dtype=G.dtype) / 4.0

        # Additional finite guard.
        if not torch.isfinite(alpha).all():
            alpha = torch.ones(4, device=G.device, dtype=G.dtype) / 4.0

        g_shared_final_flat = torch.matmul(G, alpha)
        self._assign_grads(g_shared_final_flat, specific)
        return self._metrics(flat, g_shared_final_flat)


# class NashMTLOptimizer(_BaseMTLGradientOptimizer):
#     """Nash-MTL optimizer matching the user's four-task Nash implementation."""

#     def backward(self, loss_main, loss_dfi, loss_w, loss_ramp):
#         flat, specific = self._collect_task_grads(loss_main, loss_dfi, loss_w, loss_ramp)
#         eps = 1e-8

#         norms = {
#             "main": torch.norm(flat["main"]) + eps,
#             "dfi": torch.norm(flat["dfi"]) + eps,
#             "weather": torch.norm(flat["weather"]) + eps,
#             "ramp": torch.norm(flat["ramp"]) + eps,
#         }
#         unit = {name: flat[name] / norms[name] for name in norms.keys()}

#         cos_md = torch.dot(unit["main"], unit["dfi"])
#         cos_mw = torch.dot(unit["main"], unit["weather"])
#         cos_mr = torch.dot(unit["main"], unit["ramp"])
#         cos_dw = torch.dot(unit["dfi"], unit["weather"])
#         cos_dr = torch.dot(unit["dfi"], unit["ramp"])
#         cos_wr = torch.dot(unit["weather"], unit["ramp"])

#         C = torch.stack([
#             torch.stack([torch.tensor(1.0, device=cos_md.device), cos_md, cos_mw, cos_mr]),
#             torch.stack([cos_md, torch.tensor(1.0, device=cos_md.device), cos_dw, cos_dr]),
#             torch.stack([cos_mw, cos_dw, torch.tensor(1.0, device=cos_md.device), cos_wr]),
#             torch.stack([cos_mr, cos_dr, cos_wr, torch.tensor(1.0, device=cos_md.device)]),
#         ])
#         C = C + torch.eye(4, device=C.device) * 1e-4

#         alpha = torch.ones(4, device=C.device)
#         for _ in range(10):
#             grad_alpha = torch.matmul(C, alpha) - 1.0 / (alpha + 1e-8)
#             hessian = C + torch.diag(1.0 / (alpha ** 2 + 1e-8))
#             try:
#                 delta = torch.linalg.solve(hessian, grad_alpha)
#             except RuntimeError:
#                 delta = grad_alpha * 0.1
#             alpha = torch.clamp(alpha - delta, min=1e-4)

#         alpha_norm = alpha / torch.sum(alpha)
#         g_shared_direction = (
#             alpha_norm[0] * unit["main"] +
#             alpha_norm[1] * unit["dfi"] +
#             alpha_norm[2] * unit["weather"] +
#             alpha_norm[3] * unit["ramp"]
#         )
#         avg_norm = (norms["main"] + norms["dfi"] + norms["weather"] + norms["ramp"]) / 4.0
#         g_shared_final_flat = g_shared_direction * avg_norm

#         self._assign_grads(g_shared_final_flat, specific)
#         return self._metrics(flat, g_shared_final_flat)


# =============================================================================
# 3. Model: CGC(Hetero) + RouteReg + optional All-Aux Temporal CA-FiLM
# =============================================================================
class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.value_embedding = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.value_embedding(x))


class PVTransformer_QuadTask_OptimizerAblation(pl.LightningModule):
    """
    Fixed model for optimizer ablation.

    Fixed components:
        - Backbone
        - CGC(Hetero)
        - RouteReg
        - Task-aware heads

    Controlled components:
        - use_cafilm=True/False
        - optimizer_type: linear / pcgrad / cagrad / imtl / nash
    """

    def __init__(
        self,
        nwp_dim,
        num_classes,
        d_model=128,
        nhead=4,
        patch_len=8,
        stride=4,
        pred_len=24,
        use_cafilm=True,
        optimizer_type="nash",
        lr=1e-4,
        weight_decay=1e-2,
        route_threshold=0.40,
        route_lambda=3.0,
        cagrad_c=0.4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.nwp_dim = nwp_dim
        self.num_classes = num_classes
        self.d_model = d_model
        self.pred_len = pred_len
        self.use_cafilm = bool(use_cafilm)
        self.optimizer_type = str(optimizer_type).lower()
        self.lr = lr
        self.weight_decay = weight_decay
        self.route_threshold = route_threshold
        self.route_lambda = route_lambda
        self.cagrad_c = cagrad_c

        # Backbone.
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

        # CGC(Hetero): match the latest improved model.
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

        # Task-aware heads.
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

        # Optional All-Aux Temporal CA-FiLM.
        if self.use_cafilm:
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
        else:
            self.phys_projector = None
            self.film_generator = None

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

    def forward(self, ts, nwp):
        p_ts = self.ts_patcher(ts)
        p_nwp = self.nwp_patcher(nwp)
        p_ts = p_ts + self.pos_enc_ts[:, :p_ts.size(1), :]
        p_nwp = p_nwp + self.pos_enc_nwp[:, :p_nwp.size(1), :]

        attn_out, _ = self.cross_attention(query=p_nwp, key=p_ts, value=p_ts)
        fused_feat = self.fusion_norm(p_nwp + attn_out)
        bottom_out = self.shared_bottom(fused_feat)

        out_shared = self.expert_shared(bottom_out)
        gate_input = torch.mean(bottom_out, dim=1)

        out_main = self.expert_main(bottom_out)
        out_dfi = self.expert_aux_dfi(bottom_out)
        out_w = self.expert_aux_w(bottom_out)
        out_ramp = self.expert_aux_ramp(bottom_out)

        fused_main, w_main = self._gate_fuse(self.gate_main, gate_input, out_shared, out_main)
        fused_dfi, w_dfi = self._gate_fuse(self.gate_aux_dfi, gate_input, out_shared, out_dfi)
        fused_w, w_w = self._gate_fuse(self.gate_aux_w, gate_input, out_shared, out_w)
        fused_ramp, w_ramp = self._gate_fuse(self.gate_aux_ramp, gate_input, out_shared, out_ramp)

        dfi_pred = self.head_dfi(fused_dfi).squeeze(-1)
        weather_context = torch.mean(fused_w, dim=1)
        w_logits = self.head_weather(weather_context)
        ramp_logits = self.head_ramp(fused_ramp)

        if self.use_cafilm:
            with torch.no_grad():
                dfi_context = dfi_pred.detach().unsqueeze(-1)
                weather_prob = F.softmax(w_logits.detach(), dim=-1)
                weather_context_step = weather_prob.unsqueeze(1).expand(-1, self.pred_len, -1)
                ramp_context = F.softmax(ramp_logits.detach(), dim=-1)

                phys_context_step = torch.cat(
                    [dfi_context, weather_context_step, ramp_context],
                    dim=-1,
                )

            phys_embedding_step = self.phys_projector(phys_context_step)
            combined_info = torch.cat([fused_main, phys_embedding_step], dim=-1)
            film_params = self.film_generator(combined_info)
            gamma, beta = torch.chunk(film_params, 2, dim=-1)
            fused_main_for_power = fused_main * (1.0 + gamma) + beta
        else:
            fused_main_for_power = fused_main

        p_pred = self.head_power(fused_main_for_power)

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
            list(self.head_power.parameters())
        )
        if self.use_cafilm:
            main_params += list(self.phys_projector.parameters()) + list(self.film_generator.parameters())

        dfi_params = (
            list(self.expert_aux_dfi.parameters()) +
            list(self.gate_aux_dfi.parameters()) +
            list(self.head_dfi.parameters())
        )
        w_params = (
            list(self.expert_aux_w.parameters()) +
            list(self.gate_aux_w.parameters()) +
            list(self.head_weather.parameters())
        )
        ramp_params = (
            list(self.expert_aux_ramp.parameters()) +
            list(self.gate_aux_ramp.parameters()) +
            list(self.head_ramp.parameters())
        )

        opt_type = self.optimizer_type
        if opt_type == "linear":
            self.mtl_optimizer = None
        elif opt_type == "pcgrad":
            self.mtl_optimizer = PCGradOptimizer(base_opt, shared_params, main_params, dfi_params, w_params, ramp_params)
        elif opt_type == "cagrad":
            self.mtl_optimizer = CAGradOptimizer(base_opt, shared_params, main_params, dfi_params, w_params, ramp_params, c=self.cagrad_c)
        elif opt_type == "imtl":
            self.mtl_optimizer = IMTLOptimizer(base_opt, shared_params, main_params, dfi_params, w_params, ramp_params)
        elif opt_type == "nash":
            self.mtl_optimizer = NashMTLOptimizer(base_opt, shared_params, main_params, dfi_params, w_params, ramp_params)
        else:
            raise ValueError(f"Unknown optimizer_type: {self.optimizer_type}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            base_opt,
            T_0=15,
            T_mult=2,
            eta_min=1e-6,
        )
        return [base_opt], [scheduler]

    def _compute_losses(self, batch):
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

        # RouteReg is fixed in all optimizer ablation variants.
        penalty_w = F.relu(w_w[:, 0] - self.route_threshold).mean() * self.route_lambda
        penalty_r = F.relu(w_r[:, 0] - self.route_threshold).mean() * self.route_lambda
        loss_w = loss_w_raw + penalty_w
        loss_ramp = loss_ramp_raw + penalty_r

        outputs = {
            "p_pred": p_pred,
            "dfi_pred": dfi_pred,
            "w_logits": w_logits,
            "ramp_logits": ramp_logits,
            "p_sh_dict": p_sh_dict,
            "w_w": w_w,
            "w_r": w_r,
        }
        losses = {
            "loss_p": loss_p,
            "loss_dfi": loss_dfi,
            "loss_w": loss_w,
            "loss_w_raw": loss_w_raw,
            "loss_ramp": loss_ramp,
            "loss_ramp_raw": loss_ramp_raw,
            "penalty_w": penalty_w,
            "penalty_r": penalty_r,
        }
        return outputs, losses

    def training_step(self, batch, batch_idx):
        outputs, losses = self._compute_losses(batch)

        # Finite check before backward.
        for name, value in losses.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise RuntimeError(f"[NaN/Inf] {name} is invalid: {value}")

        if self.optimizer_type == "linear":
            opt = self.optimizers()
            opt.zero_grad()
            total_loss = losses["loss_p"] + losses["loss_dfi"] + losses["loss_w"] + losses["loss_ramp"]
            self.manual_backward(total_loss)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            opt.step()
            metrics = {
                "cos_combined": 0.0,
                "cos_md": 0.0,
                "cos_mw": 0.0,
                "cos_mr": 0.0,
                "cos_dw": 0.0,
                "cos_dr": 0.0,
                "cos_wr": 0.0,
            }
        else:
            metrics = self.mtl_optimizer.backward(
                losses["loss_p"],
                losses["loss_dfi"],
                losses["loss_w"],
                losses["loss_ramp"],
            )
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.mtl_optimizer.step()

        if self.trainer.is_last_batch:
            self.lr_schedulers().step()

        p_sh_dict = outputs["p_sh_dict"]
        self.log("train_loss_p", losses["loss_p"], prog_bar=True)
        self.log("train_loss_total_linear_form",
                 losses["loss_p"] + losses["loss_dfi"] + losses["loss_w"] + losses["loss_ramp"],
                 prog_bar=False)
        self.log("1_Loss/Aux_DFI", losses["loss_dfi"], prog_bar=False)
        self.log("1_Loss/Aux_Weather", losses["loss_w"], prog_bar=False)
        self.log("1_Loss/Aux_Ramp", losses["loss_ramp"], prog_bar=False)
        self.log("1_Loss/Penalty_Weather", losses["penalty_w"], prog_bar=False)
        self.log("1_Loss/Penalty_Ramp", losses["penalty_r"], prog_bar=False)

        self.log("6_Gate/Main_use_Shared", p_sh_dict["m"], prog_bar=False)
        self.log("6_Gate/DFI_use_Shared", p_sh_dict["d"], prog_bar=False)
        self.log("6_Gate/Weather_use_Shared", p_sh_dict["w"], prog_bar=False)
        self.log("6_Gate/Ramp_use_Shared", p_sh_dict["r"], prog_bar=False)

        self.log("3_MTL/CosTheta_Overall_Combined", float(metrics.get("cos_combined", 0.0)), prog_bar=True)
        self.log("7_Gradient_Angle/Cos_Main_vs_DFI", float(metrics.get("cos_md", 0.0)), prog_bar=False)
        self.log("7_Gradient_Angle/Cos_Main_vs_Weather", float(metrics.get("cos_mw", 0.0)), prog_bar=False)
        self.log("7_Gradient_Angle/Cos_Main_vs_Ramp", float(metrics.get("cos_mr", 0.0)), prog_bar=False)

        return losses["loss_p"]

    def validation_step(self, batch, batch_idx):
        p_pred, _, _, _, _, _, _ = self(batch["ts_input"], batch["nwp_input"])
        mae = F.l1_loss(p_pred, batch["target_p"])
        self.log("val_mae", mae, on_epoch=True, prog_bar=True)
        return mae


# =============================================================================
# 4. Experiment configs and summarization
# =============================================================================
OPTIMIZER_ABLATION_CONFIGS = [
    {
        "id": "M1",
        "name": "CAFiLM_LinearWeighting",
        "feature_interaction": "CA-FiLM",
        "optimizer_name": "Linear Weighting",
        "use_cafilm": True,
        "optimizer_type": "linear",
    },
    {
        "id": "M2",
        "name": "CAFiLM_PCGrad",
        "feature_interaction": "CA-FiLM",
        "optimizer_name": "PCGrad",
        "use_cafilm": True,
        "optimizer_type": "pcgrad",
    },
    {
        "id": "M3",
        "name": "CAFiLM_CAGrad",
        "feature_interaction": "CA-FiLM",
        "optimizer_name": "CAGrad",
        "use_cafilm": True,
        "optimizer_type": "cagrad",
    },
    {
        "id": "M4",
        "name": "CAFiLM_IMTL",
        "feature_interaction": "CA-FiLM",
        "optimizer_name": "IMTL",
        "use_cafilm": True,
        "optimizer_type": "imtl",
    },
    {
        "id": "M5",
        "name": "NoInteraction_NashMTL",
        "feature_interaction": "No interaction",
        "optimizer_name": "Nash-MTL",
        "use_cafilm": False,
        "optimizer_type": "nash",
    },
    {
        "id": "OURS",
        "name": "CAFiLM_NashMTL",
        "feature_interaction": "CA-FiLM",
        "optimizer_name": "Nash-MTL",
        "use_cafilm": True,
        "optimizer_type": "nash",
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


def summarize_optimizer_ablation_results(exp_root: Path, dataset_name: str, seeds: List[int]) -> None:
    summary_dir = exp_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    power_all = []
    weather_all = []
    aux_all = []
    stress_all = []

    for cfg in OPTIMIZER_ABLATION_CONFIGS:
        exp_dir = exp_root / f"{cfg['id']}_{cfg['name']}"
        model_name = f"{cfg['id']}_{cfg['name']}"
        for seed in seeds:
            run_dir = exp_dir / f"seed_{seed}"
            file_name = f"{dataset_name}_{model_name}_seed{seed}"
            metrics_dir = run_dir / "metrics"
            keys = {
                "experiment_id": cfg["id"],
                "experiment_name": cfg["name"],
                "feature_interaction": cfg["feature_interaction"],
                "optimizer": cfg["optimizer_name"],
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
            power_df.groupby(["experiment_id", "experiment_name", "feature_interaction", "optimizer", "metric"], as_index=False)
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
            index=["experiment_id", "experiment_name", "feature_interaction", "optimizer"],
            columns="metric",
            values="mean_std",
            aggfunc="first",
        ).reset_index()
        order = {cfg["id"]: i for i, cfg in enumerate(OPTIMIZER_ABLATION_CONFIGS)}
        compact["order"] = compact["experiment_id"].map(order)
        compact = compact.sort_values("order").drop(columns=["order"])
        compact.to_csv(summary_dir / "optimizer_ablation_power_summary.csv", index=False, encoding="utf-8-sig")

    if not weather_df.empty:
        weather_df.to_csv(summary_dir / "weather_metrics_all_seeds_long.csv", index=False, encoding="utf-8-sig")
        weather_summary = (
            weather_df.groupby(["experiment_id", "experiment_name", "weather_id", "weather_name", "metric"], as_index=False)
            .agg(mean=("value", "mean"), std=("value", "std"), seeds=("seed", "nunique"))
        )
        weather_summary.to_csv(summary_dir / "weather_metrics_mean_std.csv", index=False, encoding="utf-8-sig")

    if not aux_df.empty:
        aux_df.to_csv(summary_dir / "auxiliary_metrics_all_seeds_long.csv", index=False, encoding="utf-8-sig")
        aux_summary = (
            aux_df.groupby(["experiment_id", "experiment_name", "task", "metric"], as_index=False)
            .agg(mean=("value", "mean"), std=("value", "std"), seeds=("seed", "nunique"))
        )
        aux_summary.to_csv(summary_dir / "auxiliary_metrics_mean_std.csv", index=False, encoding="utf-8-sig")

    if not stress_df.empty:
        stress_df.to_csv(summary_dir / "stress_metrics_all_seeds_long.csv", index=False, encoding="utf-8-sig")
        stress_summary = (
            stress_df.groupby(["experiment_id", "experiment_name", "scenario", "metric"], as_index=False)
            .agg(mean=("value", "mean"), std=("value", "std"), seeds=("seed", "nunique"))
        )
        stress_summary.to_csv(summary_dir / "stress_metrics_mean_std.csv", index=False, encoding="utf-8-sig")

    print(f"\n[SUMMARY] Optimizer ablation summaries saved to: {summary_dir}")


# =============================================================================
# 5. Main runner
# =============================================================================
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # CONFIG: edit these paths/settings for your machine.
    # ------------------------------------------------------------------
    DATASET_NAME = "private"
    SEEDS = [0]

    BASE_DIR = "/home/cyx/FusionSF/MTL/PV_QuadTask_Project/data"
    MTL_FILE = os.path.join(BASE_DIR, "MTL_Dataset_pro.csv")
    NWP_FILE = os.path.join(BASE_DIR, "nwp_filtered_clean_1H_new.csv")
    DAILY_LABEL_FILE = os.path.join(BASE_DIR, "历史天级别-处理.csv")

    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.1
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    MAX_EPOCHS = 3
    PATIENCE = 20

    D_MODEL = 128
    NHEAD = 4
    PRED_LEN = 24
    LR = 1e-4
    WEIGHT_DECAY = 1e-2
    CAGRAD_C = 0.4

    PROJECT_DIR = Path("/home/cyx/FusionSF/MTL/PV_QuadTask_Project")
    EXP_ROOT = PROJECT_DIR / "experiments" / DATASET_NAME / "Optimizer_Ablation"
    EXP_ROOT.mkdir(parents=True, exist_ok=True)

    for f in [MTL_FILE, NWP_FILE, DAILY_LABEL_FILE]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing required file: {f}")

    for cfg in OPTIMIZER_ABLATION_CONFIGS:
        model_name = f"{cfg['id']}_{cfg['name']}"
        exp_dir = EXP_ROOT / model_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "#" * 100)
        print(f"[OPTIMIZER ABLATION] {model_name} | interaction={cfg['feature_interaction']} | optimizer={cfg['optimizer_name']}")
        print("#" * 100)

        for seed in SEEDS:
            print("\n" + "=" * 90)
            print(f"[RUN] Dataset={DATASET_NAME} | Model={model_name} | Seed={seed}")
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

            model = PVTransformer_QuadTask_OptimizerAblation(
                nwp_dim=dataset.nwp_dim,
                num_classes=dataset.num_classes,
                d_model=D_MODEL,
                nhead=NHEAD,
                pred_len=PRED_LEN,
                use_cafilm=cfg["use_cafilm"],
                optimizer_type=cfg["optimizer_type"],
                lr=LR,
                weight_decay=WEIGHT_DECAY,
                cagrad_c=CAGRAD_C,
            )

            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

            config = {
                "dataset_name": DATASET_NAME,
                "model_name": model_name,
                "seed": seed,
                "optimizer_ablation": cfg,
                "mtl_file": MTL_FILE,
                "nwp_file": NWP_FILE,
                "daily_label_file": DAILY_LABEL_FILE,
                "train_ratio": TRAIN_RATIO,
                "val_ratio": VAL_RATIO,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "num_trainable_parameters": param_count,
                "model_params": {
                    "nwp_dim": dataset.nwp_dim,
                    "num_classes": dataset.num_classes,
                    "d_model": D_MODEL,
                    "nhead": NHEAD,
                    "pred_len": PRED_LEN,
                    "use_cafilm": cfg["use_cafilm"],
                    "optimizer_type": cfg["optimizer_type"],
                    "lr": LR,
                    "weight_decay": WEIGHT_DECAY,
                    "cagrad_c": CAGRAD_C,
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

            print(f"Start training: {model_name}, seed={seed}, params={param_count:,}")
            trainer.fit(model, train_loader, val_loader)

            best_model_path = my_checkpoint.best_model_path
            if best_model_path and os.path.exists(best_model_path):
                print(f"\nLoading best validation checkpoint: {best_model_path}")
                best_model = PVTransformer_QuadTask_OptimizerAblation.load_from_checkpoint(
                    checkpoint_path=best_model_path,
                    nwp_dim=dataset.nwp_dim,
                    num_classes=dataset.num_classes,
                    d_model=D_MODEL,
                    nhead=NHEAD,
                    pred_len=PRED_LEN,
                    use_cafilm=cfg["use_cafilm"],
                    optimizer_type=cfg["optimizer_type"],
                    lr=LR,
                    weight_decay=WEIGHT_DECAY,
                    cagrad_c=CAGRAD_C,
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
                make_plots=(seed == SEEDS[0] and cfg["id"] == "OURS"),
                compute_aux=True,
            )

            print(f"[DONE] {run_name}: outputs saved to {run_dir}")

    summarize_optimizer_ablation_results(EXP_ROOT, DATASET_NAME, SEEDS)
