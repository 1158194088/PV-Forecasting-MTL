import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.optimizer.NashOptimizer import NashMTLOptimizer


# ==============================================================================
# 1. Series Decomposition
# ==============================================================================
class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# ==============================================================================
# 2. QuadTask Temporal CA-FiLM Module for DLinear Backbone
# ==============================================================================
class QuadTaskTemporalCAFiLMHead(nn.Module):
    """
    Backbone-agnostic QuadTask module adapted for DLinear.

    输入为 DLinear 原始未来预测窗口表征 bottom_input: [B, P, input_dim]。
    该模块只负责：
      1) 将 DLinear 的线性未来表征投影到 d_model；
      2) CGC 异构共享-专属专家解耦；
      3) DFI / Weather / Ramp 辅助预测；
      4) All-Aux Temporal CA-FiLM 逐小时调制；
      5) 输出主任务 residual correction，与 DLinear base forecast 相加。
    """
    def __init__(self, input_dim, num_classes, d_model=128, pred_len=24, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.d_model = d_model
        self.pred_len = pred_len

        # 将 DLinear 原始未来表征映射到统一隐藏维度。
        self.feature_projector = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ------------------------------------------------------------------
        # CGC heterogeneous shared/private experts
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
        self.expert_aux_ramp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model * 2, d_model),
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
        # Task-shape-aware auxiliary heads
        # ------------------------------------------------------------------
        # DFI: step-wise regression [B, P, D] -> [B, P]
        self.head_dfi = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1),
        )

        # Weather: daily classification via temporal pooling [B, D] -> [B, C]
        self.head_weather = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes),
        )

        # Ramp: step-wise classification [B, P, D] -> [B, P, 5]
        self.head_ramp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 5),
        )

        # ------------------------------------------------------------------
        # All-Aux Temporal CA-FiLM
        # 每个小时的物理上下文 = DFI(1) + Weather(num_classes) + Ramp(5)
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
        # 零初始化保护：训练初期 Temporal CA-FiLM 等价于恒等映射。
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)

        # 主任务 residual prediction head：逐小时输出 residual correction。
        self.head_power_residual = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1),
        )

    @staticmethod
    def _gate_fuse(gate, gate_input, shared_feat, task_feat):
        weights = F.softmax(gate(gate_input), dim=-1)  # [B, 2]
        fused = weights[:, 0:1, None] * shared_feat + weights[:, 1:2, None] * task_feat
        return fused, weights

    def forward(self, bottom_input):
        # bottom_input: [B, P, input_dim]
        B = bottom_input.size(0)
        bottom_out = self.feature_projector(bottom_input)  # [B, P, D]

        # CGC expert routing
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

        # Auxiliary predictions
        dfi_pred = self.head_dfi(fused_dfi).squeeze(-1)       # [B, P]
        weather_context = torch.mean(fused_w, dim=1)          # [B, D]
        w_logits = self.head_weather(weather_context)         # [B, C]
        ramp_logits = self.head_ramp(fused_ramp)              # [B, P, 5]

        # Temporal CA-FiLM with auxiliary predictions only, no true labels.
        with torch.no_grad():
            dfi_context = dfi_pred.detach().unsqueeze(-1)     # [B, P, 1]

            weather_prob = F.softmax(w_logits.detach(), dim=-1)
            weather_context_step = weather_prob.unsqueeze(1).expand(-1, self.pred_len, -1)

            ramp_context = F.softmax(ramp_logits.detach(), dim=-1)  # [B, P, 5]

            phys_context_step = torch.cat(
                [dfi_context, weather_context_step, ramp_context],
                dim=-1,
            )  # [B, P, 1 + C + 5]

        phys_embedding_step = self.phys_projector(phys_context_step)  # [B, P, D]
        combined_info = torch.cat([fused_main, phys_embedding_step], dim=-1)
        film_params = self.film_generator(combined_info)              # [B, P, 2D]
        gamma, beta = torch.chunk(film_params, 2, dim=-1)

        fused_main_refined = fused_main * (1.0 + gamma) + beta
        p_residual = self.head_power_residual(fused_main_refined).squeeze(-1)  # [B, P]

        p_sh_dict = {
            'm': w_main[:, 0].mean().detach().item(),
            'd': w_dfi[:, 0].mean().detach().item(),
            'w': w_w[:, 0].mean().detach().item(),
            'r': w_ramp[:, 0].mean().detach().item(),
        }

        return p_residual, dfi_pred, w_logits, ramp_logits, p_sh_dict, w_w, w_ramp


# ==============================================================================
# 3. DLinear + QuadTask module 泛化验证版
# ==============================================================================
class PVTransformer_QuadTask_Advanced(pl.LightningModule):
    """
    DLinear backbone + QuadTask module 泛化验证版。

    公平性原则：
      1. 保留 DLinear 原始 series decomposition；
      2. 保留历史 seasonal/trend 线性外推；
      3. 保留未来 NWP 的 time-distributed linear bias；
      4. 仅在 DLinear 产生的未来窗口表征之后接入 QuadTask 模块；
      5. 主任务最终预测 = DLinear base forecast + QuadTask residual correction。
    """
    def __init__(self, nwp_dim, num_classes, seq_len=48, pred_len=24,
                 d_model=128, dropout=0.2):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.num_classes = num_classes
        self.nwp_dim = nwp_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.unified_dim = 1 + nwp_dim + 6

        # Original DLinear components: unchanged.
        self.decomp = series_decomp(kernel_size=25)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        nwp_feature_count = nwp_dim + 6
        self.Linear_NWP = nn.Linear(nwp_feature_count, 1)

        # DLinear future representation for QuadTask module:
        # pred_history: [B, P, unified_dim]
        # nwp_features: [B, P, nwp_dim + 6]
        # p_base: [B, P, 1]
        quadtask_input_dim = self.unified_dim + nwp_feature_count + 1
        self.quadtask_head = QuadTaskTemporalCAFiLMHead(
            input_dim=quadtask_input_dim,
            num_classes=num_classes,
            d_model=d_model,
            pred_len=pred_len,
            dropout=dropout,
        )

    def _dlinear_backbone(self, ts, nwp):
        B, seq_len, _ = ts.shape

        # 1. Heterogeneous data alignment: same as original baseline.
        ts_unified = torch.cat(
            [
                ts[:, :, 0:1],
                torch.zeros(B, seq_len, self.nwp_dim, device=ts.device, dtype=ts.dtype),
                ts[:, :, 1:7],
            ],
            dim=-1,
        )

        nwp_unified = torch.cat(
            [
                torch.zeros(B, self.pred_len, 1, device=nwp.device, dtype=nwp.dtype),
                nwp[:, :, 0:self.nwp_dim],
                nwp[:, :, self.nwp_dim:],
            ],
            dim=-1,
        )

        # 2. Series decomposition and original DLinear temporal extrapolation.
        seasonal_init, trend_init = self.decomp(ts_unified)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        seasonal_output = self.Linear_Seasonal(seasonal_init)  # [B, M, P]
        trend_output = self.Linear_Trend(trend_init)           # [B, M, P]
        pred_history = (seasonal_output + trend_output).permute(0, 2, 1)  # [B, P, M]

        # 3. Original base forecast = historical DLinear extrapolation + NWP linear bias.
        p_hist_pred = pred_history[:, :, 0]  # [B, P]
        nwp_features = nwp_unified[:, :, 1:]  # [B, P, nwp_dim + 6]
        p_nwp_contribution = self.Linear_NWP(nwp_features).squeeze(-1)  # [B, P]
        p_base = p_hist_pred + p_nwp_contribution  # [B, P]

        return p_base, pred_history, nwp_features

    def forward(self, ts, nwp):
        # Original DLinear base forecast and future representations.
        p_base, pred_history, nwp_features = self._dlinear_backbone(ts, nwp)

        # Build future-window representation without changing original DLinear computation.
        bottom_input = torch.cat(
            [
                pred_history,
                nwp_features,
                p_base.unsqueeze(-1),
            ],
            dim=-1,
        )  # [B, P, unified_dim + nwp_feature_count + 1]

        p_residual, dfi_pred, w_logits, ramp_logits, p_sh_dict, w_w, w_r = self.quadtask_head(bottom_input)

        # Residual correction preserves original DLinear prediction path.
        p_pred = p_base + p_residual

        return p_pred, dfi_pred, w_logits, ramp_logits, p_sh_dict, w_w, w_r

    def configure_optimizers(self):
        base_opt = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)

        # Shared parameters: original DLinear backbone + QuadTask feature projector + shared expert.
        shared_params = (
            list(self.Linear_Seasonal.parameters()) +
            list(self.Linear_Trend.parameters()) +
            list(self.Linear_NWP.parameters()) +
            list(self.quadtask_head.feature_projector.parameters()) +
            list(self.quadtask_head.expert_shared.parameters())
        )

        # Main task parameters: main expert, main gate, CA-FiLM, residual power head.
        main_params = (
            list(self.quadtask_head.expert_main.parameters()) +
            list(self.quadtask_head.gate_main.parameters()) +
            list(self.quadtask_head.phys_projector.parameters()) +
            list(self.quadtask_head.film_generator.parameters()) +
            list(self.quadtask_head.head_power_residual.parameters())
        )

        dfi_params = (
            list(self.quadtask_head.expert_aux_dfi.parameters()) +
            list(self.quadtask_head.gate_aux_dfi.parameters()) +
            list(self.quadtask_head.head_dfi.parameters())
        )

        w_params = (
            list(self.quadtask_head.expert_aux_w.parameters()) +
            list(self.quadtask_head.gate_aux_w.parameters()) +
            list(self.quadtask_head.head_weather.parameters())
        )

        ramp_params = (
            list(self.quadtask_head.expert_aux_ramp.parameters()) +
            list(self.quadtask_head.gate_aux_ramp.parameters()) +
            list(self.quadtask_head.head_ramp.parameters())
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
            batch['ts_input'],
            batch['nwp_input'],
        )

        loss_p = F.huber_loss(p_pred, batch['target_p'], delta=0.5)
        loss_dfi = F.mse_loss(dfi_pred, batch['target_dfi'])
        loss_w = F.cross_entropy(w_logits, batch['daily_weather'])
        loss_ramp = F.cross_entropy(
            ramp_logits.reshape(-1, 5),
            batch['target_ramp'].reshape(-1),
        )

        # RouteReg: Weather / Ramp should not overuse shared expert.
        penalty_w = F.relu(w_w[:, 0] - 0.40).mean() * 3.0
        penalty_r = F.relu(w_r[:, 0] - 0.40).mean() * 3.0
        loss_w = loss_w + penalty_w
        loss_ramp = loss_ramp + penalty_r

        metrics = opt.backward(loss_p, loss_dfi, loss_w, loss_ramp)

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        opt.step()
        if self.trainer.is_last_batch:
            self.lr_schedulers().step()

        self.log('train_loss_p', loss_p, prog_bar=True)
        self.log('1_Loss/Aux_DFI_Raw', loss_dfi, prog_bar=False)
        self.log('1_Loss/Aux_Weather_Raw', loss_w, prog_bar=False)
        self.log('1_Loss/Aux_Ramp_Raw', loss_ramp, prog_bar=False)

        self.log('6_Gate/Main_use_Shared', p_sh_dict['m'], prog_bar=False)
        self.log('6_Gate/DFI_use_Shared', p_sh_dict['d'], prog_bar=False)
        self.log('6_Gate/Weather_use_Shared', p_sh_dict['w'], prog_bar=False)
        self.log('6_Gate/Ramp_use_Shared', p_sh_dict['r'], prog_bar=False)

        self.log('3_Nash/CosTheta_Overall_Combined', metrics['cos_combined'], prog_bar=True)
        self.log('7_Gradient_Angle/Cos_Main_vs_DFI', metrics['cos_md'], prog_bar=False)
        self.log('7_Gradient_Angle/Cos_Main_vs_Weather', metrics['cos_mw'], prog_bar=False)
        self.log('7_Gradient_Angle/Cos_Main_vs_Ramp', metrics['cos_mr'], prog_bar=False)

        return loss_p

    def validation_step(self, batch, batch_idx):
        p_pred, _, _, _, _, _, _ = self(batch['ts_input'], batch['nwp_input'])
        mae = F.l1_loss(p_pred, batch['target_p'])
        self.log('val_mae', mae, on_epoch=True, prog_bar=True)
        return mae
