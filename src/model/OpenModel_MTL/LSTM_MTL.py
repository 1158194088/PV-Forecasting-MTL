import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.optimizer.NashOptimizer import NashMTLOptimizer


class PVTransformer_QuadTask_Advanced(pl.LightningModule):
    """
    LSTM backbone + QuadTask module 泛化验证版

    公平性原则：
    1. 保留原始 LSTM 的输入构造方式：
       - 历史段: Power + zero NWP + time
       - 未来段: zero Power + future NWP + future time
       - 二者拼接后输入 LSTM
    2. 不改变 LSTM 主干的数据融合流程；
    3. 仅将原始单任务 head_power 替换为 QuadTask 多任务模块；
    4. DFI / Weather / Ramp 均作为辅助任务；
    5. 三个辅助预测均进入逐小时 Temporal CA-FiLM；
    6. 使用 Nash-MTL 进行多任务梯度协同优化。
    """

    def __init__(
        self,
        nwp_dim,
        num_classes,
        d_model=128,
        num_layers=2,
        seq_len=48,
        pred_len=24
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.num_classes = num_classes
        self.nwp_dim = nwp_dim
        self.pred_len = pred_len
        self.seq_len = seq_len

        # 兼容日志报错的占位符
        self.log_vars = nn.Parameter(torch.zeros(4))

        # ==========================================================
        # 1. 原始 LSTM 数据统一化与主干部分：保持不变
        # ==========================================================
        # 异构统一化后的总维度 = 1(Power) + nwp_dim + 6(Time)
        self.unified_dim = 1 + nwp_dim + 6

        self.embedding = nn.Linear(self.unified_dim, d_model)

        dropout_rate = 0.2 if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout_rate
        )

        # ==========================================================
        # 2. CGC 共享-专属专家池
        # 输入来自 LSTM 的 future_out: [B, pred_len, d_model]
        # ==========================================================
        self.expert_shared = nn.Sequential(
            nn.Linear(d_model, d_model * 3),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model * 3, d_model)
        )

        self.expert_main = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model, d_model)
        )

        self.expert_aux_dfi = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model // 2, d_model)
        )

        self.expert_aux_w = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model // 2, d_model)
        )

        self.expert_aux_ramp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model * 2, d_model)
        )

        self.gate_main = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)
        )

        self.gate_aux_dfi = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)
        )

        self.gate_aux_w = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)
        )

        self.gate_aux_ramp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)
        )

        flat_dim = d_model * pred_len

        # ==========================================================
        # 3. 任务形态感知辅助头
        # ==========================================================
        # DFI：逐小时连续物理量，[B, P, D] -> [B, P]
        self.head_dfi = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1)
        )

        # Weather：日级宏观天气类别，mean pooling 后分类 [B, D] -> [B, C]
        self.head_weather = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes)
        )

        # Ramp：逐小时突变状态分类，[B, P, D] -> [B, P, 5]
        self.head_ramp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 5)
        )

        # ==========================================================
        # 4. All-Aux Temporal CA-FiLM
        # 每个小时的物理上下文：
        # DFI: 1
        # Weather: num_classes，广播到每个小时
        # Ramp: 5
        # ==========================================================
        input_phys_step_dim = 1 + num_classes + 5

        self.phys_projector = nn.Sequential(
            nn.Linear(input_phys_step_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        self.film_generator = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model * 2)
        )

        # 零初始化保护：训练初期 Temporal CA-FiLM 等价于恒等映射
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)

        # ==========================================================
        # 5. 主任务功率预测头
        # ==========================================================
        self.head_power = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, pred_len)
        )

    def _gate_fuse(self, gate, gate_input, shared_feat, task_feat):
        weights = F.softmax(gate(gate_input), dim=-1)  # [B, 2]
        fused = weights[:, 0:1, None] * shared_feat + weights[:, 1:2, None] * task_feat
        return fused, weights

    def forward(self, ts, nwp):
        B, seq_len, _ = ts.shape

        # ==========================================================
        # 1. 原始 LSTM 输入构造：保持不变
        # ==========================================================
        ts_power = ts[:, :, 0:1]
        ts_time = ts[:, :, 1:7]
        ts_nwp_zeros = torch.zeros(
            B,
            seq_len,
            self.nwp_dim,
            device=ts.device,
            dtype=ts.dtype
        )
        ts_unified = torch.cat([ts_power, ts_nwp_zeros, ts_time], dim=-1)

        nwp_feat = nwp[:, :, 0:self.nwp_dim]
        nwp_time = nwp[:, :, self.nwp_dim:]
        nwp_power_zeros = torch.zeros(
            B,
            self.pred_len,
            1,
            device=nwp.device,
            dtype=nwp.dtype
        )
        nwp_unified = torch.cat([nwp_power_zeros, nwp_feat, nwp_time], dim=-1)

        # 历史序列 + 未来 NWP 序列拼接
        full_seq = torch.cat([ts_unified, nwp_unified], dim=1)

        # ==========================================================
        # 2. 原始 LSTM 编码：保持不变
        # ==========================================================
        emb = self.embedding(full_seq)
        lstm_out, _ = self.lstm(emb)

        # 取未来预测窗口对应的 LSTM 表征
        bottom_out = lstm_out[:, -self.pred_len:, :]  # [B, P, D]

        # ==========================================================
        # 3. CGC 共享-专属专家解耦
        # ==========================================================
        out_shared = self.expert_shared(bottom_out)
        out_main = self.expert_main(bottom_out)
        out_dfi = self.expert_aux_dfi(bottom_out)
        out_w = self.expert_aux_w(bottom_out)
        out_ramp = self.expert_aux_ramp(bottom_out)

        gate_input = torch.mean(bottom_out, dim=1)

        fused_main, w_main = self._gate_fuse(
            self.gate_main,
            gate_input,
            out_shared,
            out_main
        )

        fused_dfi, w_dfi = self._gate_fuse(
            self.gate_aux_dfi,
            gate_input,
            out_shared,
            out_dfi
        )

        fused_w, w_w = self._gate_fuse(
            self.gate_aux_w,
            gate_input,
            out_shared,
            out_w
        )

        fused_ramp, w_ramp = self._gate_fuse(
            self.gate_aux_ramp,
            gate_input,
            out_shared,
            out_ramp
        )

        # ==========================================================
        # 4. 三个辅助任务预测
        # ==========================================================
        dfi_pred = self.head_dfi(fused_dfi).squeeze(-1)  # [B, P]

        weather_context = torch.mean(fused_w, dim=1)
        w_logits = self.head_weather(weather_context)    # [B, C]

        ramp_logits = self.head_ramp(fused_ramp)         # [B, P, 5]

        # ==========================================================
        # 5. All-Aux Temporal CA-FiLM
        # 使用辅助任务“预测值”构造物理上下文，不使用真实标签
        # detach 保证单向知识注入，避免主任务梯度污染辅助分支
        # ==========================================================
        with torch.no_grad():
            dfi_context = dfi_pred.detach().unsqueeze(-1)  # [B, P, 1]

            weather_prob = F.softmax(w_logits.detach(), dim=-1)
            weather_context_step = weather_prob.unsqueeze(1).expand(
                -1,
                self.pred_len,
                -1
            )  # [B, P, C]

            ramp_context = F.softmax(ramp_logits.detach(), dim=-1)  # [B, P, 5]

            phys_context_step = torch.cat(
                [
                    dfi_context,
                    weather_context_step,
                    ramp_context
                ],
                dim=-1
            )  # [B, P, 1 + C + 5]

        phys_embedding_step = self.phys_projector(phys_context_step)  # [B, P, D]

        combined_info = torch.cat(
            [fused_main, phys_embedding_step],
            dim=-1
        )  # [B, P, 2D]

        film_params = self.film_generator(combined_info)  # [B, P, 2D]
        gamma, beta = torch.chunk(film_params, 2, dim=-1)

        fused_main_refined = fused_main * (1.0 + gamma) + beta  # [B, P, D]

        # ==========================================================
        # 6. 主任务预测
        # ==========================================================
        p_pred = self.head_power(fused_main_refined)  # [B, P]

        p_sh_dict = {
            'm': w_main[:, 0].mean().detach().item(),
            'd': w_dfi[:, 0].mean().detach().item(),
            'w': w_w[:, 0].mean().detach().item(),
            'r': w_ramp[:, 0].mean().detach().item()
        }

        return p_pred, dfi_pred, w_logits, ramp_logits, p_sh_dict, w_w, w_ramp

    def configure_optimizers(self):
        base_opt = torch.optim.AdamW(
            self.parameters(),
            lr=1e-4,
            weight_decay=1e-2
        )

        # 共享参数：LSTM 主干 + 共享专家
        shared_params = (
            list(self.embedding.parameters()) +
            list(self.lstm.parameters()) +
            list(self.expert_shared.parameters())
        )

        # 主任务参数：主专家、主门控、CA-FiLM、功率头
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

        self.nash_optimizer = NashMTLOptimizer(
            base_opt,
            shared_params,
            main_params,
            dfi_params,
            w_params,
            ramp_params
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            base_opt,
            T_0=15,
            T_mult=2,
            eta_min=1e-6
        )

        return [base_opt], [scheduler]

    def training_step(self, batch, batch_idx):
        opt = self.nash_optimizer

        p_pred, dfi_pred, w_logits, ramp_logits, p_sh_dict, w_w, w_r = self(
            batch['ts_input'],
            batch['nwp_input']
        )

        loss_p = F.huber_loss(p_pred, batch['target_p'], delta=0.5)
        loss_dfi = F.mse_loss(dfi_pred, batch['target_dfi'])
        loss_w = F.cross_entropy(w_logits, batch['daily_weather'])
        loss_ramp = F.cross_entropy(
            ramp_logits.reshape(-1, 5),
            batch['target_ramp'].reshape(-1)
        )

        # 保留你核心模型中的专家防退化约束
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
        p_pred, _, _, _, _, _, _ = self(
            batch['ts_input'],
            batch['nwp_input']
        )

        mae = F.l1_loss(p_pred, batch['target_p'])
        self.log('val_mae', mae, on_epoch=True, prog_bar=True)

        return mae