import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# iTransformer + QuadTask module 泛化验证版
# 保留 iTransformer 原始输入对齐、倒置嵌入、Transformer Encoder、hist/nwp bottleneck 融合流程。
# 仅将原始 joint_decoder 替换为 CGC + All-Aux Temporal CA-FiLM + Nash-MTL 多任务模块。
from src.optimizer.NashOptimizer import NashMTLOptimizer

# 为验证所提多任务模块的主干无关性，本文在不改变 iTransformer 原始倒置变量建模与后端 NWP 瓶颈融合流程的前提下
# 仅将其单任务联合解码器替换为 CGC–Temporal CA-FiLM–Nash 多任务模块。该设置保证性能变化主要来源于所提多任务协同模块
# 而非 iTransformer 主干结构的重构

# ==============================================================================
# 1. 倒置嵌入层 (Inverted Data Embedding)
# ==============================================================================
class DataEmbedding_inverted(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x)
        return self.dropout(x)


class QuadTaskTemporalCAFiLMHead(nn.Module):
    """
    Backbone-agnostic QuadTask module.

    输入为 iTransformer 原始瓶颈融合后的 fused_emb: [B, 256]。
    模块内部只负责多任务解耦、辅助先验生成、逐小时 CA-FiLM 调制和主功率预测，
    不改变 iTransformer 原始的历史/NWP 数据融合方式。
    """
    def __init__(self, num_classes, d_model=128, pred_len=24, fused_dim=256, dropout=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.pred_len = pred_len

        # 将原 iTransformer 的 fused_emb 解码为预测窗口表征 [B, P, D]
        # 这相当于替换原来的 joint_decoder，但不改变 joint_decoder 之前的公开主干流程。
        self.future_feature_decoder = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, pred_len * d_model),
        )

        # ------------------------------------------------------------------
        # CGC 共享-专属专家池
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
        # 任务形态感知辅助头
        # ------------------------------------------------------------------
        # DFI: 逐小时连续回归 [B, P, D] -> [B, P]
        self.head_dfi = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1),
        )

        # Weather: 日级宏观语义分类，时间池化后分类 [B, D] -> [B, C]
        self.head_weather = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes),
        )

        # Ramp: 逐小时突变状态分类 [B, P, D] -> [B, P, 5]
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

        # 主任务功率预测头：保留整段曲线解码能力
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
        weights = F.softmax(gate(gate_input), dim=-1)  # [B, 2]
        fused = weights[:, 0:1, None] * shared_feat + weights[:, 1:2, None] * task_feat
        return fused, weights

    def forward(self, fused_emb):
        B = fused_emb.size(0)

        # 1. 从 iTransformer 融合瓶颈向量生成预测窗口表征
        bottom_out = self.future_feature_decoder(fused_emb).view(B, self.pred_len, self.d_model)

        # 2. CGC 专家路由
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

        # 3. 辅助预测
        dfi_pred = self.head_dfi(fused_dfi).squeeze(-1)       # [B, P]
        weather_context = torch.mean(fused_w, dim=1)          # [B, D]
        w_logits = self.head_weather(weather_context)         # [B, C]
        ramp_logits = self.head_ramp(fused_ramp)              # [B, P, 5]

        # 4. All-Aux Temporal CA-FiLM：使用辅助预测值构造上下文，不使用真实标签
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
        p_pred = self.head_power(fused_main_refined)                  # [B, P]

        p_sh_dict = {
            'm': w_main[:, 0].mean().detach().item(),
            'd': w_dfi[:, 0].mean().detach().item(),
            'w': w_w[:, 0].mean().detach().item(),
            'r': w_ramp[:, 0].mean().detach().item(),
        }

        return p_pred, dfi_pred, w_logits, ramp_logits, p_sh_dict, w_w, w_ramp


# ==============================================================================
# 2. iTransformer + QuadTask 模块泛化验证版
# ==============================================================================
class PVTransformer_QuadTask_Advanced(pl.LightningModule):
    def __init__(self, nwp_dim, num_classes, d_model=128, nhead=4, num_layers=3,
                 seq_len=48, pred_len=24, dropout=0.2):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.num_classes = num_classes
        self.nwp_dim = nwp_dim
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 异构变量对齐后的总通道数 M
        self.unified_dim = 1 + nwp_dim + 6

        # 1. 倒置序列嵌入：保持 iTransformer 原始逻辑
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, dropout)

        # 2. Transformer Encoder：保持 iTransformer 原始 Variate-as-Token 主干
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. 原始瓶颈融合流程：保持不变
        self.hist_projector = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
        )

        nwp_feature_count = nwp_dim + 6
        self.nwp_projector = nn.Sequential(
            nn.Linear(self.pred_len * nwp_feature_count, 128),
            nn.GELU(),
        )

        # 4. 用 QuadTask 模块替换原 joint_decoder
        self.quadtask_head = QuadTaskTemporalCAFiLMHead(
            num_classes=num_classes,
            d_model=d_model,
            pred_len=pred_len,
            fused_dim=128 + 128,
            dropout=dropout,
        )

    def forward(self, ts, nwp):
        B, seq_len, _ = ts.shape

        # ==============================================================
        # 1. 异构数据对齐：保持原 iTransformer 基线不变
        # ==============================================================
        ts_unified = torch.cat(
            [
                ts[:, :, 0:1],
                torch.zeros(B, seq_len, self.nwp_dim, device=ts.device, dtype=ts.dtype),
                ts[:, :, 1:7],
            ],
            dim=-1,
        )

        # ==============================================================
        # 2. 维度倒置：[B, seq_len, M] -> [B, M, seq_len]
        # ==============================================================
        x_enc = ts_unified.permute(0, 2, 1).contiguous()

        # ==============================================================
        # 3. iTransformer 主干编码：保持原始逻辑
        # ==============================================================
        enc_out = self.enc_embedding(x_enc)  # [B, M, d_model]
        enc_out = self.encoder(enc_out)      # [B, M, d_model]

        # ==============================================================
        # 4. 原始瓶颈融合：power token + flattened NWP
        # ==============================================================
        power_token_feat = enc_out[:, 0, :]              # [B, d_model]
        hist_emb = self.hist_projector(power_token_feat) # [B, 128]

        # 注意：保持原基线逻辑，直接使用最纯净的 nwp，不额外构造 nwp_unified。
        flat_nwp = nwp.flatten(start_dim=1)
        nwp_emb = self.nwp_projector(flat_nwp)           # [B, 128]

        fused_emb = torch.cat([hist_emb, nwp_emb], dim=-1)  # [B, 256]

        # ==============================================================
        # 5. QuadTask 多任务模块
        # ==============================================================
        return self.quadtask_head(fused_emb)

    def configure_optimizers(self):
        base_opt = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)

        # 共享参数：iTransformer 主干 + 原始瓶颈融合 + QuadTask 的窗口特征生成器 + 共享专家
        shared_params = (
            list(self.enc_embedding.parameters()) +
            list(self.encoder.parameters()) +
            list(self.hist_projector.parameters()) +
            list(self.nwp_projector.parameters()) +
            list(self.quadtask_head.future_feature_decoder.parameters()) +
            list(self.quadtask_head.expert_shared.parameters())
        )

        # 主任务参数：主专家、主门控、CA-FiLM 和功率头
        main_params = (
            list(self.quadtask_head.expert_main.parameters()) +
            list(self.quadtask_head.gate_main.parameters()) +
            list(self.quadtask_head.phys_projector.parameters()) +
            list(self.quadtask_head.film_generator.parameters()) +
            list(self.quadtask_head.head_power.parameters())
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

        # 保留核心模型中的专家防退化约束：Weather / Ramp 不应过度占用共享专家。
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
