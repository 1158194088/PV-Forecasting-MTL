import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


# ==============================================================================
# 无 Patch 的逐点嵌入层
# ==============================================================================
class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.value_embedding = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.value_embedding(x))


class PVTransformer_QuadTask_Advanced(pl.LightningModule):
    """
    框架消融版：CGC(Hetero) + Linear Weighting。

    与最新完整模型相比，本版本仅保留：
    1. 双模态硬共享底座；
    2. 异构 CGC 共享-专属专家池；
    3. DFI / Weather / Ramp 三个辅助任务头；
    4. 主任务功率预测头；
    5. 四任务线性加权训练。

    去掉的部分：
    1. RouteReg：不再对 Weather/Ramp 的共享专家权重添加 penalty；
    2. CA-FiLM：辅助任务预测不再调制主任务特征；
    3. Nash-MTL：不再使用 Nash 梯度协同，改为普通 Linear Weighting。

    该版本适合用于结构对比表中的：
    CGC(Hetero) + Linear Weighting。
    """

    def __init__(self, nwp_dim, num_classes, d_model=128, nhead=4,
                 patch_len=8, stride=4, pred_len=24):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = True

        self.nwp_dim = nwp_dim
        self.num_classes = num_classes
        self.d_model = d_model
        self.pred_len = pred_len

        # 1. 基础组件：双模态硬共享底座
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

        # 2. CGC(Hetero)：异构共享-专属专家池
        #    保留不同任务专家的容量差异，用于区别于 CGC(Standard)。
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

        # 任务专属门控。注意：保留 CGC 门控，但去掉 RouteReg penalty。
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

        # 3. 任务形态感知辅助头
        # DFI：逐小时连续物理量，[B, P, D] -> [B, P]
        self.head_dfi = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1),
        )

        # Weather：日级天气分类，先池化再分类 [B, D] -> [B, C]
        self.head_weather = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes),
        )

        # Ramp：逐小时突变状态分类，[B, P, D] -> [B, P, 5]
        self.head_ramp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 5),
        )

        # 4. 主任务预测头
        #    不使用 CA-FiLM，主任务直接使用 fused_main。
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

    def forward(self, ts, nwp):
        # 1. 双模态基础特征提取
        p_ts = self.ts_patcher(ts)
        p_nwp = self.nwp_patcher(nwp)
        p_ts = p_ts + self.pos_enc_ts[:, :p_ts.size(1), :]
        p_nwp = p_nwp + self.pos_enc_nwp[:, :p_nwp.size(1), :]

        attn_out, _ = self.cross_attention(query=p_nwp, key=p_ts, value=p_ts)
        fused_feat = self.fusion_norm(p_nwp + attn_out)
        bottom_out = self.shared_bottom(fused_feat)

        # 2. CGC 专家输出
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

        # 3. 辅助任务预测
        dfi_pred = self.head_dfi(fused_dfi).squeeze(-1)        # [B, P]
        weather_context = torch.mean(fused_w, dim=1)           # [B, D]
        w_logits = self.head_weather(weather_context)          # [B, C]
        ramp_logits = self.head_ramp(fused_ramp)               # [B, P, 5]

        # 4. 主任务预测：去掉 CA-FiLM 后，直接由 fused_main 输出功率
        p_pred = self.head_power(fused_main)                   # [B, P]

        p_sh_dict = {
            'm': w_main[:, 0].mean().detach().item(),
            'd': w_dfi[:, 0].mean().detach().item(),
            'w': w_w[:, 0].mean().detach().item(),
            'r': w_ramp[:, 0].mean().detach().item(),
        }

        # 保留 w_w / w_ramp 返回，方便评估器继续记录门控权重；但训练中不再使用 RouteReg penalty。
        return p_pred, dfi_pred, w_logits, ramp_logits, p_sh_dict, w_w, w_ramp

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=15,
            T_mult=2,
            eta_min=1e-6,
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        p_pred, dfi_pred, w_logits, ramp_logits, p_sh_dict, _, _ = self(
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

        # Linear Weighting：去掉 Nash-MTL 后，四个任务 loss 直接线性相加。
        # RouteReg 已去掉，因此不再添加 penalty_w / penalty_r。
        total_loss = loss_p + loss_dfi + loss_w + loss_ramp

        self.log('train_loss_total', total_loss, prog_bar=True)
        self.log('train_loss_p', loss_p, prog_bar=True)
        self.log('1_Loss/Aux_DFI_Raw', loss_dfi, prog_bar=False)
        self.log('1_Loss/Aux_Weather_Raw', loss_w, prog_bar=False)
        self.log('1_Loss/Aux_Ramp_Raw', loss_ramp, prog_bar=False)

        self.log('6_Gate/Main_use_Shared', p_sh_dict['m'], prog_bar=False)
        self.log('6_Gate/Weather_use_Shared', p_sh_dict['w'], prog_bar=False)
        self.log('6_Gate/Ramp_use_Shared', p_sh_dict['r'], prog_bar=False)
        self.log('6_Gate/DFI_use_Shared', p_sh_dict['d'], prog_bar=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        p_pred, _, _, _, _, _, _ = self(batch['ts_input'], batch['nwp_input'])
        mae = F.l1_loss(p_pred, batch['target_p'])
        self.log('val_mae', mae, on_epoch=True, prog_bar=True)
        return mae
