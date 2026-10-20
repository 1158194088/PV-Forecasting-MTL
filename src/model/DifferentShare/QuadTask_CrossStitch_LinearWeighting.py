import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

# ==============================================================================
# ★ 无 Patch 的逐点嵌入层 (Point-wise Token Embedding)
# ==============================================================================
class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.value_embedding = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: [B, L, C] -> [B, L, d_model]
        return self.norm(self.value_embedding(x))

# ==============================================================================
# ★ 基线模型: Cross-Stitch (经典软共享) + 平等加权 (Equal Weighting)
# ==============================================================================
class PVTransformer_QuadTask_Advanced(pl.LightningModule):
    def __init__(self, nwp_dim, num_classes, d_model=128, nhead=4, patch_len=8, stride=4, pred_len=24):
        super().__init__()
        self.save_hyperparameters()
        
        # 恢复自动优化机制，完全交由 Lightning 处理反向传播
        self.automatic_optimization = True

        self.ts_patcher = TokenEmbedding(input_dim=7, d_model=d_model)
        self.nwp_patcher = TokenEmbedding(input_dim=nwp_dim+6, d_model=d_model)
        
        self.pos_enc_ts = nn.Parameter(torch.randn(1, 100, d_model) * 0.1)
        self.pos_enc_nwp = nn.Parameter(torch.randn(1, 100, d_model) * 0.1)
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=0.1, batch_first=True)
        self.fusion_norm = nn.LayerNorm(d_model)
        
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True, dropout=0.2, norm_first=True)
        self.shared_bottom = nn.TransformerEncoder(layer, num_layers=2)
        
        # ======================================================================
        # ★ 核心修改 1: 软共享的独立任务分支 (Task-Specific Branches)
        # 废除公共专家，每个任务拥有自己独立参数的特征提取通道
        # ======================================================================
        self.branch_main = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(0.15), nn.Linear(d_model, d_model)
        )
        self.branch_dfi = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(0.15), nn.Linear(d_model // 2, d_model)
        )
        self.branch_w = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(0.15), nn.Linear(d_model // 2, d_model)
        )
        self.branch_ramp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(0.15), nn.Linear(d_model * 2, d_model)
        )
        
        # ======================================================================
        # ★ 核心修改 2: Cross-Stitch 可学习软共享矩阵
        # 一个 4x4 的矩阵。初始化策略：倾向于保留自身特征(0.85)，少量吸收其他任务特征(0.05)
        # 任务顺序: 0:Power, 1:DFI, 2:Weather, 3:Ramp
        # ======================================================================
        init_cs_weights = torch.eye(4) * 0.85 + torch.ones(4, 4) * 0.05
        self.cross_stitch_matrix = nn.Parameter(init_cs_weights)
        
        flat_dim = d_model * pred_len
        
        # 独立预测头 (知识孤岛，不进行 CA-FiLM 跨流交互)
        self.head_power = nn.Sequential(nn.Flatten(), nn.Linear(flat_dim, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 24))
        self.head_dfi = nn.Sequential(nn.Flatten(), nn.Linear(flat_dim, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 24))
        self.head_weather = nn.Sequential(nn.Flatten(), nn.Linear(flat_dim, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, num_classes))
        self.head_ramp = nn.Sequential(nn.Flatten(), nn.Linear(flat_dim, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 24 * 5)) 

    def forward(self, ts, nwp):
        p_ts = self.ts_patcher(ts)
        p_nwp = self.nwp_patcher(nwp)
        p_ts = p_ts + self.pos_enc_ts[:, :p_ts.size(1), :]
        p_nwp = p_nwp + self.pos_enc_nwp[:, :p_nwp.size(1), :]
        
        attn_out, _ = self.cross_attention(query=p_nwp, key=p_ts, value=p_ts)
        fused_feat = self.fusion_norm(p_nwp + attn_out)
        
        bottom_out = self.shared_bottom(fused_feat)
        
        # 1. 各个独立分支进行特征提取
        out_main = self.branch_main(bottom_out)
        out_dfi = self.branch_dfi(bottom_out)
        out_w = self.branch_w(bottom_out)
        out_ramp = self.branch_ramp(bottom_out)
        
        B, L, D = out_main.size()
        
        # ======================================================================
        # ★ 核心修改 3: 实施 Cross-Stitch 线性特征缝合
        # ======================================================================
        # 将 4 个任务的特征 Stack 在一起 -> Shape: [4, B, L, D]
        stacked_features = torch.stack([out_main, out_dfi, out_w, out_ramp], dim=0)
        
        # 展平以便与 4x4 的十字绣矩阵相乘 -> Shape: [4, B*L*D]
        stacked_flat = stacked_features.view(4, -1)
        
        # 执行软共享：每个任务的新特征 = 自己特征*w1 + 其它任务特征*w2...
        stitched_flat = torch.matmul(self.cross_stitch_matrix, stacked_flat) # [4, B*L*D]
        
        # 还原维度 -> Shape: [4, B, L, D]
        stitched_features = stitched_flat.view(4, B, L, D)
        
        # 拆分给各个任务的预测头
        fused_main = stitched_features[0]
        fused_dfi = stitched_features[1]
        fused_w = stitched_features[2]
        fused_ramp = stitched_features[3]
        
        ramp_logits = self.head_ramp(fused_ramp).view(-1, 24, 5) 

        # ======================================================================
        # ★ 记录软共享权重 (用于 Tensorboard 观察主任务究竟吸收了谁的特征)
        # ======================================================================
        p_sh_dict = {
            'main_self': self.cross_stitch_matrix[0, 0].item(), # 主任务保留自身特征的比例
            'main_from_dfi': self.cross_stitch_matrix[0, 1].item(), # 主任务从DFI借用的比例
            'main_from_w': self.cross_stitch_matrix[0, 2].item(), # 主任务从Weather借用的比例
            'main_from_ramp': self.cross_stitch_matrix[0, 3].item() # 主任务从Ramp借用的比例
        }

        return self.head_power(fused_main), self.head_dfi(fused_dfi), self.head_weather(fused_w), ramp_logits, p_sh_dict

    def configure_optimizers(self):
        # 标准优化器配置
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        p_pred, dfi_pred, w_logits, ramp_logits, p_sh_dict = self(batch['ts_input'], batch['nwp_input'])
        
        loss_p = F.huber_loss(p_pred, batch['target_p'], delta=0.5)
        loss_dfi = F.mse_loss(dfi_pred, batch['target_dfi'])
        loss_w = F.cross_entropy(w_logits, batch['daily_weather'])
        loss_ramp = F.cross_entropy(ramp_logits.view(-1, 5), batch['target_ramp'].view(-1))
        
        # 痛点展示区: 传统的静态平等加权 (Equal Weighting)
        total_loss = loss_p + loss_dfi + loss_w + loss_ramp
        
        self.log('train_loss_total', total_loss, prog_bar=True)
        self.log('1_Loss/Main_Power', loss_p, prog_bar=False)
        self.log('1_Loss/Aux_DFI', loss_dfi, prog_bar=False)
        self.log('1_Loss/Aux_Weather', loss_w, prog_bar=False)
        self.log('1_Loss/Aux_Ramp', loss_ramp, prog_bar=False)
        
        # 监控 Cross-Stitch 权重的演变
        self.log('6_CS_Weight/Main_Self', p_sh_dict['main_self'], prog_bar=False)
        self.log('6_CS_Weight/Main_from_DFI', p_sh_dict['main_from_dfi'], prog_bar=False)
        self.log('6_CS_Weight/Main_from_Weather', p_sh_dict['main_from_w'], prog_bar=False)
        self.log('6_CS_Weight/Main_from_Ramp', p_sh_dict['main_from_ramp'], prog_bar=False)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        p_pred, _, _, _, _ = self(batch['ts_input'], batch['nwp_input'])
        mae = F.l1_loss(p_pred, batch['target_p'])
        self.log('val_mae', mae, on_epoch=True, prog_bar=True) 
        return mae