import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

# 从本地包导入优化器
from src.optimizer.NashOptimizer_Hardshare import NashMTLOptimizer

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


class PVTransformer_QuadTask_Advanced(pl.LightningModule):
    def __init__(self, nwp_dim, num_classes, d_model=128, nhead=4, pred_len=24):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        # 主任务不确定性权重 (保留以进行博弈)
        self.log_vars = nn.Parameter(torch.zeros(4)) 

        # 1. 嵌入层
        self.ts_patcher = TokenEmbedding(input_dim=7, d_model=d_model)
        self.nwp_patcher = TokenEmbedding(input_dim=nwp_dim+6, d_model=d_model)
        
        # 位置编码
        self.pos_enc_ts = nn.Parameter(torch.randn(1, 100, d_model) * 0.1)
        self.pos_enc_nwp = nn.Parameter(torch.randn(1, 100, d_model) * 0.1)
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=0.1, batch_first=True)
        self.fusion_norm = nn.LayerNorm(d_model)
        
        # 2. 【硬共享底座】 (Shared Bottom)
        # 这是消融实验中唯一的共享部分
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True, dropout=0.2, norm_first=True)
        self.shared_bottom = nn.TransformerEncoder(layer, num_layers=2)
        
        # 展平后的特征维度
        flat_dim = d_model * pred_len
        
        # 3. 任务预测头 (Task Heads)
        # 所有的头直接连接在 Shared Bottom 之后
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
        
        # 直接通过硬共享底座
        bottom_out = self.shared_bottom(fused_feat)
        
        # 移除路由融合逻辑，直接将 bottom_out 送入各预测头
        ramp_logits = self.head_ramp(bottom_out).view(-1, 24, 5) 

        # 为保持兼容性，返回一个空的共享使用字典
        p_sh_dict = {'m': 1.0, 'd': 1.0, 'w': 1.0, 'r': 1.0}

        return self.head_power(bottom_out), self.head_dfi(bottom_out), self.head_weather(bottom_out), ramp_logits, p_sh_dict

    def configure_optimizers(self):
        base_opt = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)
        
        # 共享参数：包括底座及之前的所有结构
        shared_params = list(self.ts_patcher.parameters()) + list(self.nwp_patcher.parameters()) + \
                        [self.pos_enc_ts, self.pos_enc_nwp] + list(self.cross_attention.parameters()) + \
                        list(self.fusion_norm.parameters()) + list(self.shared_bottom.parameters()) 
        
        # 私有参数：仅包含各自的任务头
        main_params = list(self.head_power.parameters()) + [self.log_vars]
        dfi_params = list(self.head_dfi.parameters())
        w_params = list(self.head_weather.parameters())
        ramp_params = list(self.head_ramp.parameters()) 
        
        self.nash_optimizer = NashMTLOptimizer(base_opt, shared_params, main_params, dfi_params, w_params, ramp_params) 
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(base_opt, T_0=15, T_mult=2, eta_min=1e-6)
        return [base_opt], [scheduler]

    def training_step(self, batch, batch_idx):
        opt = self.nash_optimizer 
        p_pred, dfi_pred, w_logits, ramp_logits, _ = self(batch['ts_input'], batch['nwp_input'])
        
        loss_p_raw = F.huber_loss(p_pred, batch['target_p'], delta=0.5)
        loss_dfi_raw = F.mse_loss(dfi_pred, batch['target_dfi'])
        loss_w_raw = F.cross_entropy(w_logits, batch['daily_weather'])
        loss_ramp_raw = F.cross_entropy(ramp_logits.view(-1, 5), batch['target_ramp'].view(-1))
        
        log_var_p = torch.clamp(self.log_vars[0], max=2.0)
        log_var_dfi = torch.clamp(self.log_vars[1], max=2.0)
        log_var_w = torch.clamp(self.log_vars[2], max=2.0)
        log_var_ramp = torch.clamp(self.log_vars[3], max=2.0)
        
        loss_p = torch.exp(-log_var_p) * loss_p_raw + log_var_p
        loss_dfi = torch.exp(-log_var_dfi) * loss_dfi_raw + log_var_dfi
        loss_w = torch.exp(-log_var_w) * loss_w_raw + log_var_w
        loss_ramp = torch.exp(-log_var_ramp) * loss_ramp_raw + log_var_ramp
        
        decay_factor = 0.2 + 0.8 * 0.5 * (1 + np.cos(np.pi * self.current_epoch / self.trainer.max_epochs)) 
        metrics = opt.backward(loss_p, loss_dfi, loss_w, loss_ramp, decay_factor)
        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        opt.step()
        if self.trainer.is_last_batch: self.lr_schedulers().step()

        self.log('train_loss_p', loss_p_raw, prog_bar=True)
        self.log('3_Nash/CosTheta_Overall_Combined', metrics['cos_combined'], prog_bar=True) 
        self.log('7_Gradient_Angle/Cos_Main_vs_DFI', metrics['cos_md'], prog_bar=False)
        self.log('7_Gradient_Angle/Cos_Main_vs_Weather', metrics['cos_mw'], prog_bar=False)
        self.log('7_Gradient_Angle/Cos_Main_vs_Ramp', metrics['cos_mr'], prog_bar=False) 
        
        return loss_p

    def validation_step(self, batch, batch_idx):
        p_pred, _, _, _, _ = self(batch['ts_input'], batch['nwp_input'])
        mae = F.l1_loss(p_pred, batch['target_p'])
        self.log('val_mae', mae, on_epoch=True, prog_bar=True) 
        return mae