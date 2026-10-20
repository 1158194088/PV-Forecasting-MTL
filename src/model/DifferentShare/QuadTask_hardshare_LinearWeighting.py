import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
 
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
    def __init__(self, nwp_dim, num_classes, d_model=128, nhead=4, pred_len=24, task_weights=(1.0, 1.0, 1.0, 1.0)):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        # 最简单的线性加权：固定权重，可通过外部传入调整比例，默认等权
        self.task_weights = task_weights
 
        # 1. 嵌入层
        self.ts_patcher = TokenEmbedding(input_dim=7, d_model=d_model)
        self.nwp_patcher = TokenEmbedding(input_dim=nwp_dim+6, d_model=d_model)
        
        # 位置编码
        self.pos_enc_ts = nn.Parameter(torch.randn(1, 100, d_model) * 0.1)
        self.pos_enc_nwp = nn.Parameter(torch.randn(1, 100, d_model) * 0.1)
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=0.1, batch_first=True)
        self.fusion_norm = nn.LayerNorm(d_model)
        
        # 2. 【硬共享底座】 (Shared Bottom)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True, dropout=0.2, norm_first=True)
        self.shared_bottom = nn.TransformerEncoder(layer, num_layers=2)
        
        # 展平后的特征维度
        flat_dim = d_model * pred_len
        
        # 3. 任务预测头 (Task Heads)
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
        
        ramp_logits = self.head_ramp(bottom_out).view(-1, 24, 5) 
 
        return self.head_power(bottom_out), self.head_dfi(bottom_out), self.head_weather(bottom_out), ramp_logits
 
    def configure_optimizers(self):
        # 直接使用基础优化器，无需自定义Nash优化器
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)
        return [optimizer], [scheduler]
 
    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        
        p_pred, dfi_pred, w_logits, ramp_logits = self(batch['ts_input'], batch['nwp_input'])
        
        loss_p_raw = F.huber_loss(p_pred, batch['target_p'], delta=0.5)
        loss_dfi_raw = F.mse_loss(dfi_pred, batch['target_dfi'])
        loss_w_raw = F.cross_entropy(w_logits, batch['daily_weather'])
        loss_ramp_raw = F.cross_entropy(ramp_logits.view(-1, 5), batch['target_ramp'].view(-1))
        
        # ================= 核心改动：最简单的线性加权 =================
        # 不使用 log_vars 进行不确定性加权，直接使用固定常数加权求和
        w_p, w_d, w_w, w_r = self.task_weights
        total_loss = w_p * loss_p_raw + w_d * loss_dfi_raw + w_w * loss_w_raw + w_r * loss_ramp_raw
        
        # 标准的反向传播流程
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        
        if self.trainer.is_last_batch: 
            self.lr_schedulers().step()
 
        # 记录日志
        self.log('train_loss_p', loss_p_raw, prog_bar=True)
        self.log('train_loss_total', total_loss, prog_bar=True)
        
        return total_loss
 
    def validation_step(self, batch, batch_idx):
        p_pred, _, _, _ = self(batch['ts_input'], batch['nwp_input'])
        mae = F.l1_loss(p_pred, batch['target_p'])
        self.log('val_mae', mae, on_epoch=True, prog_bar=True) 
        return mae