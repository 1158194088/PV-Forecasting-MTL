import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.modules.PatchTST_Modules import PatchTST_Embedding, RevIN
from src.optimizer.BaselineOptimizer import Standard_MTLOptimizer

class PatchTST_SingleTask_Baseline(pl.LightningModule):
    def __init__(self, nwp_dim, num_classes, d_model=128, nhead=4, patch_len=8, stride=4, seq_len=48, pred_len=24):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False 
        self.num_classes = num_classes
        
        self.ts_channels = 7
        self.nwp_channels = nwp_dim + 6
        self.pred_len = pred_len
        
        # 1. 局部可逆归一化 (RevIN)
        self.revin_ts = RevIN(num_features=self.ts_channels)
        self.revin_nwp = RevIN(num_features=self.nwp_channels)
        
        # 2. PatchTST Embedding
        self.ts_patch_emb = PatchTST_Embedding(d_model, patch_len, stride)
        self.nwp_patch_emb = PatchTST_Embedding(d_model, patch_len, stride)
        
        # 3. 共享 Transformer 编码器
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True, dropout=0.2, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=3)
        
        # 计算 Patch 数量
        ts_pad = stride - (seq_len - patch_len) % stride
        ts_L = seq_len + ts_pad if ts_pad < stride else seq_len
        self.ts_N = (ts_L - patch_len) // stride + 1
        
        nwp_pad = stride - (pred_len - patch_len) % stride
        nwp_L = pred_len + nwp_pad if nwp_pad < stride else pred_len
        self.nwp_N = (nwp_L - patch_len) // stride + 1
        
        # ==============================================================
        # ★ 满血版重构：通道独立预测头 (Channel Independent Heads)
        # 舍弃原本暴力的全连接，改为每个通道独立映射出预测长度
        # ==============================================================
        self.ts_head = nn.Linear(self.ts_N * d_model, pred_len)
        self.nwp_head = nn.Linear(self.nwp_N * d_model, pred_len)
        
        # 轻量级通道融合：将不同通道的预测结果汇聚成最终的 1 维功率预测
        self.fusion_head = nn.Linear(self.ts_channels + self.nwp_channels, 1)

    def forward(self, ts, nwp):
        B = ts.size(0)
        
        # 1. 归一化 (Norm)
        ts_norm = self.revin_ts(ts, mode='norm')
        nwp_norm = self.revin_nwp(nwp, mode='norm')
        
        # 2. 通道独立 (CI)
        ts_ci = ts_norm.transpose(1, 2).reshape(B * self.ts_channels, -1, 1)
        nwp_ci = nwp_norm.transpose(1, 2).reshape(B * self.nwp_channels, -1, 1)
        
        # 3. Patch & Encode
        ts_emb = self.ts_patch_emb(ts_ci)       
        nwp_emb = self.nwp_patch_emb(nwp_ci)    
        ts_out = self.transformer_encoder(ts_emb)   
        nwp_out = self.transformer_encoder(nwp_emb) 
        
        # ==============================================================
        # ★ 满血版重构：独立映射与晚期轻量融合
        # ==============================================================
        # 每个变量独立映射：[B*C, N, d_model] -> [B*C, pred_len]
        ts_proj = self.ts_head(ts_out.reshape(B * self.ts_channels, -1))
        nwp_proj = self.nwp_head(nwp_out.reshape(B * self.nwp_channels, -1))
        
        # 恢复通道维度：[B, C, pred_len]
        ts_proj = ts_proj.reshape(B, self.ts_channels, self.pred_len)
        nwp_proj = nwp_proj.reshape(B, self.nwp_channels, self.pred_len)
        
        # 融合所有通道：拼接后转置为 [B, pred_len, C_total]，通过 fusion_head 降维到 1
        fused_channels = torch.cat([ts_proj, nwp_proj], dim=1).transpose(1, 2)
        p_pred_norm = self.fusion_head(fused_channels) # [B, pred_len, 1]
        
        # 5. 反归一化 (Denorm) -> 仅针对功率维度(channel_idx=0)
        p_pred = self.revin_ts.denorm_target(p_pred_norm, channel_idx=0)
        p_pred = p_pred.squeeze(-1) # [B, 24]
        
        # 6. 生成 Dummy 变量以兼容多任务训练与评估框架
        dummy_dfi = torch.zeros_like(p_pred)
        dummy_w_logits = torch.zeros(B, self.num_classes, device=self.device)
        dummy_ramp_logits = torch.zeros(B, 24, 5, device=self.device)
        dummy_p_sh_dict = {'m': 1.0, 'd': 0.0, 'w': 0.0, 'r': 0.0}
        
        return p_pred, dummy_dfi, dummy_w_logits, dummy_ramp_logits, dummy_p_sh_dict

    def configure_optimizers(self):
        base_opt = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)
        self.standard_optimizer = Standard_MTLOptimizer(base_opt) 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(base_opt, T_0=15, T_mult=2, eta_min=1e-6)
        return [base_opt], [scheduler]

    def training_step(self, batch, batch_idx):
        opt = self.standard_optimizer 
        p_pred, _, _, _, _ = self(batch['ts_input'], batch['nwp_input'])
        
        loss_p = F.huber_loss(p_pred, batch['target_p'], delta=0.5)
        
        opt.backward(loss_p, loss_p, loss_p, loss_p, p_sh_dict=None, decay_factor=1.0)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        opt.step()
        
        if self.trainer.is_last_batch: 
            self.lr_schedulers().step()

        self.log('train_loss_p', loss_p, prog_bar=True)
        return loss_p

    def validation_step(self, batch, batch_idx):
        p_pred, _, _, _, _ = self(batch['ts_input'], batch['nwp_input'])
        mae = F.l1_loss(p_pred, batch['target_p'])
        self.log('val_mae', mae, on_epoch=True, prog_bar=True) 
        return mae