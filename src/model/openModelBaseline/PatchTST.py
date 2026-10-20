import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# 导入你完美封装的单任务适配器
from src.optimizer.BaselineOptimizer import Standard_MTLOptimizer

# ==============================================================================
# 1. Patch 嵌入层 (Patch Embedding)
# ==============================================================================
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout=0.1):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.shape[-1]
        if (seq_len - self.patch_len) % self.stride != 0:
            pad_len = self.stride - ((seq_len - self.patch_len) % self.stride)
            x = F.pad(x, (0, pad_len), mode='replicate')
            
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x_emb = self.value_embedding(patches)
        return self.dropout(x_emb), patches.shape[1]

# ==============================================================================
# 2. PatchTST 满血自适应基线 (瓶颈潜空间融合，严守 CI)
# ==============================================================================
class PatchTST(pl.LightningModule):
    def __init__(self, nwp_dim, num_classes, d_model=128, nhead=4, num_layers=3, 
                 seq_len=48, pred_len=24, patch_len=16, stride=8, dropout=0.2):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False 
        
        self.num_classes = num_classes
        self.nwp_dim = nwp_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 异构变量对齐后的总通道数 M
        self.unified_dim = 1 + nwp_dim + 6
        
        # 1. Patch 嵌入与位置编码
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, dropout)
        pad_len = 0 if (seq_len - patch_len) % stride == 0 else stride - ((seq_len - patch_len) % stride)
        self.num_patches = (seq_len + pad_len - patch_len) // stride + 1
        self.W_pos = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        
        # 2. 纯粹的 Transformer Encoder (Channel Independence)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, 
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # ======================================================================
        # 3. ★ 核心改进：基于瓶颈对齐的联合解码器 (Bottleneck Fusion) ★
        # ======================================================================
        # 流 A：提取主功率历史特征并降维 (保留 CI)
        self.hist_projector = nn.Sequential(
            nn.Linear(self.num_patches * d_model, 128),
            nn.GELU()
        )
        
        # 流 B：NWP 特征展平并投影到相同的潜空间维度
        nwp_feature_count = nwp_dim + 6 
        self.nwp_projector = nn.Sequential(
            nn.Linear(self.pred_len * nwp_feature_count, 128),
            nn.GELU()
        )
        
        # 流 C：潜空间联合解码器 (参数量极小，允许非线性化学反应)
        self.joint_decoder = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, pred_len)
        )

    def forward(self, ts, nwp):
        B, seq_len, _ = ts.shape
        
        # ==============================================================
        # 1. 异构数据对齐
        # ==============================================================
        ts_unified = torch.cat([ts[:, :, 0:1], torch.zeros(B, seq_len, self.nwp_dim, device=ts.device), ts[:, :, 1:7]], dim=-1)

        # ==============================================================
        # 2. Channel Independence (CI) 机制与骨干特征提取
        # ==============================================================
        x_enc = ts_unified.permute(0, 2, 1).contiguous()
        x_enc = x_enc.view(B * self.unified_dim, seq_len)
        
        enc_out, num_patches = self.patch_embedding(x_enc)
        enc_out = enc_out + self.W_pos[:, :num_patches, :]
        enc_out = self.encoder(enc_out) # [B * M, num_patches, d_model]
        
        # 还原回多变量视图: [B, M, num_patches * d_model]
        enc_out = enc_out.view(B, self.unified_dim, -1)

        # ==============================================================
        # 3. 瓶颈特征融合与联合解码
        # ==============================================================
        # 严守 CI：仅仅抽取出主功率通道 (索引 0)
        power_hist_feat = enc_out[:, 0, :] # [B, num_patches * d_model]
        
        # 历史特征投影到 128 维潜空间
        hist_emb = self.hist_projector(power_hist_feat) # [B, 128]
        
        # NWP 展平并投影到 128 维潜空间
        flat_nwp = nwp.flatten(start_dim=1) # [B, pred_len * nwp_feature_count]
        nwp_emb = self.nwp_projector(flat_nwp) # [B, 128]
        
        # 在潜空间进行特征级拼接 (Latent Concat)
        fused_emb = torch.cat([hist_emb, nwp_emb], dim=-1) # [B, 256]
        
        # 通过联合解码器得到最终功率预测
        p_pred = self.joint_decoder(fused_emb) # [B, pred_len]

        # ==============================================================
        # 4. MTL 框架兼容
        # ==============================================================
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
        p_pred, dummy_dfi, dummy_w, dummy_ramp, p_sh_dict = self(batch['ts_input'], batch['nwp_input'])
        loss_p = F.huber_loss(p_pred, batch['target_p'], delta=0.5)
        
        metrics = opt.backward(
            loss_main=loss_p, loss_dfi=loss_p, loss_w=loss_p, loss_ramp=loss_p, 
            p_sh_dict=p_sh_dict, decay_factor=1.0
        )
        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        opt.step()
        if self.trainer.is_last_batch: 
            self.lr_schedulers().step()

        self.log('train_loss_p', loss_p, prog_bar=True)
        self.log('3_Nash/CosTheta_Overall_Combined', metrics['cos_combined'], prog_bar=True) 
        self.log('7_Gradient_Angle/Cos_Main_vs_DFI', metrics['cos_md'], prog_bar=False)
        return loss_p

    def validation_step(self, batch, batch_idx):
        p_pred, _, _, _, _ = self(batch['ts_input'], batch['nwp_input'])
        mae = F.l1_loss(p_pred, batch['target_p'])
        self.log('val_mae', mae, on_epoch=True, prog_bar=True) 
        return mae