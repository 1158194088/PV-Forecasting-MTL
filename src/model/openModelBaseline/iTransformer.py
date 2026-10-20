import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# 导入完美封装的单任务适配器
from src.optimizer.BaselineOptimizer import Standard_MTLOptimizer

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

# ==============================================================================
# 2. iTransformer 满血自适应版 (严守 Variate-as-Token 与轻量化解码)
# ==============================================================================
class iTransformer(pl.LightningModule):
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
        
        # 1. 倒置序列嵌入
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, dropout)
        
        # 2. 纯粹的 Transformer Encoder (Channel Mixing)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, 
            dropout=dropout, activation='gelu', batch_first=True, norm_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # ======================================================================
        # 3. ★ 核心改进：遵循 iTransformer 哲学的瓶颈联合解码器 ★
        # ======================================================================
        # 流 A：仅仅映射主功率这一个 Token 的特征 (约 16K 参数)
        self.hist_projector = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU()
        )
        
        # 流 B：提取纯净的未来 NWP 数据，降维对齐 (约 46K 参数)
        nwp_feature_count = nwp_dim + 6
        self.nwp_projector = nn.Sequential(
            nn.Linear(self.pred_len * nwp_feature_count, 128),
            nn.GELU()
        )
        
        # 流 C：潜空间联合解码 (约 36K 参数)
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
        # 2. 核心：维度倒置 (Invert Dimensions)
        # [B, seq_len, M] -> [B, M, seq_len]
        # ==============================================================
        x_enc = ts_unified.permute(0, 2, 1).contiguous()
        
        # ==============================================================
        # 3. 嵌入与注意力交互 (Variate-as-Token)
        # ==============================================================
        enc_out = self.enc_embedding(x_enc) # [B, M, d_model]
        enc_out = self.encoder(enc_out)     # [B, M, d_model]
        
        # ==============================================================
        # 4. 瓶颈特征融合与联合解码
        # ==============================================================
        # 严守原著逻辑：仅仅抽取出主功率变量 (索引 0) 的 Token 表达
        power_token_feat = enc_out[:, 0, :] # [B, d_model]
        
        # 历史特征潜空间映射
        hist_emb = self.hist_projector(power_token_feat) # [B, 128]
        
        # NWP 展平并映射到潜空间 (注意：这里不需要补 0 的 nwp_unified，直接用最纯净的 nwp)
        flat_nwp = nwp.flatten(start_dim=1)              # [B, pred_len * (nwp_dim+6)]
        nwp_emb = self.nwp_projector(flat_nwp)           # [B, 128]
        
        # 潜空间特征级联
        fused_emb = torch.cat([hist_emb, nwp_emb], dim=-1) # [B, 256]
        
        # 最终非线性解码
        p_pred = self.joint_decoder(fused_emb)           # [B, pred_len]

        # ==============================================================
        # 5. MTL 框架兼容占位符
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
        return loss_p

    def validation_step(self, batch, batch_idx):
        p_pred, _, _, _, _ = self(batch['ts_input'], batch['nwp_input'])
        mae = F.l1_loss(p_pred, batch['target_p'])
        self.log('val_mae', mae, on_epoch=True, prog_bar=True) 
        return mae