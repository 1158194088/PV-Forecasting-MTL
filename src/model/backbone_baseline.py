import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
# Backbone‑only model without multi‑task branches
# ==============================================================================
# Map each hourly time‑step independently to d_model dimension,
# no compression along the time dimension
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
    def __init__(self, nwp_dim, d_model=128, nhead=4, patch_len=8, stride=4, pred_len=24):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = True
        
        self.ts_embed = TokenEmbedding(input_dim=7, d_model=d_model)
        self.nwp_embed = TokenEmbedding(input_dim=nwp_dim+6, d_model=d_model)
        
        self.pos_enc_ts = nn.Parameter(torch.randn(1, 100, d_model) * 0.1)
        self.pos_enc_nwp = nn.Parameter(torch.randn(1, 100, d_model) * 0.1)
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=0.1, batch_first=True)
        self.fusion_norm = nn.LayerNorm(d_model)
        
        # =======================================================
        # 2. Backbone network
        # =======================================================
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True, dropout=0.2, norm_first=True)
        self.shared_bottom = nn.TransformerEncoder(layer, num_layers=2)
        
        self.feature_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(0.15), nn.Linear(d_model, d_model)
        )
        
        # =======================================================
        # 3. Prediction heads
        # =======================================================
        self.flat_dim = d_model * pred_len 
        
        self.head_power = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(self.flat_dim, 256), 
            nn.GELU(), 
            nn.Dropout(0.2), 
            nn.Linear(256, 24)
        )

    def forward(self, ts, nwp):
        # Point‑wise feature projection
        p_ts = self.ts_embed(ts)
        p_nwp = self.nwp_embed(nwp)
        
        p_ts = p_ts + self.pos_enc_ts[:, :p_ts.size(1), :]
        p_nwp = p_nwp + self.pos_enc_nwp[:, :p_nwp.size(1), :]
        
        attn_out, _ = self.cross_attention(query=p_nwp, key=p_ts, value=p_ts)
        fused_feat = self.fusion_norm(p_nwp + attn_out)
        
        bottom_out = self.shared_bottom(fused_feat)
        main_out = self.feature_proj(bottom_out)
        
        p_pred = self.head_power(main_out)
        
        B = p_pred.shape[0]
        device = p_pred.device
        
        # 1. Dummy DFI prediction: all‑zero tensor
        dummy_dfi = torch.zeros_like(p_pred)
        
        # 2. Dummy weather classification logits: all‑zero tensor
        dummy_w_logits = torch.zeros((B, 5), device=device)
        
        # 3. Dummy ramp‑rate classification logits: all‑zero tensor
        dummy_ramp_logits = torch.zeros((B, 24, 5), device=device)
        
        # 4. Dummy gating weight dictionary
        dummy_p_sh_dict = {'m': 1.0, 'd': 0.0, 'w': 0.0, 'r': 0.0}
        
        return p_pred, dummy_dfi, dummy_w_logits, dummy_ramp_logits, dummy_p_sh_dict

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        p_pred, _, _, _, _ = self(batch['ts_input'], batch['nwp_input'])
        
        loss = F.huber_loss(p_pred, batch['target_p'], delta=0.5)
        self.log('train_loss_p', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        p_pred, _, _, _, _ = self(batch['ts_input'], batch['nwp_input'])
        
        mae = F.l1_loss(p_pred, batch['target_p'])
        self.log('val_mae', mae, on_epoch=True, prog_bar=True) 
        return mae
