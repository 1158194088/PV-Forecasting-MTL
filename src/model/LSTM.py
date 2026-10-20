import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.optimizer.BaselineOptimizer import Standard_MTLOptimizer

class PVTransformer_QuadTask_Advanced(pl.LightningModule):
    def __init__(self, nwp_dim, num_classes, d_model=128, num_layers=2, seq_len=48, pred_len=24):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False 
        self.num_classes = num_classes
        self.nwp_dim = nwp_dim
        self.pred_len = pred_len
        
        # 兼容日志报错的占位符
        self.log_vars = nn.Parameter(torch.zeros(4)) 
        
        # 异构统一化后的总维度 = 1(Power) + nwp_dim + 6(Time)
        self.unified_dim = 1 + nwp_dim + 6
        
        self.embedding = nn.Linear(self.unified_dim, d_model)
        
        # 纯正单向 LSTM
        dropout_rate = 0.2 if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=d_model, 
            hidden_size=d_model, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=False, 
            dropout=dropout_rate
        )
            
        self.head_power = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, ts, nwp):
        B, seq_len, _ = ts.shape
        
        # 1. 异构数据统一化
        ts_power = ts[:, :, 0:1]
        ts_time = ts[:, :, 1:7]
        ts_nwp_zeros = torch.zeros(B, seq_len, self.nwp_dim, device=ts.device)
        ts_unified = torch.cat([ts_power, ts_nwp_zeros, ts_time], dim=-1)

        nwp_feat = nwp[:, :, 0:self.nwp_dim]
        nwp_time = nwp[:, :, self.nwp_dim:]
        nwp_power_zeros = torch.zeros(B, self.pred_len, 1, device=nwp.device)
        nwp_unified = torch.cat([nwp_power_zeros, nwp_feat, nwp_time], dim=-1)

        # 2. 时序拼接
        full_seq = torch.cat([ts_unified, nwp_unified], dim=1)
        
        # 3. 编码
        emb = self.embedding(full_seq)
        lstm_out, _ = self.lstm(emb) 
        
        # 4. 预测
        future_out = lstm_out[:, -self.pred_len:, :]
        p_pred = self.head_power(future_out).squeeze(-1) 
        
        # 5. Dummy 占位符
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
        if self.trainer.is_last_batch: self.lr_schedulers().step()

        self.log('train_loss_p', loss_p, prog_bar=True)
        return loss_p

    def validation_step(self, batch, batch_idx):
        p_pred, _, _, _, _ = self(batch['ts_input'], batch['nwp_input'])
        mae = F.l1_loss(p_pred, batch['target_p'])
        self.log('val_mae', mae, on_epoch=True, prog_bar=True) 
        return mae