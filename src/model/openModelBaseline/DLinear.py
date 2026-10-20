import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# 导入你完美封装的单任务适配器
from src.optimizer.BaselineOptimizer import Standard_MTLOptimizer

# ==============================================================================
# 1. 序列分解机制 (Series Decomposition) 
# ==============================================================================
class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

# ==============================================================================
# 2. DLinear 满血方案B版 (时间共享权重的纯线性多元回归)
# ==============================================================================
class DLinear(pl.LightningModule):
    def __init__(self, nwp_dim, num_classes, seq_len=48, pred_len=24):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False 
        
        self.num_classes = num_classes
        self.nwp_dim = nwp_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.unified_dim = 1 + nwp_dim + 6
        
        self.decomp = series_decomp(kernel_size=25)
        
        # 1. 历史演化流 (跨越时间步的纯线性映射)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        
        # 2. ★ 核心修正：气象先验流 (时间分布的多元线性偏置) ★
        # 注意：这里仅仅是一个 [nwp_feature_count, 1] 的极小权重矩阵。
        # 它会在 pred_len 的每一个时间步上共享参数，完美防止高维灾难。
        nwp_feature_count = nwp_dim + 6
        self.Linear_NWP = nn.Linear(nwp_feature_count, 1)

    def forward(self, ts, nwp):
        B, seq_len, _ = ts.shape
        
        # ==============================================================
        # 1. 异构数据对齐
        # ==============================================================
        ts_unified = torch.cat([ts[:, :, 0:1], torch.zeros(B, seq_len, self.nwp_dim, device=ts.device), ts[:, :, 1:7]], dim=-1)
        # 将 NWP 也对齐为统一格式，方便后续处理
        nwp_unified = torch.cat([torch.zeros(B, self.pred_len, 1, device=nwp.device), nwp[:, :, 0:self.nwp_dim], nwp[:, :, self.nwp_dim:]], dim=-1)

        # ==============================================================
        # 2. 序列分解与历史纯线性推演
        # ==============================================================
        seasonal_init, trend_init = self.decomp(ts_unified)
        
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        
        seasonal_output = self.Linear_Seasonal(seasonal_init) # [B, M, pred_len]
        trend_output = self.Linear_Trend(trend_init)          # [B, M, pred_len]
        
        pred_history = (seasonal_output + trend_output).permute(0, 2, 1) # [B, pred_len, M]
        
        # 仅仅提取主功率通道的历史线性外推结果
        p_hist_pred = pred_history[:, :, 0] # [B, pred_len]

        # ==============================================================
        # 3. 物理意义级的 NWP 注入 (Time-Distributed Linear Bias)
        # ==============================================================
        # 提取真正有意义的气象特征列 (排除了第 0 列的占位符 0)
        nwp_features = nwp_unified[:, :, 1:] # shape: [B, pred_len, nwp_feature_count]
        
        # PyTorch 的 nn.Linear 默认作用于张量的最后一个维度。
        # 因此，这里的计算等价于：对未来的每一个时间步 t，独立计算气象变量的加权和。
        # nwp_contribution shape: [B, pred_len, 1]
        nwp_contribution = self.Linear_NWP(nwp_features) 
        
        # 压缩掉最后的维度，使其与功率预测形状一致 [B, pred_len]
        p_nwp_contribution = nwp_contribution.squeeze(-1) 
        
        # 最终预测 = 历史长期趋势的线性外推 + 当前时刻天气的即时线性偏置
        p_pred = p_hist_pred + p_nwp_contribution

        # ==============================================================
        # 4. MTL 框架兼容伪造占位符
        # ==============================================================
        dummy_dfi = torch.zeros_like(p_pred)
        dummy_w_logits = torch.zeros(B, self.num_classes, device=self.device)
        dummy_ramp_logits = torch.zeros(B, 24, 5, device=self.device)
        dummy_p_sh_dict = {'m': 1.0, 'd': 0.0, 'w': 0.0, 'r': 0.0}
        
        return p_pred, dummy_dfi, dummy_w_logits, dummy_ramp_logits, dummy_p_sh_dict

    def configure_optimizers(self):
        base_opt = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
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