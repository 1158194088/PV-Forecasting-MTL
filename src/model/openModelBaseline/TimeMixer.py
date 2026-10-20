import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# 导入你完美封装的单任务适配器
from src.optimizer.BaselineOptimizer import Standard_MTLOptimizer

# ==============================================================================
# 1. 序列分解机制 (Series Decomposition) - 与 Autoformer 保持一致
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
# 2. 混合器模块 (Mixer Block) - ICLR 2024 核心
# 纯 MLP 架构：沿时间维度混合 (Temporal Mix) + 沿变量维度混合 (Feature Mix)
# ==============================================================================
class MixerBlock(nn.Module):
    def __init__(self, channels, seq_len, dropout=0.1):
        super(MixerBlock, self).__init__()
        # Time Mixer (作用于序列长度维度)
        self.norm1 = nn.LayerNorm(channels)
        self.time_mixer = nn.Sequential(
            nn.Linear(seq_len, seq_len * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len * 2, seq_len)
        )
        
        # Feature Mixer (作用于变量通道维度)
        self.norm2 = nn.LayerNorm(channels)
        self.feature_mixer = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [B, seq_len, channels]
        # 1. Temporal Mixing: 需要将 seq_len 放到最后一维
        x_norm = self.norm1(x).transpose(1, 2)
        x_time = self.time_mixer(x_norm).transpose(1, 2)
        x = x + self.dropout(x_time)
        
        # 2. Feature Mixing: 直接作用于 channels 维
        x_norm = self.norm2(x)
        x_feat = self.feature_mixer(x_norm)
        x = x + self.dropout(x_feat)
        return x

# ==============================================================================
# 3. TimeMixer 满血版主网络 (Past Decomposable Mixing + Future Multipredictor Mixing)
# 融入轻量化瓶颈 MLP 融合，完美契合 All-MLP 哲学
# ==============================================================================
class TimeMixer(pl.LightningModule):
    def __init__(self, nwp_dim, num_classes, d_model=128, seq_len=48, pred_len=24, 
                 down_sampling_layers=2, dropout=0.2):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False 
        
        self.num_classes = num_classes
        self.nwp_dim = nwp_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.down_sampling_layers = down_sampling_layers
        
        # 异构变量对齐后的总通道数 M
        self.unified_dim = 1 + nwp_dim + 6
        
        # 序列分解
        self.decomp = series_decomp(kernel_size=25)
        
        # 多尺度降采样池化层 (例如：48 -> 24 -> 12)
        self.down_pools = nn.ModuleList([
            nn.AvgPool1d(kernel_size=2, stride=2) for _ in range(down_sampling_layers)
        ])
        
        # 记录每个尺度的序列长度 [48, 24, 12]
        self.scale_lens = [seq_len // (2 ** i) for i in range(down_sampling_layers + 1)]
        
        # 为每个尺度创建一个 MixerBlock
        self.mixers = nn.ModuleList([
            MixerBlock(channels=self.unified_dim, seq_len=sl, dropout=dropout) 
            for sl in self.scale_lens
        ])
        
        # FMM (Future Multipredictor Mixing): 为每个尺度独立投影到 pred_len
        self.predictors = nn.ModuleList([
            nn.Linear(sl, pred_len) for sl in self.scale_lens
        ])
        
        # ======================================================================
        # ★ 核心改进：TimeMixer 专属轻量化瓶颈 MLP 融合头 ★
        # ======================================================================
        # 流 A：仅仅映射历史主功率这一条通道的预测结果
        self.hist_projector = nn.Sequential(
            nn.Linear(pred_len, 128),
            nn.GELU()
        )
        
        # 流 B：提取纯净的未来 NWP 数据，降维对齐
        nwp_feature_count = nwp_dim + 6
        self.nwp_projector = nn.Sequential(
            nn.Linear(self.pred_len * nwp_feature_count, 128),
            nn.GELU()
        )
        
        # 流 C：潜空间联合解码，完美契合 TimeMixer 的 All-MLP 特性
        self.joint_decoder = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, pred_len)
        )

    def forward(self, ts, nwp):
        B, seq_len, _ = ts.shape
        
        # ==============================================================
        # 1. 异构数据对齐 (保证公平对齐)
        # ==============================================================
        ts_unified = torch.cat([ts[:, :, 0:1], torch.zeros(B, seq_len, self.nwp_dim, device=ts.device), ts[:, :, 1:7]], dim=-1)

        # ==============================================================
        # 2. PDM (Past Decomposable Mixing) - 多尺度构建与混合
        # ==============================================================
        # 构建多尺度输入集合: [原始序列, 1/2降采样, 1/4降采样]
        x_scales = [ts_unified]
        for pool in self.down_pools:
            # AvgPool1d 作用于时间维度，因此需要先 transpose
            x_down = pool(x_scales[-1].transpose(1, 2)).transpose(1, 2)
            x_scales.append(x_down)
            
        mixed_outputs = []
        for i in range(len(self.scale_lens)):
            x_i = x_scales[i]
            
            # 提取季节性 (res) 和趋势 (trend)
            season, trend = self.decomp(x_i)
            
            # 对微观的季节性特征进行纯 MLP 的时空混合
            mixed_season = self.mixers[i](season)
            
            # 重构：混合后的季节性 + 原始宏观趋势
            mixed_x = mixed_season + trend
            mixed_outputs.append(mixed_x)
            
        # ==============================================================
        # 3. FMM (Future Multipredictor Mixing) - 多尺度结果融合
        # ==============================================================
        pred_history = torch.zeros(B, self.pred_len, self.unified_dim, device=ts.device)
        
        for i in range(len(self.scale_lens)):
            # mixed_outputs[i] shape: [B, scale_lens[i], M]
            # transpose 后用 Linear 映射时间维度到 pred_len
            pred_i = self.predictors[i](mixed_outputs[i].transpose(1, 2)).transpose(1, 2)
            # 直接将各尺度的预测相加 (类似于从宏观到微观的残差补充)
            pred_history = pred_history + pred_i

        # ==============================================================
        # 4. ★ 改进的轻量化双流融合 (潜空间对齐) ★
        # ==============================================================
        # 仅仅抽取出主功率通道 (索引 0) 的历史多尺度推演结果
        p_hist_feat = pred_history[:, :, 0] # [B, pred_len]
        
        # 历史特征潜空间映射
        hist_emb = self.hist_projector(p_hist_feat) # [B, 128]
        
        # NWP 展平并映射到潜空间 (直接使用最纯净的 nwp 参数，无补零)
        flat_nwp = nwp.flatten(start_dim=1)         # [B, pred_len * nwp_feature_count]
        nwp_emb = self.nwp_projector(flat_nwp)      # [B, 128]
        
        # 潜空间特征级联
        fused_emb = torch.cat([hist_emb, nwp_emb], dim=-1) # [B, 256]
        
        # 最终 MLP 解码 (完美符合 TimeMixer 架构天性)
        p_pred = self.joint_decoder(fused_emb)      # [B, pred_len]

        # ==============================================================
        # 5. MTL 框架兼容伪造占位符
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
            loss_main=loss_p, 
            loss_dfi=loss_p,   
            loss_w=loss_p,     
            loss_ramp=loss_p,  
            p_sh_dict=p_sh_dict, 
            decay_factor=1.0
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