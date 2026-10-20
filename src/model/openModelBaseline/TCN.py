import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn.utils import weight_norm

# 导入你完美封装的单任务适配器
from src.optimizer.BaselineOptimizer import Standard_MTLOptimizer

# ==============================================================================
# 1. TCN 核心组件：因果截断与残差块
# ==============================================================================
class Chomp1d(nn.Module):
    """
    因果卷积的核心：裁剪掉 padding 带来的多余未来信息，确保严格的因果律
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # 第一次因果膨胀卷积
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.GELU() # 替换原版 ReLU 为更现代的 GELU，对齐其他 SOTA
        self.dropout1 = nn.Dropout(dropout)

        # 第二次因果膨胀卷积
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # 1x1 卷积用于残差连接的通道对齐
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.GELU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            # 因果卷积的 padding 计算
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                        padding=padding, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ==============================================================================
# 2. TCN 满血版主网络 (方案 B：架构自适应潜空间融合)
# ==============================================================================
class TCN(pl.LightningModule):
    def __init__(self, nwp_dim, num_classes, seq_len=48, pred_len=24, 
                 num_channels=[64, 128, 256], kernel_size=3, dropout=0.2):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False 
        
        self.num_classes = num_classes
        self.nwp_dim = nwp_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 异构变量对齐后的总通道数 M
        self.unified_dim = 1 + nwp_dim + 6
        
        # 1. TCN 骨干网络
        self.tcn_backbone = TemporalConvNet(
            num_inputs=self.unified_dim, 
            num_channels=num_channels, 
            kernel_size=kernel_size, 
            dropout=dropout
        )
        
        # ======================================================================
        # 2. ★ 核心改进：TCN 专属的靶向瓶颈潜空间融合 ★
        # ======================================================================
        # 流 A：历史演化流。TCN 最后一层输出通道数为 num_channels[-1] (如 256)
        # 我们只取最后一个时间步，将其映射到 128 维潜空间
        tcn_out_dim = num_channels[-1]
        self.hist_projector = nn.Sequential(
            nn.Linear(tcn_out_dim, 128),
            nn.GELU()
        )
        
        # 流 B：气象约束流。展平 NWP 数据，映射到 128 维潜空间
        nwp_feature_count = nwp_dim + 6
        self.nwp_projector = nn.Sequential(
            nn.Linear(self.pred_len * nwp_feature_count, 128),
            nn.GELU()
        )
        
        # 流 C：联合解码器。在潜空间拼接后进行联合决策
        self.joint_decoder = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, pred_len)
        )

    def forward(self, ts, nwp):
        B, seq_len, _ = ts.shape
        
        # ==============================================================
        # 1. 异构数据对齐 (保证所有模型的输入规范绝对一致)
        # ==============================================================
        ts_unified = torch.cat([ts[:, :, 0:1], torch.zeros(B, seq_len, self.nwp_dim, device=ts.device), ts[:, :, 1:7]], dim=-1)

        # ==============================================================
        # 2. TCN 时序特征提取 (坚守因果律)
        # ==============================================================
        # TCN 的输入规范为 [Batch, Channels, Seq_Len]
        x_enc = ts_unified.permute(0, 2, 1).contiguous()
        
        # tcn_out shape: [B, num_channels[-1], seq_len]
        tcn_out = self.tcn_backbone(x_enc)
        
        # ==============================================================
        # 3. 瓶颈特征融合与联合解码
        # ==============================================================
        # 提取最后一个时间步 (包含了整个序列的最大历史感受野信息)
        # hist_feat shape: [B, num_channels[-1]]
        hist_feat = tcn_out[:, :, -1] 
        
        # 历史特征映射到潜空间
        hist_emb = self.hist_projector(hist_feat) # [B, 128]
        
        # NWP 展平并映射到潜空间
        flat_nwp = nwp.flatten(start_dim=1)       # [B, pred_len * nwp_feature_count]
        nwp_emb = self.nwp_projector(flat_nwp)    # [B, 128]
        
        # 潜空间级联与解码
        fused_emb = torch.cat([hist_emb, nwp_emb], dim=-1) # [B, 256]
        p_pred = self.joint_decoder(fused_emb)             # [B, pred_len]

        # ==============================================================
        # 4. MTL 框架兼容伪造占位符
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