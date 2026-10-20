import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

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

# ==============================================================================
# ★ 基线模型: MMoE (多门控混合专家) + 平等加权 (Equal Weighting)
# ==============================================================================
class PVTransformer_QuadTask_Advanced(pl.LightningModule):
    def __init__(self, nwp_dim, num_classes, d_model=128, nhead=4, patch_len=8, stride=4, pred_len=24):
        super().__init__()
        self.save_hyperparameters()
        
        # 恢复自动优化机制，完全交由 Lightning 处理反向传播
        self.automatic_optimization = True

        self.ts_patcher = TokenEmbedding(input_dim=7, d_model=d_model)
        self.nwp_patcher = TokenEmbedding(input_dim=nwp_dim+6, d_model=d_model)
        
        self.pos_enc_ts = nn.Parameter(torch.randn(1, 100, d_model) * 0.1)
        self.pos_enc_nwp = nn.Parameter(torch.randn(1, 100, d_model) * 0.1)
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=0.1, batch_first=True)
        self.fusion_norm = nn.LayerNorm(d_model)
        
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True, dropout=0.2, norm_first=True)
        self.shared_bottom = nn.TransformerEncoder(layer, num_layers=2)
        
        # ======================================================================
        # ★ 核心修改 1: MMoE 的统一专家池 (Unified Expert Pool)
        # 废除专属/共享专家的隔离概念，建立 5 个容量一致的通用专家公共池
        # ======================================================================
        self.num_experts = 5
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2), 
                nn.GELU(), 
                nn.Dropout(0.15), 
                nn.Linear(d_model * 2, d_model)
            ) for _ in range(self.num_experts)
        ])
        
        # ======================================================================
        # ★ 核心修改 2: MMoE 的多门控网络 (Multi-Gate)
        # 每个任务的门控输出维度必须等于专家总数 (self.num_experts = 5)
        # ======================================================================
        self.gate_main = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, self.num_experts))
        self.gate_aux_dfi = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, self.num_experts))
        self.gate_aux_w = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, self.num_experts))
        self.gate_aux_ramp = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, self.num_experts))
        
        flat_dim = d_model * pred_len
        
        # 独立预测头 (知识孤岛，不进行 CA-FiLM 跨流交互)
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
        
        # ======================================================================
        # ★ 核心修改 3: 获取所有专家的输出并堆叠
        # expert_outputs 维度从 [B, L, d_model] * 5 变为 [B, L, 5, d_model]
        # ======================================================================
        expert_outputs = [expert(bottom_out) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=2) 
        
        # 门控路由计算
        gate_input = torch.mean(bottom_out, dim=1)  # [B, d_model]
        w_main = F.softmax(self.gate_main(gate_input), dim=-1)   # [B, 5]
        w_dfi = F.softmax(self.gate_aux_dfi(gate_input), dim=-1) # [B, 5]
        w_w = F.softmax(self.gate_aux_w(gate_input), dim=-1)     # [B, 5]
        w_ramp = F.softmax(self.gate_aux_ramp(gate_input), dim=-1) # [B, 5]
        
        # ======================================================================
        # ★ 核心修改 4: 广播机制实现特征动态加权融合
        # w_main.unsqueeze(1).unsqueeze(-1) 变成 [B, 1, 5, 1] 
        # 与 expert_outputs [B, L, 5, d_model] 相乘后，在专家维度(dim=2)求和
        # ======================================================================
        fused_main = torch.sum(expert_outputs * w_main.unsqueeze(1).unsqueeze(-1), dim=2)
        fused_dfi = torch.sum(expert_outputs * w_dfi.unsqueeze(1).unsqueeze(-1), dim=2)
        fused_w = torch.sum(expert_outputs * w_w.unsqueeze(1).unsqueeze(-1), dim=2)
        fused_ramp = torch.sum(expert_outputs * w_ramp.unsqueeze(1).unsqueeze(-1), dim=2) 
        
        ramp_logits = self.head_ramp(fused_ramp).view(-1, 24, 5) 

        # 记录每项任务分配给 0 号专家的权重（用于监控 MMoE 门控状态）
        p_sh_dict = {
            'm': w_main[:, 0].mean().item(), 
            'd': w_dfi[:, 0].mean().item(),
            'w': w_w[:, 0].mean().item(), 
            'r': w_ramp[:, 0].mean().item()
        }

        return self.head_power(fused_main), self.head_dfi(fused_dfi), self.head_weather(fused_w), ramp_logits, p_sh_dict

    def configure_optimizers(self):
        # 标准优化器配置
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # 接收前向传播特征
        p_pred, dfi_pred, w_logits, ramp_logits, p_sh_dict = self(batch['ts_input'], batch['nwp_input'])
        
        # 计算独立任务损失
        loss_p = F.huber_loss(p_pred, batch['target_p'], delta=0.5)
        loss_dfi = F.mse_loss(dfi_pred, batch['target_dfi'])
        loss_w = F.cross_entropy(w_logits, batch['daily_weather'])
        loss_ramp = F.cross_entropy(ramp_logits.view(-1, 5), batch['target_ramp'].view(-1))
        
        # ==============================================================
        # ★ 痛点展示区: 纯粹的静态平等加权 (Equal Weighting)
        # 暴露出 MMoE 在量纲碾压下依然无法学好辅助特征的缺陷
        # ==============================================================
        total_loss = loss_p + loss_dfi + loss_w + loss_ramp
        
        # 日志记录
        self.log('train_loss_total', total_loss, prog_bar=True)
        self.log('1_Loss/Main_Power', loss_p, prog_bar=False)
        self.log('1_Loss/Aux_DFI', loss_dfi, prog_bar=False)
        self.log('1_Loss/Aux_Weather', loss_w, prog_bar=False)
        self.log('1_Loss/Aux_Ramp', loss_ramp, prog_bar=False)
        
        # 监控门控行为
        self.log('6_Gate/Main_use_Exp0', p_sh_dict['m'], prog_bar=False)
        self.log('6_Gate/Weather_use_Exp0', p_sh_dict['w'], prog_bar=False)
        self.log('6_Gate/Ramp_use_Exp0', p_sh_dict['r'], prog_bar=False)
        self.log('6_Gate/DFI_use_Exp0', p_sh_dict['d'], prog_bar=False)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        p_pred, _, _, _, _ = self(batch['ts_input'], batch['nwp_input'])
        mae = F.l1_loss(p_pred, batch['target_p'])
        self.log('val_mae', mae, on_epoch=True, prog_bar=True) 
        return mae