import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import math

from src.optimizer.RAIS_NashOptimizer import RAIS_NashOptimizer

# ==========================================
# 1. Informer 基础组件 (时序嵌入, 注意力, 蒸馏)
# ==========================================
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        nn.init.kaiming_normal_(self.tokenConv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)

class ProbAttention(nn.Module):
    def __init__(self, factor=5, attention_dropout=0.1):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, mask=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        U_part = int(self.factor * math.ceil(math.log(L_K)))
        u = min(U_part, L_Q)
        sample_k = min(U_part, L_K)
        index_sample = torch.randint(L_K, (sample_k,))
        K_sample = keys[:, index_sample, :, :]
        Q_K_sample = torch.einsum("bqhd,bkhd->bhqk", [queries, K_sample])
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(u, dim=-1)[1]
        
        Q_reduce = torch.zeros([B, u, H, D], device=queries.device)
        for i in range(B):
            for j in range(H):
                Q_reduce[i, :, j, :] = queries[i, M_top[i, j], j, :]
                
        scores = torch.einsum("bqhd,bkhd->bhqk", [Q_reduce, keys])
        if mask is not None: scores.masked_fill_(mask == 0, -1e9)
        attn = self.dropout(torch.softmax(scores / math.sqrt(D), dim=-1))
        out_reduce = torch.einsum("bhqk,bkhd->bqhd", [attn, values])
        
        out = torch.zeros([B, L_Q, H, D], device=queries.device)
        V_mean = values.mean(dim=1)
        for i in range(B):
            for j in range(H):
                out[i, :, j, :] = V_mean[i, j, :]
                out[i, M_top[i, j], j, :] = out_reduce[i, :, j, :]
        return out.contiguous()

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in, out_channels=c_in, kernel_size=3, padding=1, padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        return x.permute(0, 2, 1)


# ==========================================
# 2. Informer 骨干 + 四任务 MoE 融合网络
# ==========================================
class Informer_QuadTask_Nash(pl.LightningModule):
    def __init__(self, nwp_dim, num_classes, d_model=128, nhead=4, seq_len=48, label_len=24, pred_len=24):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False 
        self.num_classes = num_classes
        self.nwp_dim = nwp_dim
        self.label_len = label_len
        self.pred_len = pred_len
        self.unified_dim = 1 + nwp_dim + 6
        
        # 历史指标EMA缓存 (来自你的 MTL)
        self.register_buffer('ema_cos_md', torch.tensor(0.0)) 
        self.register_buffer('ema_cos_mw', torch.tensor(0.0)) 
        self.register_buffer('ema_cos_mr', torch.tensor(0.0)) 

        # ---- Informer 骨干 (Encoder-Decoder) ----
        self.enc_embedding = DataEmbedding(self.unified_dim, d_model)
        self.dec_embedding = DataEmbedding(self.unified_dim, d_model)
        
        self.enc_qkv = nn.Linear(d_model, d_model * 3)
        self.prob_attention = ProbAttention(factor=5)
        self.enc_ffn1 = nn.Sequential(nn.Linear(d_model, d_model*2), nn.GELU(), nn.Linear(d_model*2, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.distill = ConvLayer(d_model)
        
        self.enc_qkv2 = nn.Linear(d_model, d_model * 3)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, batch_first=True), num_layers=2
        )

        # ---- 四任务 MoE 专家与路由 (来自你的 MTL) ----
        self.expert_shared = nn.Sequential(nn.Linear(d_model, d_model * 3), nn.GELU(), nn.Dropout(0.2), nn.Linear(d_model * 3, d_model))
        self.expert_main = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(0.15), nn.Linear(d_model, d_model))
        self.expert_aux_ramp = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(0.15), nn.Linear(d_model * 2, d_model))
        self.expert_aux_dfi = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(0.15), nn.Linear(d_model // 2, d_model))
        self.expert_aux_w = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(0.15), nn.Linear(d_model // 2, d_model))
        
        self.gate_main = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 2))
        self.gate_aux_dfi = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 2))
        self.gate_aux_w = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 2))
        self.gate_aux_ramp = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 2))

        # ---- 四任务特定预测头 ----
        flat_dim = d_model * pred_len
        self.head_power = nn.Sequential(nn.Flatten(), nn.Linear(flat_dim, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, pred_len))
        self.head_dfi = nn.Sequential(nn.Flatten(), nn.Linear(flat_dim, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, pred_len))
        self.head_weather = nn.Sequential(nn.Flatten(), nn.Linear(flat_dim, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, num_classes))
        self.head_ramp = nn.Sequential(nn.Flatten(), nn.Linear(flat_dim, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, pred_len * 5)) 

    def run_encoder_layer(self, x, qkv_layer, norm_a, norm_b):
        B, L, _ = x.shape
        qkv = qkv_layer(x).reshape(B, L, 3, 4, -1).permute(2, 0, 1, 3, 4) 
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_out = self.prob_attention(q, k, v)
        attn_out = attn_out.view(B, L, -1)
        x = norm_a(x + attn_out)
        x = norm_b(x + self.enc_ffn1(x))
        return x

    def forward(self, ts, nwp):
        B, seq_len, _ = ts.shape
        
        # 1. 异构数据统一化 (数据已在外部归一化，直接使用)
        ts_unified = torch.cat([ts[:, :, 0:1], torch.zeros(B, seq_len, self.nwp_dim, device=ts.device), ts[:, :, 1:7]], dim=-1)
        nwp_unified = torch.cat([torch.zeros(B, self.pred_len, 1, device=nwp.device), nwp[:, :, 0:self.nwp_dim], nwp[:, :, self.nwp_dim:]], dim=-1)
        
        x_enc = ts_unified
        
        label_len = min(self.label_len, seq_len)
        dec_token_zeros = torch.zeros(B, self.pred_len, self.unified_dim, device=ts.device)
        dec_token_zeros[:, :, 1:1+self.nwp_dim] = nwp[:, :, 0:self.nwp_dim] 
        dec_token_zeros[:, :, 1+self.nwp_dim:] = nwp[:, :, self.nwp_dim:]
        x_dec = torch.cat([ts_unified[:, -label_len:, :], dec_token_zeros], dim=1)

        # 2. Informer 嵌入与骨干处理
        enc_emb = self.enc_embedding(x_enc)
        dec_emb = self.dec_embedding(x_dec)

        enc_out = self.run_encoder_layer(enc_emb, self.enc_qkv, self.norm1, self.norm2)
        enc_out = self.distill(enc_out)
        memory = self.run_encoder_layer(enc_out, self.enc_qkv2, self.norm3, self.norm4)

        dec_out = self.decoder(dec_emb, memory)
        # 截取预测部分的隐藏状态作为多任务的底座输入 [B, pred_len, d_model]
        bottom_out = dec_out[:, -self.pred_len:, :]

        # 3. RAIS MoE 专家路由 (复用多任务逻辑)
        out_shared = self.expert_shared(bottom_out)
        out_main = self.expert_main(bottom_out)
        out_dfi = self.expert_aux_dfi(bottom_out)
        out_w = self.expert_aux_w(bottom_out)
        out_ramp = self.expert_aux_ramp(bottom_out) 
        
        gate_input = torch.mean(bottom_out, dim=1)
        gl_m = self.gate_main(gate_input)
        gl_d = self.gate_aux_dfi(gate_input)
        gl_w = self.gate_aux_w(gate_input)
        gl_r = self.gate_aux_ramp(gate_input)
        
        gamma = torch.sigmoid(torch.tensor((self.current_epoch - 15.0) / 3.0, device=self.device)).item()
        p_sh_m = F.softmax(gl_m, dim=-1)[:, 0].detach()
        p_sh_d = F.softmax(gl_d, dim=-1)[:, 0].detach()
        p_sh_w = F.softmax(gl_w, dim=-1)[:, 0].detach()
        p_sh_r = F.softmax(gl_r, dim=-1)[:, 0].detach()
        
        gl_d[:, 0] += gamma * torch.tanh(self.ema_cos_md) * torch.where(self.ema_cos_md > 0, 1. - p_sh_d, p_sh_d)
        gl_w[:, 0] += gamma * torch.tanh(self.ema_cos_mw) * torch.where(self.ema_cos_mw > 0, 1. - p_sh_w, p_sh_w)
        gl_r[:, 0] += gamma * torch.tanh(self.ema_cos_mr) * torch.where(self.ema_cos_mr > 0, 1. - p_sh_r, p_sh_r)
        
        epsilon = 0.1
        prior = torch.tensor([0.5, 0.5], device=self.device)
        
        w_m = (1 - epsilon) * F.softmax(gl_m, dim=-1) + epsilon * prior
        w_d = (1 - epsilon) * F.softmax(gl_d, dim=-1) + epsilon * prior
        w_w = (1 - epsilon) * F.softmax(gl_w, dim=-1) + epsilon * prior
        w_r = (1 - epsilon) * F.softmax(gl_r, dim=-1) + epsilon * prior
        
        p_sh_dict = {'m': w_m[:, 0].mean().item(), 'd': w_d[:, 0].mean().item(), 'w': w_w[:, 0].mean().item(), 'r': w_r[:, 0].mean().item()}
        
        fused_main = w_m[:, 0:1, None] * out_shared + w_m[:, 1:2, None] * out_main
        fused_dfi = w_d[:, 0:1, None] * out_shared + w_d[:, 1:2, None] * out_dfi
        fused_w = w_w[:, 0:1, None] * out_shared + w_w[:, 1:2, None] * out_w
        fused_ramp = w_r[:, 0:1, None] * out_shared + w_r[:, 1:2, None] * out_ramp 
        
        # 4. 预测头计算 (直接输出归一化尺度的预测结果)
        p_pred = self.head_power(fused_main)
        dfi_pred = self.head_dfi(fused_dfi)
        w_logits = self.head_weather(fused_w)
        ramp_logits = self.head_ramp(fused_ramp).view(-1, self.pred_len, 5)
        
        return p_pred, dfi_pred, w_logits, ramp_logits, p_sh_dict, w_w, w_r

    def configure_optimizers(self):
        base_opt = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)
        
        # 移除了 self.revin.parameters()
        shared_params = list(self.enc_embedding.parameters()) + \
                        list(self.dec_embedding.parameters()) + list(self.prob_attention.parameters()) + \
                        list(self.enc_qkv.parameters()) + list(self.distill.parameters()) + \
                        list(self.decoder.parameters()) + list(self.expert_shared.parameters())
                        
        main_params = list(self.expert_main.parameters()) + list(self.gate_main.parameters()) + list(self.head_power.parameters())
        dfi_params = list(self.expert_aux_dfi.parameters()) + list(self.gate_aux_dfi.parameters()) + list(self.head_dfi.parameters())
        w_params = list(self.expert_aux_w.parameters()) + list(self.gate_aux_w.parameters()) + list(self.head_weather.parameters())
        ramp_params = list(self.expert_aux_ramp.parameters()) + list(self.gate_aux_ramp.parameters()) + list(self.head_ramp.parameters())
        
        self.nash_optimizer = RAIS_NashOptimizer(base_opt, shared_params, main_params, dfi_params, w_params, ramp_params) 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(base_opt, T_0=15, T_mult=2, eta_min=1e-6)
        return [base_opt], [scheduler]

    def training_step(self, batch, batch_idx):
        opt = self.nash_optimizer 
        p_pred, dfi_pred, w_logits, ramp_logits, p_sh_dict, w_w, w_r = self(batch['ts_input'], batch['nwp_input'])
        
        loss_p_raw = F.huber_loss(p_pred, batch['target_p'], delta=0.5)
        loss_dfi_raw = F.mse_loss(dfi_pred, batch['target_dfi'])
        loss_w_raw = F.cross_entropy(w_logits, batch['daily_weather'])
        loss_ramp_raw = F.cross_entropy(ramp_logits.view(-1, 5), batch['target_ramp'].view(-1))
        
        TARGET_MAX_USAGE = 0.40
        PENALTY_WEIGHT = 3.0
        penalty_w = F.relu(w_w[:, 0] - TARGET_MAX_USAGE).mean() * PENALTY_WEIGHT
        penalty_r = F.relu(w_r[:, 0] - TARGET_MAX_USAGE).mean() * PENALTY_WEIGHT
        loss_w_raw += penalty_w
        loss_ramp_raw += penalty_r

        decay_factor = 0.2 + 0.8 * 0.5 * (1 + np.cos(np.pi * self.current_epoch / self.trainer.max_epochs)) 
        metrics = opt.backward(loss_p_raw, loss_dfi_raw, loss_w_raw, loss_ramp_raw, p_sh_dict, decay_factor)
        
        self.ema_cos_md = 0.9 * self.ema_cos_md + 0.1 * metrics['cos_md']
        self.ema_cos_mw = 0.9 * self.ema_cos_mw + 0.1 * metrics['cos_mw']
        self.ema_cos_mr = 0.9 * self.ema_cos_mr + 0.1 * metrics['cos_mr']
        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        opt.step()
        if self.trainer.is_last_batch: self.lr_schedulers().step()

        self.log('train_loss_p', loss_p_raw, prog_bar=True)
        return loss_p_raw

    def validation_step(self, batch, batch_idx):
        p_pred, _, _, _, _, _, _ = self(batch['ts_input'], batch['nwp_input'])
        mae = F.l1_loss(p_pred, batch['target_p'])
        self.log('val_mae', mae, on_epoch=True, prog_bar=True) 
        return mae