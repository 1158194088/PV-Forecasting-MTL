import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math

from src.optimizer.BaselineOptimizer import Standard_MTLOptimizer

# ==========================================
# 1. 时序专属嵌入 (Data Embedding)
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

# ==========================================
# 2. 概率稀疏注意力 (ProbSparse Attention)
# ==========================================
class ProbAttention(nn.Module):
    def __init__(self, factor=5, attention_dropout=0.1):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, mask=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        # 采样 Q 以计算 MMD (Max-Mean Divergence)
        U_part = int(self.factor * math.ceil(math.log(L_K)))
        u = min(U_part, L_Q)
        
        # 随机采样 Key 进行稀疏评估
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
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attn = self.dropout(torch.softmax(scores / math.sqrt(D), dim=-1))
        out_reduce = torch.einsum("bhqk,bkhd->bqhd", [attn, values])
        
        out = torch.zeros([B, L_Q, H, D], device=queries.device)
        V_mean = values.mean(dim=1)
        for i in range(B):
            for j in range(H):
                out[i, :, j, :] = V_mean[i, j, :]
                out[i, M_top[i, j], j, :] = out_reduce[i, :, j, :]
                
        return out.contiguous()

# ==========================================
# 3. 蒸馏与网络层 
# ==========================================
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
# 4. 满血版 Informer 主模型 (纯净版)
# ==========================================
class Informer(pl.LightningModule):
    def __init__(self, nwp_dim, num_classes, d_model=128, nhead=4, seq_len=48, label_len=24, pred_len=24):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False 
        self.num_classes = num_classes
        self.nwp_dim = nwp_dim
        self.label_len = label_len
        self.pred_len = pred_len
        
        self.unified_dim = 1 + nwp_dim + 6
        
        # 时序嵌入
        self.enc_embedding = DataEmbedding(self.unified_dim, d_model)
        self.dec_embedding = DataEmbedding(self.unified_dim, d_model)
        
        # 概率稀疏编码器
        self.enc_qkv = nn.Linear(d_model, d_model * 3)
        self.prob_attention = ProbAttention(factor=5)
        self.enc_ffn1 = nn.Sequential(nn.Linear(d_model, d_model*2), nn.GELU(), nn.Linear(d_model*2, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.distill = ConvLayer(d_model)
        
        self.enc_qkv2 = nn.Linear(d_model, d_model * 3)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        
        # 标准解码器
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, batch_first=True), num_layers=2
        )
        self.head_power = nn.Linear(d_model, 1)

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
        
        # --- 1. 异构数据统一化 ---
        ts_unified = torch.cat([
            ts[:, :, 0:1], 
            torch.zeros(B, seq_len, self.nwp_dim, device=ts.device), 
            ts[:, :, 1:7]
        ], dim=-1)

        nwp_unified = torch.cat([
            torch.zeros(B, self.pred_len, 1, device=nwp.device), 
            nwp[:, :, 0:self.nwp_dim], 
            nwp[:, :, self.nwp_dim:]
        ], dim=-1)

        # 数据已在外部归一化，直接使用对齐后的张量
        x_enc = ts_unified
        label_len = min(self.label_len, seq_len)
        
        # 构造 Decoder 输入：尾部真实标签片段 + 预测位用0或特征填充
        dec_token_zeros = torch.zeros(B, self.pred_len, self.unified_dim, device=ts.device)
        dec_token_zeros[:, :, 1:1+self.nwp_dim] = nwp[:, :, 0:self.nwp_dim] 
        dec_token_zeros[:, :, 1+self.nwp_dim:] = nwp[:, :, self.nwp_dim:]
        
        x_dec = torch.cat([ts_unified[:, -label_len:, :], dec_token_zeros], dim=1)

        # --- 2. 嵌入 ---
        enc_emb = self.enc_embedding(x_enc)
        dec_emb = self.dec_embedding(x_dec)

        # --- 3. 满血 Informer 编码器 (ProbSparse + Distill) ---
        enc_out = self.run_encoder_layer(enc_emb, self.enc_qkv, self.norm1, self.norm2)
        enc_out = self.distill(enc_out)
        memory = self.run_encoder_layer(enc_out, self.enc_qkv2, self.norm3, self.norm4)

        # --- 4. 解码与预测 ---
        dec_out = self.decoder(dec_emb, memory)
        p_pred_total = self.head_power(dec_out)
        
        # 截取预测窗口并移除最后一维，形状转为 [B, pred_len]
        p_pred = p_pred_total[:, -self.pred_len:, 0]

        # --- 5. MTL 兼容占位符 ---
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