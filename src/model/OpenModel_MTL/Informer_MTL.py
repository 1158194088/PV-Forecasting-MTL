import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.optimizer.NashOptimizer import NashMTLOptimizer


# ==============================================================================
# 1. Informer 时序嵌入模块：保持原始 Informer 基线逻辑
# ==============================================================================
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        nn.init.kaiming_normal_(self.tokenConv.weight, mode="fan_in", nonlinearity="leaky_relu")

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


# ==============================================================================
# 2. ProbSparse Attention：保持 Informer 主干注意力形式
# ==============================================================================
class ProbAttention(nn.Module):
    def __init__(self, factor=5, attention_dropout=0.1):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, mask=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        U_part = int(self.factor * math.ceil(math.log(max(L_K, 2))))
        u = min(U_part, L_Q)
        sample_k = min(U_part, L_K)

        # 与原始逻辑一致，只是把随机索引放到同一 device，避免 CUDA 索引问题。
        index_sample = torch.randint(L_K, (sample_k,), device=queries.device)
        K_sample = keys[:, index_sample, :, :]

        Q_K_sample = torch.einsum("bqhd,bkhd->bhqk", queries, K_sample)
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(u, dim=-1)[1]

        Q_reduce = torch.zeros(B, u, H, D, device=queries.device, dtype=queries.dtype)
        for i in range(B):
            for j in range(H):
                Q_reduce[i, :, j, :] = queries[i, M_top[i, j], j, :]

        scores = torch.einsum("bqhd,bkhd->bhqk", Q_reduce, keys)
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)

        attn = self.dropout(torch.softmax(scores / math.sqrt(D), dim=-1))
        out_reduce = torch.einsum("bhqk,bkhd->bqhd", attn, values)

        out = torch.zeros(B, L_Q, H, D, device=queries.device, dtype=queries.dtype)
        V_mean = values.mean(dim=1)
        for i in range(B):
            for j in range(H):
                out[i, :, j, :] = V_mean[i, j, :]
                out[i, M_top[i, j], j, :] = out_reduce[i, :, j, :]

        return out.contiguous()


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        return x.permute(0, 2, 1)


# ==============================================================================
# 3. Backbone-agnostic QuadTask module
# ==============================================================================
class QuadTaskTemporalCAFiLMHead(nn.Module):
    """
    输入 Informer decoder 的未来窗口隐状态 future_hidden: [B, pred_len, d_model]。
    不改变 Informer 的 Encoder/Decoder 主干，只替换原本的 head_power 为多任务模块。
    """
    def __init__(self, num_classes, d_model=128, pred_len=24, dropout=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.pred_len = pred_len

        # ------------------------------------------------------------------
        # CGC 异构共享-专属专家池
        # ------------------------------------------------------------------
        self.expert_shared = nn.Sequential(
            nn.Linear(d_model, d_model * 3),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model * 3, d_model),
        )
        self.expert_main = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model, d_model),
        )
        self.expert_aux_dfi = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model // 2, d_model),
        )
        self.expert_aux_w = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model // 2, d_model),
        )
        self.expert_aux_ramp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model * 2, d_model),
        )

        self.gate_main = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2),
        )
        self.gate_aux_dfi = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2),
        )
        self.gate_aux_w = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2),
        )
        self.gate_aux_ramp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2),
        )

        # ------------------------------------------------------------------
        # 任务形态感知辅助头
        # ------------------------------------------------------------------
        self.head_dfi = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1),
        )

        self.head_weather = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes),
        )

        self.head_ramp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 5),
        )

        # ------------------------------------------------------------------
        # All-Aux Temporal CA-FiLM: DFI(1) + Weather(C) + Ramp(5)
        # ------------------------------------------------------------------
        input_phys_step_dim = 1 + num_classes + 5
        self.phys_projector = nn.Sequential(
            nn.Linear(input_phys_step_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.film_generator = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model * 2),
        )
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)

        flat_dim = d_model * pred_len
        self.head_power = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, pred_len),
        )

    @staticmethod
    def _gate_fuse(gate, gate_input, shared_feat, task_feat):
        weights = F.softmax(gate(gate_input), dim=-1)
        fused = weights[:, 0:1, None] * shared_feat + weights[:, 1:2, None] * task_feat
        return fused, weights

    def forward(self, future_hidden):
        # future_hidden: [B, P, D]
        bottom_out = future_hidden

        out_shared = self.expert_shared(bottom_out)
        out_main = self.expert_main(bottom_out)
        out_dfi = self.expert_aux_dfi(bottom_out)
        out_w = self.expert_aux_w(bottom_out)
        out_ramp = self.expert_aux_ramp(bottom_out)

        gate_input = torch.mean(bottom_out, dim=1)
        fused_main, w_main = self._gate_fuse(self.gate_main, gate_input, out_shared, out_main)
        fused_dfi, w_dfi = self._gate_fuse(self.gate_aux_dfi, gate_input, out_shared, out_dfi)
        fused_w, w_w = self._gate_fuse(self.gate_aux_w, gate_input, out_shared, out_w)
        fused_ramp, w_ramp = self._gate_fuse(self.gate_aux_ramp, gate_input, out_shared, out_ramp)

        dfi_pred = self.head_dfi(fused_dfi).squeeze(-1)       # [B, P]
        weather_context = torch.mean(fused_w, dim=1)          # [B, D]
        w_logits = self.head_weather(weather_context)         # [B, C]
        ramp_logits = self.head_ramp(fused_ramp)              # [B, P, 5]

        # 使用辅助预测值构造物理上下文，不使用真实标签，避免标签泄露。
        with torch.no_grad():
            dfi_context = dfi_pred.detach().unsqueeze(-1)
            weather_prob = F.softmax(w_logits.detach(), dim=-1)
            weather_context_step = weather_prob.unsqueeze(1).expand(-1, self.pred_len, -1)
            ramp_context = F.softmax(ramp_logits.detach(), dim=-1)
            phys_context_step = torch.cat([dfi_context, weather_context_step, ramp_context], dim=-1)

        phys_embedding_step = self.phys_projector(phys_context_step)
        combined_info = torch.cat([fused_main, phys_embedding_step], dim=-1)
        film_params = self.film_generator(combined_info)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)

        fused_main_refined = fused_main * (1.0 + gamma) + beta
        p_pred = self.head_power(fused_main_refined)

        p_sh_dict = {
            "m": w_main[:, 0].mean().detach().item(),
            "d": w_dfi[:, 0].mean().detach().item(),
            "w": w_w[:, 0].mean().detach().item(),
            "r": w_ramp[:, 0].mean().detach().item(),
        }

        return p_pred, dfi_pred, w_logits, ramp_logits, p_sh_dict, w_w, w_ramp


# ==============================================================================
# 4. Informer + QuadTask module 泛化验证版
# ==============================================================================
class Informer_QuadTask_Advanced(pl.LightningModule):
    def __init__(self, nwp_dim, num_classes, d_model=128, nhead=4,
                 seq_len=48, label_len=24, pred_len=24):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.num_classes = num_classes
        self.nwp_dim = nwp_dim
        self.nhead = nhead
        self.label_len = label_len
        self.pred_len = pred_len
        self.unified_dim = 1 + nwp_dim + 6

        # Informer 原始 embedding
        self.enc_embedding = DataEmbedding(self.unified_dim, d_model)
        self.dec_embedding = DataEmbedding(self.unified_dim, d_model)

        # Informer 原始 ProbSparse encoder + distill
        self.enc_qkv = nn.Linear(d_model, d_model * 3)
        self.prob_attention = ProbAttention(factor=5)
        self.enc_ffn1 = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.distill = ConvLayer(d_model)

        self.enc_qkv2 = nn.Linear(d_model, d_model * 3)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        # Informer 原始 decoder 保留不变
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, batch_first=True),
            num_layers=2,
        )

        # 用 QuadTask 模块替换原 head_power
        self.quadtask_head = QuadTaskTemporalCAFiLMHead(
            num_classes=num_classes,
            d_model=d_model,
            pred_len=pred_len,
            dropout=0.2,
        )

    def run_encoder_layer(self, x, qkv_layer, norm_a, norm_b):
        B, L, _ = x.shape
        qkv = qkv_layer(x).reshape(B, L, 3, self.nhead, -1).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_out = self.prob_attention(q, k, v)
        attn_out = attn_out.view(B, L, -1)
        x = norm_a(x + attn_out)
        x = norm_b(x + self.enc_ffn1(x))
        return x

    def forward(self, ts, nwp):
        B, seq_len, _ = ts.shape

        # ------------------------------------------------------------------
        # 1. 异构数据统一化：保持 Informer 基线不变
        # ------------------------------------------------------------------
        ts_unified = torch.cat(
            [
                ts[:, :, 0:1],
                torch.zeros(B, seq_len, self.nwp_dim, device=ts.device, dtype=ts.dtype),
                ts[:, :, 1:7],
            ],
            dim=-1,
        )

        # 保留原始代码中的 nwp_unified 构造语义；decoder 预测位使用未来 NWP 和未来时间特征。
        _ = torch.cat(
            [
                torch.zeros(B, self.pred_len, 1, device=nwp.device, dtype=nwp.dtype),
                nwp[:, :, 0:self.nwp_dim],
                nwp[:, :, self.nwp_dim:],
            ],
            dim=-1,
        )

        x_enc = ts_unified
        label_len = min(self.label_len, seq_len)

        dec_token_zeros = torch.zeros(B, self.pred_len, self.unified_dim, device=ts.device, dtype=ts.dtype)
        dec_token_zeros[:, :, 1:1 + self.nwp_dim] = nwp[:, :, 0:self.nwp_dim]
        dec_token_zeros[:, :, 1 + self.nwp_dim:] = nwp[:, :, self.nwp_dim:]
        x_dec = torch.cat([ts_unified[:, -label_len:, :], dec_token_zeros], dim=1)

        # ------------------------------------------------------------------
        # 2. Informer 主干：Embedding -> ProbSparse Encoder -> Distill -> Decoder
        # ------------------------------------------------------------------
        enc_emb = self.enc_embedding(x_enc)
        dec_emb = self.dec_embedding(x_dec)

        enc_out = self.run_encoder_layer(enc_emb, self.enc_qkv, self.norm1, self.norm2)
        enc_out = self.distill(enc_out)
        memory = self.run_encoder_layer(enc_out, self.enc_qkv2, self.norm3, self.norm4)

        dec_out = self.decoder(dec_emb, memory)
        future_hidden = dec_out[:, -self.pred_len:, :]  # [B, P, D]

        # ------------------------------------------------------------------
        # 3. QuadTask 多任务模块
        # ------------------------------------------------------------------
        return self.quadtask_head(future_hidden)

    def configure_optimizers(self):
        base_opt = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)

        shared_params = (
            list(self.enc_embedding.parameters()) +
            list(self.dec_embedding.parameters()) +
            list(self.enc_qkv.parameters()) +
            list(self.prob_attention.parameters()) +
            list(self.enc_ffn1.parameters()) +
            list(self.norm1.parameters()) +
            list(self.norm2.parameters()) +
            list(self.distill.parameters()) +
            list(self.enc_qkv2.parameters()) +
            list(self.norm3.parameters()) +
            list(self.norm4.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quadtask_head.expert_shared.parameters())
        )

        main_params = (
            list(self.quadtask_head.expert_main.parameters()) +
            list(self.quadtask_head.gate_main.parameters()) +
            list(self.quadtask_head.phys_projector.parameters()) +
            list(self.quadtask_head.film_generator.parameters()) +
            list(self.quadtask_head.head_power.parameters())
        )

        dfi_params = (
            list(self.quadtask_head.expert_aux_dfi.parameters()) +
            list(self.quadtask_head.gate_aux_dfi.parameters()) +
            list(self.quadtask_head.head_dfi.parameters())
        )

        w_params = (
            list(self.quadtask_head.expert_aux_w.parameters()) +
            list(self.quadtask_head.gate_aux_w.parameters()) +
            list(self.quadtask_head.head_weather.parameters())
        )

        ramp_params = (
            list(self.quadtask_head.expert_aux_ramp.parameters()) +
            list(self.quadtask_head.gate_aux_ramp.parameters()) +
            list(self.quadtask_head.head_ramp.parameters())
        )

        self.nash_optimizer = NashMTLOptimizer(
            base_opt,
            shared_params,
            main_params,
            dfi_params,
            w_params,
            ramp_params,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            base_opt,
            T_0=15,
            T_mult=2,
            eta_min=1e-6,
        )
        return [base_opt], [scheduler]

    def training_step(self, batch, batch_idx):
        opt = self.nash_optimizer
        p_pred, dfi_pred, w_logits, ramp_logits, p_sh_dict, w_w, w_r = self(
            batch["ts_input"],
            batch["nwp_input"],
        )

        loss_p = F.huber_loss(p_pred, batch["target_p"], delta=0.5)
        loss_dfi = F.mse_loss(dfi_pred, batch["target_dfi"])
        loss_w = F.cross_entropy(w_logits, batch["daily_weather"])
        loss_ramp = F.cross_entropy(
            ramp_logits.reshape(-1, 5),
            batch["target_ramp"].reshape(-1),
        )

        # RouteReg：与主模型保持一致，仅约束 Weather/Ramp 不过度占用共享专家。
        penalty_w = F.relu(w_w[:, 0] - 0.40).mean() * 3.0
        penalty_r = F.relu(w_r[:, 0] - 0.40).mean() * 3.0
        loss_w = loss_w + penalty_w
        loss_ramp = loss_ramp + penalty_r

        metrics = opt.backward(loss_p, loss_dfi, loss_w, loss_ramp)

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        opt.step()
        if self.trainer.is_last_batch:
            self.lr_schedulers().step()

        self.log("train_loss_p", loss_p, prog_bar=True)
        self.log("1_Loss/Aux_DFI_Raw", loss_dfi, prog_bar=False)
        self.log("1_Loss/Aux_Weather_Raw", loss_w, prog_bar=False)
        self.log("1_Loss/Aux_Ramp_Raw", loss_ramp, prog_bar=False)

        self.log("6_Gate/Main_use_Shared", p_sh_dict["m"], prog_bar=False)
        self.log("6_Gate/DFI_use_Shared", p_sh_dict["d"], prog_bar=False)
        self.log("6_Gate/Weather_use_Shared", p_sh_dict["w"], prog_bar=False)
        self.log("6_Gate/Ramp_use_Shared", p_sh_dict["r"], prog_bar=False)

        self.log("3_Nash/CosTheta_Overall_Combined", metrics["cos_combined"], prog_bar=True)
        self.log("7_Gradient_Angle/Cos_Main_vs_DFI", metrics["cos_md"], prog_bar=False)
        self.log("7_Gradient_Angle/Cos_Main_vs_Weather", metrics["cos_mw"], prog_bar=False)
        self.log("7_Gradient_Angle/Cos_Main_vs_Ramp", metrics["cos_mr"], prog_bar=False)

        return loss_p

    def validation_step(self, batch, batch_idx):
        p_pred, _, _, _, _, _, _ = self(batch["ts_input"], batch["nwp_input"])
        mae = F.l1_loss(p_pred, batch["target_p"])
        self.log("val_mae", mae, on_epoch=True, prog_bar=True)
        return mae


# 兼容不同训练脚本的导入命名
Informer = Informer_QuadTask_Advanced
PVTransformer_QuadTask_Advanced = Informer_QuadTask_Advanced
