import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


# ==============================================================================
# Point-wise Token Embedding
# ==============================================================================
class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.value_embedding = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, L, C] -> [B, L, d_model]
        return self.norm(self.value_embedding(x))


# ==============================================================================
# CGC (Standard) + Linear Weighting Baseline
# ==============================================================================
class PVTransformer_QuadTask_Advanced(pl.LightningModule):
    """
    Standard CGC baseline for structural comparison.

    This version corresponds to:
        CGC (Standard) + Linear Weighting

    Design:
    1. Keep the same dual-modal backbone as the compared models:
       TS/NWP token embedding -> cross-attention -> shared Transformer bottom.
    2. Use one shared expert and one task-specific expert for each task.
    3. All experts are homogeneous, i.e., they have the same hidden width and depth.
       This is the key distinction from CGC (Hetero).
    4. Use task-specific gates to fuse shared/private expert outputs.
    5. Do not use RouteReg, CA-FiLM, or Nash-MTL.
    6. Optimize the four task losses with ordinary linear/equal weighting.
    """

    def __init__(
        self,
        nwp_dim,
        num_classes,
        d_model=128,
        nhead=4,
        patch_len=8,
        stride=4,
        pred_len=24,
        lr=1e-4,
        weight_decay=1e-2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = True

        self.nwp_dim = nwp_dim
        self.num_classes = num_classes
        self.d_model = d_model
        self.pred_len = pred_len
        self.lr = lr
        self.weight_decay = weight_decay

        # ------------------------------------------------------------------
        # 1. Dual-modal backbone
        # ------------------------------------------------------------------
        self.ts_patcher = TokenEmbedding(input_dim=7, d_model=d_model)
        self.nwp_patcher = TokenEmbedding(input_dim=nwp_dim + 6, d_model=d_model)

        self.pos_enc_ts = nn.Parameter(torch.randn(1, 100, d_model) * 0.1)
        self.pos_enc_nwp = nn.Parameter(torch.randn(1, 100, d_model) * 0.1)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=0.1,
            batch_first=True,
        )
        self.fusion_norm = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.2,
            norm_first=True,
        )
        self.shared_bottom = nn.TransformerEncoder(layer, num_layers=2)

        # ------------------------------------------------------------------
        # 2. Standard homogeneous CGC expert pool
        # ------------------------------------------------------------------
        def make_standard_expert():
            return nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(d_model, d_model),
            )

        # All experts have exactly the same architecture.
        self.expert_shared = make_standard_expert()
        self.expert_main = make_standard_expert()
        self.expert_aux_dfi = make_standard_expert()
        self.expert_aux_w = make_standard_expert()
        self.expert_aux_ramp = make_standard_expert()

        # Task-specific CGC gates.
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
        # 3. Independent task heads
        #    Keep flattened heads to match the original CGC baseline setting.
        # ------------------------------------------------------------------
        flat_dim = d_model * pred_len

        self.head_power = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, pred_len),
        )
        self.head_dfi = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, pred_len),
        )
        self.head_weather = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
        self.head_ramp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, pred_len * 5),
        )

    @staticmethod
    def _gate_fuse(gate, gate_input, shared_feat, task_feat):
        weights = F.softmax(gate(gate_input), dim=-1)  # [B, 2]
        fused = weights[:, 0:1, None] * shared_feat + weights[:, 1:2, None] * task_feat
        return fused, weights

    def forward(self, ts, nwp):
        # ------------------------------------------------------------------
        # 1. Dual-modal feature extraction
        # ------------------------------------------------------------------
        p_ts = self.ts_patcher(ts)
        p_nwp = self.nwp_patcher(nwp)

        p_ts = p_ts + self.pos_enc_ts[:, :p_ts.size(1), :]
        p_nwp = p_nwp + self.pos_enc_nwp[:, :p_nwp.size(1), :]

        attn_out, _ = self.cross_attention(query=p_nwp, key=p_ts, value=p_ts)
        fused_feat = self.fusion_norm(p_nwp + attn_out)
        bottom_out = self.shared_bottom(fused_feat)

        # ------------------------------------------------------------------
        # 2. Standard CGC routing
        # ------------------------------------------------------------------
        out_shared = self.expert_shared(bottom_out)
        out_main = self.expert_main(bottom_out)
        out_dfi = self.expert_aux_dfi(bottom_out)
        out_w = self.expert_aux_w(bottom_out)
        out_ramp = self.expert_aux_ramp(bottom_out)

        gate_input = torch.mean(bottom_out, dim=1)

        fused_main, w_main = self._gate_fuse(
            self.gate_main, gate_input, out_shared, out_main
        )
        fused_dfi, w_dfi = self._gate_fuse(
            self.gate_aux_dfi, gate_input, out_shared, out_dfi
        )
        fused_w, w_w = self._gate_fuse(
            self.gate_aux_w, gate_input, out_shared, out_w
        )
        fused_ramp, w_ramp = self._gate_fuse(
            self.gate_aux_ramp, gate_input, out_shared, out_ramp
        )

        # ------------------------------------------------------------------
        # 3. Independent task predictions
        # ------------------------------------------------------------------
        p_pred = self.head_power(fused_main)                         # [B, P]
        dfi_pred = self.head_dfi(fused_dfi)                          # [B, P]
        w_logits = self.head_weather(fused_w)                        # [B, C]
        ramp_logits = self.head_ramp(fused_ramp).reshape(-1, self.pred_len, 5)

        p_sh_dict = {
            "m": w_main[:, 0].mean().detach().item(),
            "d": w_dfi[:, 0].mean().detach().item(),
            "w": w_w[:, 0].mean().detach().item(),
            "r": w_ramp[:, 0].mean().detach().item(),
        }

        return p_pred, dfi_pred, w_logits, ramp_logits, p_sh_dict

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=15,
            T_mult=2,
            eta_min=1e-6,
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        p_pred, dfi_pred, w_logits, ramp_logits, p_sh_dict = self(
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

        # Linear / equal weighting.
        total_loss = loss_p + loss_dfi + loss_w + loss_ramp

        self.log("train_loss_total", total_loss, prog_bar=True)
        self.log("1_Loss/Main_Power", loss_p, prog_bar=False)
        self.log("1_Loss/Aux_DFI", loss_dfi, prog_bar=False)
        self.log("1_Loss/Aux_Weather", loss_w, prog_bar=False)
        self.log("1_Loss/Aux_Ramp", loss_ramp, prog_bar=False)

        self.log("6_Gate/Main_use_Shared", p_sh_dict["m"], prog_bar=False)
        self.log("6_Gate/DFI_use_Shared", p_sh_dict["d"], prog_bar=False)
        self.log("6_Gate/Weather_use_Shared", p_sh_dict["w"], prog_bar=False)
        self.log("6_Gate/Ramp_use_Shared", p_sh_dict["r"], prog_bar=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        p_pred, _, _, _, _ = self(batch["ts_input"], batch["nwp_input"])
        mae = F.l1_loss(p_pred, batch["target_p"])
        self.log("val_mae", mae, on_epoch=True, prog_bar=True)
        return mae
