import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.optimizer.NashOptimizer import NashMTLOptimizer


# ==============================================================================
# ==============================================================================
class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.value_embedding = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.value_embedding(x))


class PVTransformer_QuadTask_Advanced(pl.LightningModule):
    def __init__(self, nwp_dim, num_classes, d_model=128, nhead=4,
                 patch_len=8, stride=4, pred_len=24):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.nwp_dim = nwp_dim
        self.num_classes = num_classes
        self.d_model = d_model
        self.pred_len = pred_len

        self.ts_patcher = TokenEmbedding(input_dim=7, d_model=d_model)
        self.nwp_patcher = TokenEmbedding(input_dim=nwp_dim + 6, d_model=d_model)
        self.pos_enc_ts = nn.Parameter(torch.randn(1, 100, d_model) * 0.1)
        self.pos_enc_nwp = nn.Parameter(torch.randn(1, 100, d_model) * 0.1)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=0.1,
            batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.2,
            norm_first=True
        )
        self.shared_bottom = nn.TransformerEncoder(layer, num_layers=2)

        self.expert_shared = nn.Sequential(
            nn.Linear(d_model, d_model * 3),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model * 3, d_model)
        )
        self.expert_main = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model, d_model)
        )
        self.expert_aux_ramp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model * 2, d_model)
        )
        self.expert_aux_dfi = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model // 2, d_model)
        )
        self.expert_aux_w = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model // 2, d_model)
        )

        self.gate_main = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)
        )
        self.gate_aux_dfi = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)
        )
        self.gate_aux_w = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)
        )
        self.gate_aux_ramp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)
        )

        flat_dim = d_model * pred_len

        self.head_dfi = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1)
        )

        self.head_weather = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes)
        )

        self.head_ramp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 5)
        )

        input_phys_step_dim = 1 + num_classes + 5
        self.phys_projector = nn.Sequential(
            nn.Linear(input_phys_step_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        self.film_generator = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model * 2)
        )
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)

        self.head_power = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, pred_len)
        )

    def forward(self, ts, nwp):
        p_ts = self.ts_patcher(ts)
        p_nwp = self.nwp_patcher(nwp)
        p_ts = p_ts + self.pos_enc_ts[:, :p_ts.size(1), :]
        p_nwp = p_nwp + self.pos_enc_nwp[:, :p_nwp.size(1), :]

        attn_out, _ = self.cross_attention(query=p_nwp, key=p_ts, value=p_ts)
        fused_feat = self.fusion_norm(p_nwp + attn_out)
        bottom_out = self.shared_bottom(fused_feat)

        # CGC
        out_shared = self.expert_shared(bottom_out)
        out_main = self.expert_main(bottom_out)
        out_dfi = self.expert_aux_dfi(bottom_out)
        out_w = self.expert_aux_w(bottom_out)
        out_ramp = self.expert_aux_ramp(bottom_out)

        gate_input = torch.mean(bottom_out, dim=1)
        w_main = F.softmax(self.gate_main(gate_input), dim=-1)
        w_dfi = F.softmax(self.gate_aux_dfi(gate_input), dim=-1)
        w_w = F.softmax(self.gate_aux_w(gate_input), dim=-1)
        w_ramp = F.softmax(self.gate_aux_ramp(gate_input), dim=-1)

        fused_main = w_main[:, 0:1, None] * out_shared + w_main[:, 1:2, None] * out_main
        fused_dfi = w_dfi[:, 0:1, None] * out_shared + w_dfi[:, 1:2, None] * out_dfi
        fused_w = w_w[:, 0:1, None] * out_shared + w_w[:, 1:2, None] * out_w
        fused_ramp = w_ramp[:, 0:1, None] * out_shared + w_ramp[:, 1:2, None] * out_ramp

        dfi_pred = self.head_dfi(fused_dfi).squeeze(-1)              # [B, pred_len]
        weather_context = torch.mean(fused_w, dim=1)                # [B, d_model]
        w_logits = self.head_weather(weather_context)               # [B, num_classes]
        ramp_logits = self.head_ramp(fused_ramp)                    # [B, pred_len, 5]

        with torch.no_grad():
            # DFI: [B, P] -> [B, P, 1]
            dfi_context = dfi_pred.detach().unsqueeze(-1)

            # Weather: [B, C] -> [B, P, C]
            weather_prob = F.softmax(w_logits.detach(), dim=-1)
            weather_context = weather_prob.unsqueeze(1).expand(-1, self.pred_len, -1)

            # Ramp: [B, P, 5]
            ramp_context = F.softmax(ramp_logits.detach(), dim=-1)

            # [B, P, 1 + C + 5]
            phys_context_step = torch.cat([
                dfi_context,
                weather_context,
                ramp_context,
            ], dim=-1)

        # [B, P, d_model]
        phys_embedding_step = self.phys_projector(phys_context_step)

        combined_info = torch.cat([fused_main, phys_embedding_step], dim=-1)
        film_params = self.film_generator(combined_info)            # [B, P, 2*d_model]
        gamma, beta = torch.chunk(film_params, 2, dim=-1)           #  [B, P, d_model]

        fused_main_refined = fused_main * (1.0 + gamma) + beta      # [B, P, d_model]
        p_pred = self.head_power(fused_main_refined)

        p_sh_dict = {
            'm': w_main.mean(dim=0)[0].item(),
            'd': w_dfi.mean(dim=0)[0].item(),
            'w': w_w.mean(dim=0)[0].item(),
            'r': w_ramp.mean(dim=0)[0].item()
        }

        return p_pred, dfi_pred, w_logits, ramp_logits, p_sh_dict, w_w, w_ramp

    def configure_optimizers(self):
        base_opt = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)

        shared_params = (
            list(self.ts_patcher.parameters()) +
            list(self.nwp_patcher.parameters()) +
            [self.pos_enc_ts, self.pos_enc_nwp] +
            list(self.cross_attention.parameters()) +
            list(self.fusion_norm.parameters()) +
            list(self.expert_shared.parameters()) +
            list(self.shared_bottom.parameters())
        )

        main_params = (
            list(self.expert_main.parameters()) +
            list(self.gate_main.parameters()) +
            list(self.phys_projector.parameters()) +
            list(self.film_generator.parameters()) +
            list(self.head_power.parameters())
        )

        dfi_params = (
            list(self.expert_aux_dfi.parameters()) +
            list(self.gate_aux_dfi.parameters()) +
            list(self.head_dfi.parameters())
        )
        w_params = (
            list(self.expert_aux_w.parameters()) +
            list(self.gate_aux_w.parameters()) +
            list(self.head_weather.parameters())
        )
        ramp_params = (
            list(self.expert_aux_ramp.parameters()) +
            list(self.gate_aux_ramp.parameters()) +
            list(self.head_ramp.parameters())
        )

        self.nash_optimizer = NashMTLOptimizer(
            base_opt,
            shared_params,
            main_params,
            dfi_params,
            w_params,
            ramp_params
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            base_opt,
            T_0=15,
            T_mult=2,
            eta_min=1e-6
        )
        return [base_opt], [scheduler]

    def training_step(self, batch, batch_idx):
        opt = self.nash_optimizer
        p_pred, dfi_pred, w_logits, ramp_logits, p_sh_dict, w_w, w_r = self(
            batch['ts_input'],
            batch['nwp_input']
        )

        loss_p = F.huber_loss(p_pred, batch['target_p'], delta=0.5)
        loss_dfi = F.mse_loss(dfi_pred, batch['target_dfi'])
        loss_w = F.cross_entropy(w_logits, batch['daily_weather'])
        loss_ramp = F.cross_entropy(ramp_logits.view(-1, 5), batch['target_ramp'].view(-1))

        penalty_w = F.relu(w_w[:, 0] - 0.40).mean() * 3.0
        penalty_r = F.relu(w_r[:, 0] - 0.40).mean() * 3.0
        loss_w = loss_w + penalty_w
        loss_ramp = loss_ramp + penalty_r

        metrics = opt.backward(loss_p, loss_dfi, loss_w, loss_ramp)

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        opt.step()
        if self.trainer.is_last_batch:
            self.lr_schedulers().step()

        self.log('train_loss_p', loss_p, prog_bar=True)
        self.log('1_Loss/Aux_DFI_Raw', loss_dfi, prog_bar=True)
        self.log('1_Loss/Aux_Weather_Raw', loss_w, prog_bar=True)
        self.log('1_Loss/Aux_Ramp_Raw', loss_ramp, prog_bar=True)

        self.log('6_Gate/Main_use_Shared', p_sh_dict['m'], prog_bar=False)
        self.log('6_Gate/Weather_use_Shared', p_sh_dict['w'], prog_bar=False)
        self.log('6_Gate/Ramp_use_Shared', p_sh_dict['r'], prog_bar=False)
        self.log('6_Gate/DFI_use_Shared', p_sh_dict['d'], prog_bar=False)

        self.log('3_Nash/CosTheta_Overall_Combined', metrics['cos_combined'], prog_bar=True)
        self.log('7_Gradient_Angle/Cos_Main_vs_DFI', metrics['cos_md'], prog_bar=False)
        self.log('7_Gradient_Angle/Cos_Main_vs_Weather', metrics['cos_mw'], prog_bar=False)
        self.log('7_Gradient_Angle/Cos_Main_vs_Ramp', metrics['cos_mr'], prog_bar=False)

        return loss_p

    def validation_step(self, batch, batch_idx):
        p_pred, _, _, _, _, _, _ = self(batch['ts_input'], batch['nwp_input'])
        mae = F.l1_loss(p_pred, batch['target_p'])
        self.log('val_mae', mae, on_epoch=True, prog_bar=True)
        return mae