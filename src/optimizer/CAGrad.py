import torch
import numpy as np
import scipy.optimize as opt

class CAGradOptimizer:
    def __init__(self, optimizer, shared_params, main_params, dfi_params, w_params, ramp_params, c=0.4):
        self._optim = optimizer
        self.shared_params = list(shared_params)
        self.main_params = list(main_params)
        self.dfi_params = list(dfi_params)
        self.w_params = list(w_params)
        self.ramp_params = list(ramp_params)
        self.c = c 

    @property
    def optimizer(self): return self._optim
    def zero_grad(self): return self._optim.zero_grad()
    def step(self): return self._optim.step()

    def backward(self, loss_main, loss_dfi, loss_w, loss_ramp):
        self.zero_grad()
        loss_main.backward(retain_graph=True)
        g_m_shared = self._get_grads(self.shared_params)
        g_m_main = self._get_grads(self.main_params)
        
        self.zero_grad()
        loss_dfi.backward(retain_graph=True)
        g_d_shared = self._get_grads(self.shared_params)
        g_d_dfi = self._get_grads(self.dfi_params)
        
        self.zero_grad()
        loss_w.backward(retain_graph=True) 
        g_w_shared = self._get_grads(self.shared_params)
        g_w_w = self._get_grads(self.w_params)
        
        self.zero_grad()
        loss_ramp.backward(retain_graph=False) 
        g_r_shared = self._get_grads(self.shared_params)
        g_r_ramp = self._get_grads(self.ramp_params)
        
        g_m_flat = torch.cat([g.flatten() for g in g_m_shared])
        g_d_flat = torch.cat([g.flatten() for g in g_d_shared])
        g_w_flat = torch.cat([g.flatten() for g in g_w_shared])
        g_r_flat = torch.cat([g.flatten() for g in g_r_shared])
        
        norm_m = torch.norm(g_m_flat) + 1e-8
        # 日志指标计算...
        u_m = g_m_flat / norm_m
        cos_md = torch.dot(u_m, g_d_flat / (torch.norm(g_d_flat)+1e-8)).item()
        cos_mw = torch.dot(u_m, g_w_flat / (torch.norm(g_w_flat)+1e-8)).item()
        cos_mr = torch.dot(u_m, g_r_flat / (torch.norm(g_r_flat)+1e-8)).item()


        G = torch.stack([g_m_flat, g_d_flat, g_w_flat, g_r_flat], dim=1) # [dim, 4]
        g0 = torch.mean(G, dim=1) 
        
        GG_np = torch.matmul(G.T, G).cpu().numpy() # 4x4
        g0G_np = torch.matmul(g0.unsqueeze(0), G).squeeze().cpu().numpy() # 4
        
        # 目标函数: 最小化 f(w) = c * sqrt(w^T * G^T * G * w) + g0^T * G * w
        def obj_fn(w):
            return self.c * np.sqrt(np.dot(w, np.dot(GG_np, w)) + 1e-8) + np.dot(g0G_np, w)
            
        # 约束条件: w 之和为 1，且 w >= 0
        bounds = [(0.0, 1.0) for _ in range(4)]
        cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        
        # SLSQP 优化
        res = opt.minimize(obj_fn, np.ones(4)/4.0, method='SLSQP', bounds=bounds, constraints=cons)
        w_star = torch.tensor(res.x, dtype=torch.float32, device=G.device)
        
        gw = torch.matmul(G, w_star)
        gw_norm = torch.norm(gw) + 1e-8
        g0_norm = torch.norm(g0) + 1e-8
        
        # d = g0 + c * ||g0|| * (Gw / ||Gw||)
        g_shared_final_flat = g0 + self.c * g0_norm * gw / gw_norm
        # ======================================================================
        
        norm_shared_final = torch.norm(g_shared_final_flat)
        cos_theta_combined = torch.dot(g_m_flat, g_shared_final_flat) / (norm_m * norm_shared_final + 1e-8)
            
        offset = 0
        g_shared_final = []
        for p in self.shared_params:
            numel = p.numel()
            g_shared_final.append(g_shared_final_flat[offset:offset+numel].view(p.shape))
            offset += numel
            
        self.zero_grad()
        for p, g in zip(self.shared_params, g_shared_final): p.grad = g
        for p, g in zip(self.main_params, g_m_main): p.grad = g
        for p, g in zip(self.dfi_params, g_d_dfi): p.grad = g 
        for p, g in zip(self.w_params, g_w_w): p.grad = g 
        for p, g in zip(self.ramp_params, g_r_ramp): p.grad = g 
            
        return {
            "norm_main": norm_m.item(), "norm_aux_aligned": norm_shared_final.item(),
            "cos_combined": cos_theta_combined.item(), 
            "cos_md": cos_md, "cos_mw": cos_mw, "cos_mr": cos_mr, 
            "cos_dw": 0, "cos_dr": 0, "cos_wr": 0 
        }

    def _get_grads(self, params):
        return [torch.zeros_like(p).to(p.device) if p.grad is None else p.grad.clone() for p in params]