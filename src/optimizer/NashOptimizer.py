import torch

class NashMTLOptimizer:
    def __init__(self, optimizer, shared_params, main_params, dfi_params, w_params, ramp_params):
        self._optim = optimizer
        self.shared_params = list(shared_params)
        self.main_params = list(main_params)
        self.dfi_params = list(dfi_params)
        self.w_params = list(w_params)
        self.ramp_params = list(ramp_params)

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
        norm_d = torch.norm(g_d_flat) + 1e-8
        norm_w = torch.norm(g_w_flat) + 1e-8
        norm_r = torch.norm(g_r_flat) + 1e-8
        
        u_m = g_m_flat / norm_m
        u_d = g_d_flat / norm_d
        u_w = g_w_flat / norm_w
        u_r = g_r_flat / norm_r
        
        cos_md = torch.dot(u_m, u_d).item()
        cos_mw = torch.dot(u_m, u_w).item()
        cos_mr = torch.dot(u_m, u_r).item()
        cos_dw = torch.dot(u_d, u_w).item()
        cos_dr = torch.dot(u_d, u_r).item()
        cos_wr = torch.dot(u_w, u_r).item()
        
        C = torch.tensor([
            [1.0, cos_md, cos_mw, cos_mr],
            [cos_md, 1.0, cos_dw, cos_dr],
            [cos_mw, cos_dw, 1.0, cos_wr],
            [cos_mr, cos_dr, cos_wr, 1.0]
        ], device=g_m_flat.device)
        C = C + torch.eye(4, device=g_m_flat.device) * 1e-4
        
        alpha = torch.ones(4, device=g_m_flat.device)
        for _ in range(10):
            grad_alpha = torch.matmul(C, alpha) - 1.0 / (alpha + 1e-8)
            hessian = C + torch.diag(1.0 / (alpha**2 + 1e-8))
            try: delta = torch.linalg.solve(hessian, grad_alpha)
            except RuntimeError: delta = grad_alpha * 0.1
            alpha = torch.clamp(alpha - delta, min=1e-4)
            
        alpha_norm = alpha / torch.sum(alpha)
        g_shared_direction = alpha_norm[0] * u_m + alpha_norm[1] * u_d + alpha_norm[2] * u_w + alpha_norm[3] * u_r
        
        avg_norm = (norm_m + norm_d + norm_w + norm_r) / 4.0
        g_shared_final_flat = g_shared_direction * avg_norm
        
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
            "cos_dw": cos_dw, "cos_dr": cos_dr, "cos_wr": cos_wr
        }

    def _get_grads(self, params):
        return [torch.zeros_like(p).to(p.device) if p.grad is None else p.grad.clone() for p in params]