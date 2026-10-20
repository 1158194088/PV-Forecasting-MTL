import torch

class Standard_MTLOptimizer:
    def __init__(self, optimizer):
        self._optim = optimizer

    @property
    def optimizer(self): return self._optim
    def zero_grad(self): return self._optim.zero_grad()
    def step(self): return self._optim.step()

    def backward(self, loss_main, loss_dfi, loss_w, loss_ramp, p_sh_dict=None, decay_factor=None):
        self.zero_grad()
        
        loss_main.backward()
        
        return {
            "norm_m": 0.0, "norm_d": 0.0, "norm_w": 0.0, "norm_r": 0.0,
            "alpha_m": 1.0, "alpha_d": 0.0, "alpha_w": 0.0, "alpha_r": 0.0,
            "target_norm": 0.0, "cos_combined": 1.0, 
            "cos_md": 0.0, "cos_mw": 0.0, "cos_mr": 0.0
        }