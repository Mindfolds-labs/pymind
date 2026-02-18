"""
STDP (Spike-Timing Dependent Plasticity)
"""

import torch
import torch.nn as nn

class STDP(nn.Module):
    """
    Plasticidade dependente do tempo dos spikes
    """
    
    def __init__(self, learning_rate=0.01, tau=20.0):
        super().__init__()
        self.lr = learning_rate
        self.tau = tau
    
    def forward(self, t_pre, t_post, pesos):
        """
        t_pre: tempos dos spikes pré-sinápticos
        t_post: tempos dos spikes pós-sinápticos
        """
        delta_t = t_post - t_pre
        if delta_t > 0:
            # LTP - pré antes do pós
            delta = self.lr * torch.exp(-delta_t / self.tau)
        else:
            # LTD - pós antes do pré
            delta = -self.lr * torch.exp(delta_t / self.tau)
        
        return pesos + delta
