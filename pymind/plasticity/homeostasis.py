"""
Homeostase - manutenção da atividade neuronal
"""

import torch
import torch.nn as nn

class Homeostase(nn.Module):
    """
    Ajusta thresholds para manter taxa de disparo alvo
    """
    
    def __init__(self, target_rate=0.1, adapt_rate=0.01):
        super().__init__()
        self.target_rate = target_rate
        self.adapt_rate = adapt_rate
    
    def forward(self, spike_rate, theta):
        """
        Ajusta theta baseado na taxa de disparo
        """
        error = spike_rate - self.target_rate
        return theta + self.adapt_rate * error
