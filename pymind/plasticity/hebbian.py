"""
Regra de plasticidade Hebbiana
"""

import torch
import torch.nn as nn

class Hebbian(nn.Module):
    """
    Plasticidade Hebbiana: "cells that fire together, wire together"
    """
    
    def __init__(self, learning_rate=0.01):
        super().__init__()
        self.lr = learning_rate
    
    def forward(self, pre, post, pesos):
        """
        pre: atividade pré-sináptica
        post: atividade pós-sináptica
        pesos: pesos atuais
        """
        delta = self.lr * torch.einsum('bi,bj->bij', pre, post)
        return pesos + delta.mean(dim=0)
