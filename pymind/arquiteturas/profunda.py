"""
Arquitetura Profunda para tarefas complexas
"""

import torch
import torch.nn as nn
from ..core.camada import Camada
from ..core.conexoes import ConexaoDensa

class Profunda(nn.Module):
    """
    Arquitetura profunda genérica
    """
    
    def __init__(self, dims, n_dendritos=4, n_sinapses=4):
        super().__init__()
        
        self.camadas = nn.ModuleList()
        self.projecoes = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            self.projecoes.append(
                ConexaoDensa(dims[i], dims[i+1], n_dendritos, n_sinapses)
            )
            self.camadas.append(
                Camada(dims[i+1], n_dendritos, n_sinapses, nome=f"profunda_{i}")
            )
    
    def forward(self, x):
        for proj, camada in zip(self.projecoes, self.camadas):
            x = proj(x)
            x = camada(x)
        return x
