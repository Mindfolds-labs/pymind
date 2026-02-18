"""
Arquitetura Funil para MNIST (rápida)
784 → 49 → 10
"""

import torch
import torch.nn as nn
from ..core.camada import Camada
from ..core.conexoes import ConexaoRegional, ConexaoDensa

class FunilMNIST(nn.Module):
    """
    Arquitetura em funil: 784 pixels → 49 neurônios regionais → 10 classes
    """
    
    def __init__(self, n_dendritos=4, n_sinapses=4):
        super().__init__()
        
        self.n_dendritos = n_dendritos
        self.n_sinapses = n_sinapses
        
        # Conexão regional: cada neurônio vê uma região 4x4 da imagem
        self.regional = ConexaoRegional(
            in_height=28, in_width=28,
            out_neurons=49,  # 7x7 grid
            n_dendritos=n_dendritos,
            n_sinapses=n_sinapses,
            regiao_tamanho=4
        )
        
        # Camada de neurônios regionais
        self.camada_regional = Camada(49, n_dendritos, n_sinapses, nome="regional")
        
        # Conexão para classificação
        self.classif = ConexaoDensa(49, 10, n_dendritos, n_sinapses)
        
        # Camada de saída
        self.camada_saida = Camada(10, n_dendritos, n_sinapses, nome="saida")
    
    def forward(self, x):
        """
        x: [batch, 1, 28, 28]
        """
        # Conexão regional
        x = self.regional(x)  # [batch, 49, D, S]
        
        # Neurônios regionais
        spikes_reg = self.camada_regional(x)  # [batch, 49]
        
        # Conexão para classificação
        x = self.classif(spikes_reg)  # [batch, 10, D, S]
        
        # Camada de saída
        saida = self.camada_saida(x)  # [batch, 10]
        
        return saida
