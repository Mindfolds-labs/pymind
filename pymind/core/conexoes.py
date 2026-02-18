"""
Módulo de conexões entre camadas
"""

import torch
import torch.nn as nn
import math

class ConexaoDensa(nn.Module):
    """Conexão densa (cada neurônio da camada anterior conecta a todos da próxima)"""
    
    def __init__(self, in_features, out_features, n_dendritos, n_sinapses):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_dendritos = n_dendritos
        self.n_sinapses = n_sinapses
        
        # Matriz de pesos
        self.peso = nn.Parameter(
            torch.randn(out_features, in_features, n_dendritos, n_sinapses)
        )
    
    def forward(self, x):
        """
        x: [batch, in_features]
        Returns: [batch, out_features, n_dendritos, n_sinapses]
        """
        batch = x.shape[0]
        return torch.einsum('bi,oinds->bonds', x, self.peso)


class ConexaoRegional(nn.Module):
    """Conexão regional (cada neurônio vê uma região específica da entrada)"""
    
    def __init__(self, in_height, in_width, out_neurons, 
                 n_dendritos, n_sinapses, regiao_tamanho=4):
        super().__init__()
        self.in_h = in_height
        self.in_w = in_width
        self.out_n = out_neurons
        self.n_dendritos = n_dendritos
        self.n_sinapses = n_sinapses
        self.regiao_tam = regiao_tamanho
        
        # Calcular grid de regiões
        self.grid_h = in_height // regiao_tamanho
        self.grid_w = in_width // regiao_tamanho
        
        assert out_neurons == self.grid_h * self.grid_w, \
            f"Número de neurônios deve ser {self.grid_h * self.grid_w}"
        
        # Pesos por região
        self.pesos = nn.Parameter(
            torch.randn(out_neurons, n_dendritos, n_sinapses, 
                       regiao_tamanho, regiao_tamanho)
        )
    
    def forward(self, x):
        """
        x: [batch, 1, height, width]
        Returns: [batch, out_neurons, n_dendritos, n_sinapses]
        """
        batch = x.shape[0]
        resultado = []
        
        idx = 0
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                y1, y2 = i * self.regiao_tam, (i + 1) * self.regiao_tam
                x1, x2 = j * self.regiao_tam, (j + 1) * self.regiao_tam
                
                regiao = x[:, :, y1:y2, x1:x2]  # [batch, 1, R, R]
                
                # Aplicar pesos deste neurônio
                peso = self.pesos[idx]  # [D, S, R, R]
                ativ = torch.einsum('bcrr,dsrr->bds', regiao, peso)
                resultado.append(ativ)
                
                idx += 1
        
        return torch.stack(resultado, dim=1)  # [batch, out_n, D, S]


class ConexaoEsparsa(nn.Module):
    """Conexão esparsa aleatória"""
    
    def __init__(self, in_features, out_features, n_dendritos, n_sinapses, 
                 esparsidade=0.1):
        super().__init__()
        
        # Criar máscara esparsa
        mascara = torch.rand(out_features, in_features) < esparsidade
        
        self.register_buffer('mascara', mascara.float())
        self.peso = nn.Parameter(
            torch.randn(out_features, in_features, n_dendritos, n_sinapses)
        )
    
    def forward(self, x):
        """
        x: [batch, in_features]
        Returns: [batch, out_features, n_dendritos, n_sinapses]
        """
        # Aplicar máscara para conexões esparsas
        peso_esparso = self.peso * self.mascara.view(
            self.peso.shape[0], self.peso.shape[1], 1, 1
        )
        
        return torch.einsum('bi,oinds->bonds', x, peso_esparso)
