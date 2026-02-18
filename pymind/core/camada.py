"""
Camada com múltiplos neurônios
"""

import torch
import torch.nn as nn
import math
from .neuronio import NeuronioDendritico

class Camada(nn.Module):
    """
    Camada de neurônios organizados
    
    Args:
        n_neurons: número de neurônios na camada
        n_dendritos: dendritos por neurônio
        n_sinapses_por_dendrito: sinapses por dendrito
        conexao: tipo de conexão ('densa', 'esparsa', 'grid')
        nome: identificador da camada
    """
    
    def __init__(self, n_neurons, n_dendritos=4, n_sinapses_por_dendrito=4,
                 conexao='densa', nome=None):
        super().__init__()
        
        self.n_neurons = n_neurons
        self.nome = nome or f"camada_{id(self)}"
        self.conexao = conexao
        
        # Criar neurônios
        self.neuronios = nn.ModuleList([
            NeuronioDendritico(
                n_dendritos=n_dendritos,
                n_sinapses_por_dendrito=n_sinapses_por_dendrito,
                nome=f"{self.nome}_{i}"
            ) for i in range(n_neurons)
        ])
        
        # Matriz de conexão para diferentes topologias
        if conexao != 'densa':
            self.register_buffer(
                'matriz_conexao',
                self._criar_matriz_conexao(conexao)
            )
    
    def _criar_matriz_conexao(self, tipo):
        """Cria matriz de conexão para topologias especiais"""
        if tipo == 'grid_2d':
            # Conexão em grid 2D (cada neurônio conectado aos vizinhos)
            lado = int(math.sqrt(self.n_neurons))
            matriz = torch.zeros(self.n_neurons, self.n_neurons)
            for i in range(self.n_neurons):
                for j in range(self.n_neurons):
                    # Conectar se são vizinhos (distância <= 1 no grid)
                    i1, j1 = i // lado, i % lado
                    i2, j2 = j // lado, j % lado
                    if abs(i1 - i2) <= 1 and abs(j1 - j2) <= 1:
                        matriz[i, j] = 1.0
            return matriz
        
        elif tipo == 'esparsa':
            # Conexão aleatória esparsa (10% de conectividade)
            prob = 0.1
            return (torch.rand(self.n_neurons, self.n_neurons) < prob).float()
        
        else:  # densa
            return torch.ones(self.n_neurons, self.n_neurons)
    
    def forward(self, x):
        """
        x: [batch, n_neurons, n_dendritos, n_sinapses]
        
        Returns:
            spikes: [batch, n_neurons]
        """
        # Para topologias não-densas, a matriz de conexão define
        # como as entradas de todos os neurônios contribuem para cada
        # neurônio-alvo da camada.
        #
        # x_mapeado[b, o, d, s] = Σ_i x[b, i, d, s] * M[o, i]
        if self.conexao != 'densa':
            x = torch.einsum('bids,oi->bods', x, self.matriz_conexao)

        spikes = []
        
        for i, neuronio in enumerate(self.neuronios):
            spike = neuronio(x[:, i])  # [batch]
            spikes.append(spike)
        
        return torch.stack(spikes, dim=1)  # [batch, n_neurons]
    
    def get_estados(self):
        """Retorna estados de todos os neurônios"""
        return [n.get_estado() for n in self.neuronios]
    
    def aplicar_plasticidade(self, **kwargs):
        """Aplica plasticidade em todos os neurônios"""
        for neuronio in self.neuronios:
            if hasattr(neuronio, 'atualizar_plasticidade'):
                neuronio.atualizar_plasticidade(**kwargs)
