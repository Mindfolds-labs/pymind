"""
Memória de trabalho (curto prazo)
"""

import torch
import torch.nn as nn
from collections import deque

class MemoriaTrabalho(nn.Module):
    """
    Memória de curto prazo com buffer circular
    
    Args:
        capacidade: número de itens a armazenar
        dim: dimensão de cada item
    """
    
    def __init__(self, capacidade=10, dim=784):
        super().__init__()
        self.capacidade = capacidade
        self.dim = dim
        
        self.register_buffer('buffer', torch.zeros(capacidade, dim))
        self.register_buffer('idades', torch.zeros(capacidade, dtype=torch.long))
        self.register_buffer('pos', torch.tensor(0, dtype=torch.long))
    
    def adicionar(self, item):
        """Adiciona item à memória"""
        self.buffer[self.pos] = item.detach()
        self.idades += 1
        self.idades[self.pos] = 0
        self.pos = (self.pos + 1) % self.capacidade
    
    def recuperar(self, k=3):
        """Recupera os k itens mais recentes"""
        indices = torch.argsort(self.idades)[:k]
        return self.buffer[indices]
    
    def limpar(self):
        """Limpa a memória"""
        self.buffer.zero_()
        self.idades.zero_()
        self.pos.zero_()
