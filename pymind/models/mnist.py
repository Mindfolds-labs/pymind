"""
Modelo classificador para MNIST
"""

import torch
import torch.nn as nn
from ..arquiteturas.piramidal import PiramidalMNIST

class MNISTClassifier(nn.Module):
    """
    Wrapper para classificação MNIST
    """
    
    def __init__(self, arquitetura='piramidal', config=None):
        super().__init__()
        
        if arquitetura == 'piramidal':
            self.modelo = PiramidalMNIST(config)
        else:
            raise ValueError(f"Arquitetura {arquitetura} não suportada")
    
    def forward(self, x):
        return self.modelo(x)
    
    def predict(self, x):
        with torch.no_grad():
            outputs = self.forward(x)
            return outputs.argmax(dim=1)
