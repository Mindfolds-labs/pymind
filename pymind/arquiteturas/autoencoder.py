"""
Autoencoder para MNIST
"""

import torch
import torch.nn as nn
from ..core.camada import Camada
from ..core.conexoes import ConexaoDensa

class AutoencoderMNIST(nn.Module):
    """
    Autoencoder com neurônios dendríticos
    """
    
    def __init__(self, encoding_dim=32, n_dendritos=4, n_sinapses=4):
        super().__init__()
        
        # Encoder
        self.encoder_proj1 = ConexaoDensa(784, 128, n_dendritos, n_sinapses)
        self.encoder_camada1 = Camada(128, n_dendritos, n_sinapses)
        self.encoder_proj2 = ConexaoDensa(128, encoding_dim, n_dendritos, n_sinapses)
        self.encoder_camada2 = Camada(encoding_dim, n_dendritos, n_sinapses)
        
        # Decoder
        self.decoder_proj1 = ConexaoDensa(encoding_dim, 128, n_dendritos, n_sinapses)
        self.decoder_camada1 = Camada(128, n_dendritos, n_sinapses)
        self.decoder_proj2 = ConexaoDensa(128, 784, n_dendritos, n_sinapses)
        self.decoder_camada2 = Camada(784, n_dendritos, n_sinapses)
    
    def forward(self, x):
        # Flatten
        if x.dim() == 4:
            x = x.view(x.shape[0], -1)
        
        # Encode
        x = self.encoder_proj1(x)
        x = self.encoder_camada1(x)
        x = self.encoder_proj2(x)
        encoded = self.encoder_camada2(x)
        
        # Decode
        x = self.decoder_proj1(encoded)
        x = self.decoder_camada1(x)
        x = self.decoder_proj2(x)
        decoded = self.decoder_camada2(x)
        
        return decoded, encoded
