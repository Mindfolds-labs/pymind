"""
Autoencoder para MNIST
"""

import torch
import torch.nn as nn
from ..core.camada import Camada
from ..core.conexoes import ConexaoDensa


class AutoencoderMNIST(nn.Module):
    """
    Autoencoder com neurônios dendríticos.

    Encoder : 784 → 128 → encoding_dim  (representação latente binária)
    Decoder : encoding_dim → 128 → 784  (reconstrução contínua em [0, 1])

    Problema original (Hinton & Salakhutdinov, 2006):
        A última Camada dendrítica produz spikes binários {0, 1} por
        construção (threshold no neurônio). Usar essa saída diretamente
        como reconstrução é incompatível com imagens em [0, 1]: a perda
        MSE ou BCE não converge para representações contínuas.

    Correção:
        Uma projeção linear final (decoder_logits) mapeia os spikes
        binários de 784 neurônios para logits reais em ℝ^784.
        A sigmoid transforma esses logits em probabilidades em (0, 1),
        compatíveis com BCE como função de perda de reconstrução.
        Isso desacopla o mecanismo de spike dendrítico da necessidade
        de produzir valores contínuos.
    """

    def __init__(self, encoding_dim=32, n_dendritos=4, n_sinapses=4):
        super().__init__()

        # ── Encoder ───────────────────────────────────────────────────
        self.encoder_proj1   = ConexaoDensa(784, 128, n_dendritos, n_sinapses)
        self.encoder_camada1 = Camada(128, n_dendritos, n_sinapses)
        self.encoder_proj2   = ConexaoDensa(128, encoding_dim, n_dendritos, n_sinapses)
        self.encoder_camada2 = Camada(encoding_dim, n_dendritos, n_sinapses)

        # ── Decoder ───────────────────────────────────────────────────
        self.decoder_proj1   = ConexaoDensa(encoding_dim, 128, n_dendritos, n_sinapses)
        self.decoder_camada1 = Camada(128, n_dendritos, n_sinapses)
        self.decoder_proj2   = ConexaoDensa(128, 784, n_dendritos, n_sinapses)
        self.decoder_camada2 = Camada(784, n_dendritos, n_sinapses)

        # CORRIGIDO: projeção linear + sigmoid para reconstrução contínua.
        # Mapeia os 784 spikes binários → logits reais → probabilidades (0,1).
        self.decoder_logits = nn.Linear(784, 784)

    def forward(self, x):
        """
        x    : [batch, 1, 28, 28] ou [batch, 784]
        saída: (decoded [batch, 784], encoded [batch, encoding_dim])
        """
        if x.dim() == 4:
            x = x.view(x.shape[0], -1)

        # ── Encode ────────────────────────────────────────────────────
        x       = self.encoder_proj1(x)
        x       = self.encoder_camada1(x)
        x       = self.encoder_proj2(x)
        encoded = self.encoder_camada2(x)         # [batch, encoding_dim]  binário

        # ── Decode ────────────────────────────────────────────────────
        x       = self.decoder_proj1(encoded)
        x       = self.decoder_camada1(x)
        x       = self.decoder_proj2(x)
        spikes  = self.decoder_camada2(x)          # [batch, 784]  binário

        # Logits contínuos → sigmoid → reconstrução em (0, 1)
        decoded = torch.sigmoid(self.decoder_logits(spikes))

        return decoded, encoded
