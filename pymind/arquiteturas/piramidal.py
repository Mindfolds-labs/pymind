"""
Arquitetura Piramidal para MNIST
784 → 128 → 64 → 32 → 10
"""

import torch
import torch.nn as nn
from ..core.camada import Camada
from ..core.conexoes import ConexaoDensa
from ..memory.engram import Engram


class PiramidalMNIST(nn.Module):
    """
    Rede neural em formato piramidal com memória Engram.

    Args:
        config: dicionário com configurações
            - n_dendritos    : dendritos por neurônio
            - n_sinapses     : sinapses por dendrito
            - usar_engram    : ativar memória engram
            - max_prototipos : máximo de protótipos por engram
    """

    def __init__(self, config=None):
        super().__init__()

        if config is None:
            config = {}

        self.config = config
        D = config.get('n_dendritos', 4)
        S = config.get('n_sinapses', 4)

        # Projeções entre camadas
        self.proj1 = ConexaoDensa(784, 128, D, S)
        self.proj2 = ConexaoDensa(128,  64, D, S)
        self.proj3 = ConexaoDensa( 64,  32, D, S)
        self.proj4 = ConexaoDensa( 32,  10, D, S)

        # Camadas da pirâmide
        self.camada1 = Camada(128, D, S, nome="piramide_1")
        self.camada2 = Camada( 64, D, S, nome="piramide_2")
        self.camada3 = Camada( 32, D, S, nome="piramide_3")
        self.camada4 = Camada( 10, D, S, nome="piramide_4")

        # Engrams opcionais
        self.engrams = nn.ModuleDict()
        if config.get('usar_engram', False):
            max_protos = config.get('max_prototipos', 100)
            self.engrams['camada1'] = Engram(128, max_protos)
            self.engrams['camada2'] = Engram( 64, max_protos)
            self.engrams['camada3'] = Engram( 32, max_protos)
            self.engrams['saida']   = Engram( 10, max_protos)

        # Cache para feedback top-down e ativações
        self.feedback_cache = {}
        self.ativacoes = {}

    # ------------------------------------------------------------------
    # Utilitário interno
    # ------------------------------------------------------------------
    @staticmethod
    def _aplicar_feedback(x, feedback, ganho=0.1):
        """
        Injeta o sinal de feedback top-down no tensor dendrítico.

        x        : [batch, N, D, S]
        feedback : [batch, N]         (retorno do Engram.observar)

        O feedback representa modulação somática — mesmo valor para todos
        os dendritos e sinapses de cada neurônio (Friston, 2005).

        CORRIGIDO: a soma direta causava RuntimeError por broadcasting
        inválido entre (batch, N, D, S) e (batch, N).
        A solução é expandir feedback para (batch, N, 1, 1) antes da soma;
        o broadcasting do PyTorch propaga o valor por D e S automaticamente.
        """
        return x + ganho * feedback.unsqueeze(-1).unsqueeze(-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x, erro=None):
        """
        x    : [batch, 1, 28, 28] ou [batch, 784]
        erro : erro atual para o engram (escalar ou tensor [batch])
        """
        if x.dim() == 4:
            x = x.view(x.shape[0], -1)   # [batch, 784]

        # ── Camada 1 : 128 neurônios ──────────────────────────────────
        x = self.proj1(x)                 # [batch, 128, D, S]

        if 'camada1' in self.feedback_cache:
            x = self._aplicar_feedback(x, self.feedback_cache['camada1'])

        spikes1 = self.camada1(x)
        self.ativacoes['camada1'] = spikes1.detach()

        if erro is not None and 'camada1' in self.engrams:
            self.feedback_cache['camada1'] = self.engrams['camada1'].observar(spikes1, erro)

        # ── Camada 2 : 64 neurônios ───────────────────────────────────
        x = self.proj2(spikes1)

        if 'camada2' in self.feedback_cache:
            x = self._aplicar_feedback(x, self.feedback_cache['camada2'])

        spikes2 = self.camada2(x)
        self.ativacoes['camada2'] = spikes2.detach()

        if erro is not None and 'camada2' in self.engrams:
            self.feedback_cache['camada2'] = self.engrams['camada2'].observar(spikes2, erro)

        # ── Camada 3 : 32 neurônios ───────────────────────────────────
        x = self.proj3(spikes2)

        if 'camada3' in self.feedback_cache:
            x = self._aplicar_feedback(x, self.feedback_cache['camada3'])

        spikes3 = self.camada3(x)
        self.ativacoes['camada3'] = spikes3.detach()

        if erro is not None and 'camada3' in self.engrams:
            self.feedback_cache['camada3'] = self.engrams['camada3'].observar(spikes3, erro)

        # ── Camada 4 : 10 neurônios (saída) ───────────────────────────
        x = self.proj4(spikes3)
        spikes4 = self.camada4(x)
        self.ativacoes['saida'] = spikes4.detach()

        if erro is not None and 'saida' in self.engrams:
            self.engrams['saida'].observar(spikes4, erro)

        return spikes4

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_ativacoes(self):
        """Retorna ativações para visualização."""
        return self.ativacoes

    def get_engram_stats(self):
        """Retorna estatísticas dos engrams."""
        return {nome: eng.get_estatisticas() for nome, eng in self.engrams.items()}
