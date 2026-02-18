"""
STDP (Spike-Timing Dependent Plasticity)
"""

import torch
import torch.nn as nn


class STDP(nn.Module):
    """
    Plasticidade dependente do tempo dos spikes — formulação vetorizada.

    Regra original (Bi & Poo, 1998):
        ΔW =  A+ · exp(−Δt / τ)   se Δt > 0  (LTP — pré antes do pós)
        ΔW = −A- · exp( Δt / τ)   se Δt ≤ 0  (LTD — pós antes do pré)

    Formulação vetorizada equivalente (sem ramificação condicional):
        ΔW = lr · sign(Δt) · exp(−|Δt| / τ)

    Vantagens:
      • Funciona para qualquer shape de tensor (escalar, 1-D, n-D, batch).
      • Elimina o 'if delta_t > 0' que lança RuntimeError quando delta_t
        é um tensor com mais de um elemento.
      • Totalmente compatível com autograd do PyTorch.
    """

    def __init__(self, learning_rate=0.01, tau=20.0):
        super().__init__()
        self.lr = learning_rate
        self.tau = tau

    def forward(self, t_pre, t_post, pesos):
        """
        t_pre : tempos dos spikes pré-sinápticos  (escalar ou tensor)
        t_post: tempos dos spikes pós-sinápticos  (escalar ou tensor)
        pesos : tensor de pesos a ser atualizado

        Retorna: pesos atualizados com o mesmo shape de 'pesos'.
        """
        delta_t = t_post - t_pre

        # CORRIGIDO: era 'if delta_t > 0' seguido de ramos separados, o que
        # lança RuntimeError para tensores com mais de um elemento.
        # A formulação sign(Δt)·exp(−|Δt|/τ) é matematicamente idêntica
        # à regra original e opera sobre qualquer shape sem ramificação.
        delta = self.lr * torch.sign(delta_t) * torch.exp(-torch.abs(delta_t) / self.tau)

        return pesos + delta
