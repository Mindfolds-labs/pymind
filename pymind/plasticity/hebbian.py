"""
Regra de plasticidade Hebbiana
"""

import torch
import torch.nn as nn


class Hebbian(nn.Module):
    """
    Plasticidade Hebbiana: "cells that fire together, wire together".

    Regra clássica (Hebb, 1949):
        ΔW_{ij} = η · a_i^{pre} · a_j^{post}

    Em forma matricial para batch B:
        ΔW = (η / B) · (A^{pre})ᵀ · A^{post}
        shape resultado: (i, j)

    Para pesos dendríticos de shape (out=j, in=i, D, S), o delta deve ser
    transposto para (j, i) e então expandido via view_as para cobrir as
    dimensões dendríticas uniformemente — o sinal hebbiano é o mesmo para
    todos os dendritos e sinapses de cada par (pré, pós).
    """

    def __init__(self, learning_rate=0.01):
        super().__init__()
        self.lr = learning_rate

    def forward(self, pre, post, pesos):
        """
        pre   : atividade pré-sináptica   [batch, in_features]
        post  : atividade pós-sináptica   [batch, out_features]
        pesos : tensor de pesos           [out_features, in_features, ...]

        Retorna: pesos atualizados com o mesmo shape de 'pesos'.
        """
        # ΔW: média de batch → shape (in_features, out_features)
        delta = self.lr * torch.einsum('bi,bj->ij', pre, post) / pre.shape[0]

        # Transpor para (out, in) — alinhado com a convenção (out, in, D, S)
        delta = delta.T   # shape (out_features, in_features)

        # CORRIGIDO: antes somava delta de shape (i, j) diretamente em pesos
        # de shape (j, i, D, S), causando mismatch silencioso ou RuntimeError.
        # view_as expande (out, in) → (out, in, 1, 1, ...) por broadcasting,
        # aplicando o mesmo Δ a todos os dendritos e sinapses de cada conexão.
        return pesos + delta.view_as(pesos[:, :, 0:1, 0:1].expand_as(pesos[:, :, :1, :1])).expand_as(pesos)
