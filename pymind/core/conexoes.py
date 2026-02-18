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

        # Pesos: (out, in, dendritos, sinapses)
        self.peso = nn.Parameter(
            torch.randn(out_features, in_features, n_dendritos, n_sinapses)
        )

    def forward(self, x):
        """
        x    : [batch, in_features]
        saída: [batch, out_features, n_dendritos, n_sinapses]

        Contração correta:
            z_{b,o,d,s} = Σ_i  x_{b,i} · W_{o,i,d,s}
        Notação Einstein: 'bi,oids->bods'
        (índices: b=batch, i=entrada, o=saída, d=dendrito, s=sinapse)
        """
        # CORRIGIDO: era 'bi,oinds->bonds' — índice 'n' fantasma que não existe
        # no tensor de pesos (4-D), causava RuntimeError imediato.
        return torch.einsum('bi,oids->bods', x, self.peso)


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

        # Grid de regiões
        self.grid_h = in_height // regiao_tamanho
        self.grid_w = in_width // regiao_tamanho

        assert out_neurons == self.grid_h * self.grid_w, (
            f"Número de neurônios deve ser {self.grid_h * self.grid_w}"
        )

        # Pesos por região: (out_neurons, dendritos, sinapses, R_h, R_w)
        self.pesos = nn.Parameter(
            torch.randn(out_neurons, n_dendritos, n_sinapses,
                        regiao_tamanho, regiao_tamanho)
        )

    def forward(self, x):
        """
        x    : [batch, 1, height, width]
        saída: [batch, out_neurons, n_dendritos, n_sinapses]

        Contração correta por região:
            z_{b,d,s} = Σ_h Σ_w  entrada_{b,c,h,w} · W_{d,s,h,w}
        Notação Einstein: 'bchw,dshw->bds'
        (h e w são índices independentes — altura e largura)
        """
        resultado = []
        idx = 0

        for i in range(self.grid_h):
            for j in range(self.grid_w):
                y1, y2 = i * self.regiao_tam, (i + 1) * self.regiao_tam
                x1, x2 = j * self.regiao_tam, (j + 1) * self.regiao_tam

                regiao = x[:, :, y1:y2, x1:x2]   # [batch, 1, R, R]
                peso   = self.pesos[idx]            # [D, S, R, R]

                # CORRIGIDO: era 'bcrr,dsrr->bds' — usar 'r' duas vezes no
                # mesmo operando calcula o traço (diagonal), não a soma 2-D
                # completa sobre altura e largura. Resultado: perda de ~75 %
                # da informação espacial.
                ativ = torch.einsum('bchw,dshw->bds', regiao, peso)
                resultado.append(ativ)
                idx += 1

        return torch.stack(resultado, dim=1)   # [batch, out_n, D, S]


class ConexaoEsparsa(nn.Module):
    """Conexão esparsa aleatória"""

    def __init__(self, in_features, out_features, n_dendritos, n_sinapses,
                 esparsidade=0.1):
        super().__init__()

        mascara = torch.rand(out_features, in_features) < esparsidade
        self.register_buffer('mascara', mascara.float())

        # Pesos: (out, in, dendritos, sinapses)
        self.peso = nn.Parameter(
            torch.randn(out_features, in_features, n_dendritos, n_sinapses)
        )

    def forward(self, x):
        """
        x    : [batch, in_features]
        saída: [batch, out_features, n_dendritos, n_sinapses]

        Mesma correção de ConexaoDensa aplicada aqui.
        """
        peso_esparso = self.peso * self.mascara.view(
            self.peso.shape[0], self.peso.shape[1], 1, 1
        )
        # CORRIGIDO: era 'bi,oinds->bonds' — mesmo erro de índice fantasma.
        return torch.einsum('bi,oids->bods', x, peso_esparso)
