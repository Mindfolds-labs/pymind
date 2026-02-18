"""
Neurônio base com dendritos e filamentos (N)
"""

import torch
import torch.nn as nn


class _SpikeSurrogate(torch.autograd.Function):
    """
    Spike binário com gradiente substituto (straight-through).

    Forward: degrau duro (0/1)
    Backward: derivada suave via sigmoid(beta * x)
    """

    @staticmethod
    def forward(ctx, input_tensor, beta):
        ctx.save_for_backward(input_tensor)
        ctx.beta = beta
        return (input_tensor >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input_tensor,) = ctx.saved_tensors
        beta = ctx.beta
        sig = torch.sigmoid(beta * input_tensor)
        surrogate_grad = beta * sig * (1.0 - sig)
        return grad_output * surrogate_grad, None

class NeuronioDendritico(nn.Module):
    """
    Neurônio configurável com múltiplos dendritos
    
    Args:
        n_dendritos: número de dendritos
        n_sinapses_por_dendrito: número de sinapses por dendrito
        theta_inicial: threshold inicial do neurônio
        theta_dendrito_inicial: threshold inicial dos dendritos
        nome: identificador do neurônio
    """
    
    def __init__(self, n_dendritos=4, n_sinapses_por_dendrito=4,
                 theta_inicial=2.0, theta_dendrito_inicial=0.5,
                 nome=None):
        super().__init__()
        
        self.n_dendritos = n_dendritos
        self.n_sinapses = n_sinapses_por_dendrito
        self.nome = nome or f"neuronio_{id(self)}"
        
        # Filamentos (N) - memória de longo prazo (0-31)
        self.register_buffer(
            "N",
            torch.randint(0, 8, (n_dendritos, n_sinapses_por_dendrito))
        )
        
        # Potencial interno (I) - plasticidade
        self.register_buffer(
            "I",
            torch.zeros(n_dendritos, n_sinapses_por_dendrito)
        )
        
        # Threshold do neurônio
        self.theta = nn.Parameter(torch.tensor(theta_inicial, dtype=torch.float32))
        
        # Threshold dos dendritos
        self.thetas_dendritos = nn.Parameter(
            torch.ones(n_dendritos, dtype=torch.float32) * theta_dendrito_inicial
        )
        
        # Registros para análise
        self.register_buffer("ultimo_potencial", torch.zeros(n_dendritos))
        self.register_buffer("ultimo_spike", torch.zeros(1))
        self.register_buffer("total_spikes", torch.zeros(1, dtype=torch.long))

        # Controle do gradiente substituto para permitir treino por backprop
        self.beta_surrogate = 5.0
    
    @property
    def W(self):
        """Pesos derivados dos filamentos: W = log2(1+N)/5"""
        return torch.log2(1.0 + self.N.float()) / 5.0
    
    def forward(self, x):
        """
        x: [batch, n_dendritos, n_sinapses] ou [n_dendritos, n_sinapses]
        
        Returns:
            spikes: [batch]
        """
        # Garantir batch dimension
        if x.dim() == 2:
            x = x.unsqueeze(0)
        

        # Calcular potenciais das sinapses: [batch, dendritos, sinapses]
        potenciais_sinapses = x * self.W.unsqueeze(0)
        
        # Soma por dendrito: [batch, dendritos]
        potenciais_dendritos = potenciais_sinapses.sum(dim=-1)
        
        # Aplicar threshold dos dendritos
        mascara = (potenciais_dendritos >= self.thetas_dendritos.unsqueeze(0)).float()
        potenciais_filtrados = potenciais_dendritos * mascara
        
        # Guardar para análise (média no batch)
        self.ultimo_potencial = potenciais_filtrados.mean(dim=0)
        
        # Soma total (só dendritos que passaram)
        soma = potenciais_filtrados.sum(dim=-1)  # [batch]
        
        # Gerar spikes com função degrau no forward e gradiente substituto
        # no backward para viabilizar otimização.
        spikes = _SpikeSurrogate.apply(soma - self.theta, self.beta_surrogate)
        
        # Atualizar contadores
        self.ultimo_spike = spikes.mean()
        self.total_spikes += spikes.sum().long()
        
        return spikes
    
    def atualizar_plasticidade(self, delta_I):
        """Atualiza potencial interno (chamado pelo otimizador)"""
        self.I.data += delta_I
        
        # Aplicar LTP/LTD baseado no I
        # LTP: se I > threshold, aumenta N
        ltp_mask = self.I > 5.0
        if ltp_mask.any():
            self.N.data[ltp_mask] = torch.clamp(self.N[ltp_mask] + 1, 0, 31)
            self.I.data[ltp_mask] = 0.0
        
        # LTD: se I < -5.0, diminui N
        ltd_mask = self.I < -5.0
        if ltd_mask.any():
            self.N.data[ltd_mask] = torch.clamp(self.N[ltd_mask] - 1, 0, 31)
            self.I.data[ltd_mask] = 0.0
        
        # Decaimento natural
        self.I.data *= 0.995
    
    def get_estado(self):
        """Retorna estado atual para visualização"""
        return {
            'nome': self.nome,
            'N': self.N.cpu().numpy(),
            'I': self.I.cpu().numpy(),
            'theta': self.theta.item(),
            'thetas_dendritos': self.thetas_dendritos.cpu().numpy(),
            'ultimo_potencial': self.ultimo_potencial.cpu().numpy(),
            'ultimo_spike': self.ultimo_spike.item(),
            'total_spikes': self.total_spikes.item()
        }
    
    def extra_repr(self):
        return (f"n_dendritos={self.n_dendritos}, "
                f"n_sinapses={self.n_sinapses}, "
                f"theta={self.theta.item():.2f}")
