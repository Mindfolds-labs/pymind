# Script para criar a estrutura do PyMind
Write-Host "🚀 Criando estrutura do PyMind..." -ForegroundColor Cyan

# Criar diretórios principais
$diretorios = @(
    "pymind",
    "pymind/core",
    "pymind/memory",
    "pymind/plasticity",
    "pymind/arquiteturas",
    "pymind/models",
    "pymind/utils",
    "examples",
    "tests",
    "checkpoints",
    "data"
)

foreach ($dir in $diretorios) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Write-Host "  📁 Criado: $dir"
}

# ===== ARQUIVO: pymind/__init__.py =====
@"
"""
PyMind - Pacote principal
Rede Neural Artificial com Dendritos e Memória Engram
"""

from .core.neuronio import NeuronioDendritico
from .core.camada import Camada
from .core.conexoes import ConexaoDensa, ConexaoEsparsa, ConexaoRegional
from .memory.engram import Engram
from .memory.working_memory import MemoriaTrabalho
from .plasticity.hebbian import Hebbian
from .plasticity.stdp import STDP
from .plasticity.homeostasis import Homeostase
from .arquiteturas.piramidal import PiramidalMNIST
from .arquiteturas.funil import FunilMNIST
from .arquiteturas.profunda import Profunda
from .arquiteturas.autoencoder import AutoencoderMNIST
from .models.mnist import MNISTClassifier
from .utils.treino import Treinador
from .utils.visualizacao import *

__version__ = "0.2.0"
__author__ = "Seu Nome"
__email__ = "seu.email@example.com"

__all__ = [
    # Core
    'NeuronioDendritico',
    'Camada',
    'ConexaoDensa',
    'ConexaoEsparsa',
    'ConexaoRegional',
    
    # Memória
    'Engram',
    'MemoriaTrabalho',
    
    # Plasticidade
    'Hebbian',
    'STDP',
    'Homeostase',
    
    # Arquiteturas
    'PiramidalMNIST',
    'FunilMNIST',
    'Profunda',
    'AutoencoderMNIST',
    
    # Modelos
    'MNISTClassifier',
    
    # Utils
    'Treinador',
]
"@ | Out-File -FilePath "pymind/__init__.py" -Encoding UTF8
Write-Host "  📄 pymind/__init__.py"

# ===== ARQUIVO: pymind/core/neuronio.py =====
@"
"""
Neurônio base com dendritos e filamentos (N)
"""

import torch
import torch.nn as nn
import math

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
    
    @property
    def W(self):
        """Pesos derivados dos filamentos: W = log2(1+N)/5"""
        return torch.log2(1.0 + self.N.float()) / 5.0
    
    def forward(self, x):
        """
        x: [batch, n_dendritos, n_sinapses] ou [n_dendritos, n_sinapses]
        
        Returns:
            spikes: [batch] ou escalar
        """
        # Garantir batch dimension
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
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
        
        # Gerar spikes
        spikes = (soma >= self.theta).float()
        
        # Atualizar contadores
        self.ultimo_spike = spikes.mean()
        self.total_spikes += spikes.sum().long()
        
        if batch_size == 1:
            return spikes.squeeze(0)
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
"@ | Out-File -FilePath "pymind/core/neuronio.py" -Encoding UTF8
Write-Host "  📄 pymind/core/neuronio.py"

# ===== ARQUIVO: pymind/core/camada.py =====
@"
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
        batch_size = x.shape[0]
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
"@ | Out-File -FilePath "pymind/core/camada.py" -Encoding UTF8
Write-Host "  📄 pymind/core/camada.py"

# ===== ARQUIVO: pymind/core/conexoes.py =====
@"
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
        
        # Matriz de pesos
        self.peso = nn.Parameter(
            torch.randn(out_features, in_features, n_dendritos, n_sinapses)
        )
    
    def forward(self, x):
        """
        x: [batch, in_features]
        Returns: [batch, out_features, n_dendritos, n_sinapses]
        """
        batch = x.shape[0]
        return torch.einsum('bi,oinds->bonds', x, self.peso)


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
        
        # Calcular grid de regiões
        self.grid_h = in_height // regiao_tamanho
        self.grid_w = in_width // regiao_tamanho
        
        assert out_neurons == self.grid_h * self.grid_w, \
            f"Número de neurônios deve ser {self.grid_h * self.grid_w}"
        
        # Pesos por região
        self.pesos = nn.Parameter(
            torch.randn(out_neurons, n_dendritos, n_sinapses, 
                       regiao_tamanho, regiao_tamanho)
        )
    
    def forward(self, x):
        """
        x: [batch, 1, height, width]
        Returns: [batch, out_neurons, n_dendritos, n_sinapses]
        """
        batch = x.shape[0]
        resultado = []
        
        idx = 0
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                y1, y2 = i * self.regiao_tam, (i + 1) * self.regiao_tam
                x1, x2 = j * self.regiao_tam, (j + 1) * self.regiao_tam
                
                regiao = x[:, :, y1:y2, x1:x2]  # [batch, 1, R, R]
                
                # Aplicar pesos deste neurônio
                peso = self.pesos[idx]  # [D, S, R, R]
                ativ = torch.einsum('bcrr,dsrr->bds', regiao, peso)
                resultado.append(ativ)
                
                idx += 1
        
        return torch.stack(resultado, dim=1)  # [batch, out_n, D, S]


class ConexaoEsparsa(nn.Module):
    """Conexão esparsa aleatória"""
    
    def __init__(self, in_features, out_features, n_dendritos, n_sinapses, 
                 esparsidade=0.1):
        super().__init__()
        
        # Criar máscara esparsa
        mascara = torch.rand(out_features, in_features) < esparsidade
        
        self.register_buffer('mascara', mascara.float())
        self.peso = nn.Parameter(
            torch.randn(out_features, in_features, n_dendritos, n_sinapses)
        )
    
    def forward(self, x):
        """
        x: [batch, in_features]
        Returns: [batch, out_features, n_dendritos, n_sinapses]
        """
        # Aplicar máscara para conexões esparsas
        peso_esparso = self.peso * self.mascara.view(
            self.peso.shape[0], self.peso.shape[1], 1, 1
        )
        
        return torch.einsum('bi,oinds->bonds', x, peso_esparso)
"@ | Out-File -FilePath "pymind/core/conexoes.py" -Encoding UTF8
Write-Host "  📄 pymind/core/conexoes.py"

# ===== ARQUIVO: pymind/memory/engram.py =====
@"
"""
Memória Engram - Protótipos dos padrões de ativação
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Engram(nn.Module):
    """
    Memória de protótipos que observa e lembra padrões
    
    Args:
        dim_entrada: dimensão do padrão de entrada
        max_prototipos: número máximo de protótipos
        limiar_similaridade: similaridade mínima para reforçar (0-1)
        limiar_criacao: erro mínimo para criar novo protótipo
    """
    
    def __init__(self, dim_entrada, max_prototipos=100,
                 limiar_similaridade=0.7, limiar_criacao=0.3):
        super().__init__()
        
        self.dim = dim_entrada
        self.max_protos = max_prototipos
        self.limiar_sim = limiar_similaridade
        self.limiar_novo = limiar_criacao
        
        # Banco de protótipos
        self.register_buffer('prototipos', torch.zeros(max_prototipos, dim_entrada))
        self.register_buffer('forca', torch.zeros(max_prototipos))
        self.register_buffer('idade', torch.zeros(max_prototipos, dtype=torch.long))
        self.register_buffer('classe', torch.zeros(max_prototipos, dtype=torch.long))
        self.register_buffer('n_protos', torch.tensor(0, dtype=torch.long))
        
        # Ganho do feedback
        self.feedback_gain = nn.Parameter(torch.tensor(0.1))
        
        # Estatísticas
        self.register_buffer('total_observacoes', torch.tensor(0, dtype=torch.long))
        self.register_buffer('total_criacoes', torch.tensor(0, dtype=torch.long))
        self.register_buffer('total_reforcos', torch.tensor(0, dtype=torch.long))
    
    def observar(self, padrao, erro=None, classe=None):
        """
        Observa um padrão e retorna feedback
        
        Args:
            padrao: [batch, dim_entrada] tensor
            erro: erro atual (escalar ou [batch])
            classe: classe verdadeira (opcional)
        
        Returns:
            feedback: [batch, dim_entrada] tensor
        """
        if padrao.dim() == 1:
            padrao = padrao.unsqueeze(0)
        
        batch = padrao.shape[0]
        n = self.n_protos.item()
        
        # Processar erro
        if erro is not None:
            if isinstance(erro, (float, int)):
                erro_tensor = torch.full((batch,), float(erro), device=padrao.device)
            else:
                erro_tensor = erro
        
        feedbacks = []
        
        for b in range(batch):
            p = padrao[b]
            err_b = erro_tensor[b].item() if erro is not None else 0
            
            if n == 0:
                # Primeiro protótipo
                self._criar(p, classe[b] if classe is not None else None)
                feedbacks.append(torch.zeros_like(p))
                self.total_criacoes += 1
                continue
            
            # Calcular similaridade com protótipos existentes
            protos = self.prototipos[:n]
            sim = F.cosine_similarity(p.unsqueeze(0), protos)
            max_sim, idx = sim.max(dim=0)
            
            # Decidir ação
            if max_sim >= self.limiar_sim:
                # Reforçar protótipo existente
                self._reforcar(idx, p, classe[b] if classe is not None else None)
                feedback = self._gerar_feedback(idx, p)
                feedbacks.append(feedback)
                self.total_reforcos += 1
                
            elif (err_b > self.limiar_novo) or (n < self.max_protos):
                # Criar novo protótipo
                self._criar(p, classe[b] if classe is not None else None)
                feedbacks.append(torch.zeros_like(p))
                self.total_criacoes += 1
            else:
                # Usar protótipo mais similar mesmo assim
                feedback = self._gerar_feedback(idx, p)
                feedbacks.append(feedback)
        
        self.total_observacoes += batch
        
        resultado = torch.stack(feedbacks) * self.feedback_gain
        return resultado
    
    def _criar(self, padrao, classe=None):
        """Cria um novo protótipo"""
        n = self.n_protos.item()
        
        if n >= self.max_protos:
            # Substituir o mais fraco
            fraco = self.forca[:n].argmin().item()
            idx = fraco
        else:
            idx = n
            self.n_protos += 1
        
        self.prototipos[idx] = padrao.detach().clone()
        self.forca[idx] = 1.0
        self.idade[idx] = 0
        if classe is not None:
            self.classe[idx] = classe
    
    def _reforcar(self, idx, padrao, classe=None):
        """Reforça um protótipo existente"""
        lr = 0.01
        self.prototipos[idx] = (1 - lr) * self.prototipos[idx] + lr * padrao.detach()
        self.forca[idx] += lr
        self.idade[idx] += 1
        if classe is not None:
            # Atualizar classe se consistente
            if self.classe[idx] == classe:
                self.forca[idx] += lr
            else:
                self.forca[idx] *= 0.5
    
    def _gerar_feedback(self, idx, padrao):
        """Gera feedback baseado no protótipo"""
        # Feedback é a diferença ponderada pela força
        return (self.prototipos[idx] - padrao) * self.forca[idx]
    
    def consultar(self, padrao, k=3):
        """Consulta os k protótipos mais similares"""
        n = self.n_protos.item()
        if n == 0:
            return [], []
        
        sim = F.cosine_similarity(padrao.unsqueeze(0), self.prototipos[:n])
        top_sim, top_idx = sim.topk(min(k, n))
        
        return top_idx, top_sim
    
    def get_estatisticas(self):
        """Retorna estatísticas da memória"""
        return {
            'n_prototipos': self.n_protos.item(),
            'total_observacoes': self.total_observacoes.item(),
            'total_criacoes': self.total_criacoes.item(),
            'total_reforcos': self.total_reforcos.item(),
            'forca_media': self.forca[:self.n_protos].mean().item() if self.n_protos > 0 else 0,
            'idade_media': self.idade[:self.n_protos].float().mean().item() if self.n_protos > 0 else 0
        }
    
    def extra_repr(self):
        return (f"dim={self.dim}, protos={self.n_protos.item()}/{self.max_protos}, "
                f"sim_th={self.limiar_sim}, novo_th={self.limiar_novo}")
"@ | Out-File -FilePath "pymind/memory/engram.py" -Encoding UTF8
Write-Host "  📄 pymind/memory/engram.py"

# ===== ARQUIVO: pymind/memory/working_memory.py =====
@"
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
"@ | Out-File -FilePath "pymind/memory/working_memory.py" -Encoding UTF8
Write-Host "  📄 pymind/memory/working_memory.py"

# ===== ARQUIVO: pymind/plasticity/hebbian.py =====
@"
"""
Regra de plasticidade Hebbiana
"""

import torch
import torch.nn as nn

class Hebbian(nn.Module):
    """
    Plasticidade Hebbiana: "cells that fire together, wire together"
    """
    
    def __init__(self, learning_rate=0.01):
        super().__init__()
        self.lr = learning_rate
    
    def forward(self, pre, post, pesos):
        """
        pre: atividade pré-sináptica
        post: atividade pós-sináptica
        pesos: pesos atuais
        """
        delta = self.lr * torch.einsum('bi,bj->bij', pre, post)
        return pesos + delta.mean(dim=0)
"@ | Out-File -FilePath "pymind/plasticity/hebbian.py" -Encoding UTF8
Write-Host "  📄 pymind/plasticity/hebbian.py"

# ===== ARQUIVO: pymind/plasticity/stdp.py =====
@"
"""
STDP (Spike-Timing Dependent Plasticity)
"""

import torch
import torch.nn as nn

class STDP(nn.Module):
    """
    Plasticidade dependente do tempo dos spikes
    """
    
    def __init__(self, learning_rate=0.01, tau=20.0):
        super().__init__()
        self.lr = learning_rate
        self.tau = tau
    
    def forward(self, t_pre, t_post, pesos):
        """
        t_pre: tempos dos spikes pré-sinápticos
        t_post: tempos dos spikes pós-sinápticos
        """
        delta_t = t_post - t_pre
        if delta_t > 0:
            # LTP - pré antes do pós
            delta = self.lr * torch.exp(-delta_t / self.tau)
        else:
            # LTD - pós antes do pré
            delta = -self.lr * torch.exp(delta_t / self.tau)
        
        return pesos + delta
"@ | Out-File -FilePath "pymind/plasticity/stdp.py" -Encoding UTF8
Write-Host "  📄 pymind/plasticity/stdp.py"

# ===== ARQUIVO: pymind/plasticity/homeostasis.py =====
@"
"""
Homeostase - manutenção da atividade neuronal
"""

import torch
import torch.nn as nn

class Homeostase(nn.Module):
    """
    Ajusta thresholds para manter taxa de disparo alvo
    """
    
    def __init__(self, target_rate=0.1, adapt_rate=0.01):
        super().__init__()
        self.target_rate = target_rate
        self.adapt_rate = adapt_rate
    
    def forward(self, spike_rate, theta):
        """
        Ajusta theta baseado na taxa de disparo
        """
        error = spike_rate - self.target_rate
        return theta + self.adapt_rate * error
"@ | Out-File -FilePath "pymind/plasticity/homeostasis.py" -Encoding UTF8
Write-Host "  📄 pymind/plasticity/homeostasis.py"

# ===== ARQUIVO: pymind/arquiteturas/piramidal.py =====
@"
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
    Rede neural em formato piramidal com memória Engram
    
    Args:
        config: dicionário com configurações
            - n_dendritos: dendritos por neurônio
            - n_sinapses: sinapses por dendrito
            - usar_engram: ativar memória engram
            - max_prototipos: máximo de protótipos por engram
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
        self.proj2 = ConexaoDensa(128, 64, D, S)
        self.proj3 = ConexaoDensa(64, 32, D, S)
        self.proj4 = ConexaoDensa(32, 10, D, S)
        
        # Camadas da pirâmide
        self.camada1 = Camada(128, D, S, nome="piramide_1")
        self.camada2 = Camada(64, D, S, nome="piramide_2")
        self.camada3 = Camada(32, D, S, nome="piramide_3")
        self.camada4 = Camada(10, D, S, nome="piramide_4")
        
        # Engrams opcionais
        self.engrams = nn.ModuleDict()
        if config.get('usar_engram', False):
            max_protos = config.get('max_prototipos', 100)
            self.engrams['camada1'] = Engram(128, max_protos)
            self.engrams['camada2'] = Engram(64, max_protos)
            self.engrams['camada3'] = Engram(32, max_protos)
            self.engrams['saida'] = Engram(10, max_protos)
        
        # Cache para feedback
        self.feedback_cache = {}
        self.ativacoes = {}
    
    def forward(self, x, erro=None):
        """
        x: [batch, 1, 28, 28] ou [batch, 784]
        erro: erro atual para o engram
        """
        # Flatten se necessário
        if x.dim() == 4:
            batch = x.shape[0]
            x = x.view(batch, -1)  # [batch, 784]
        
        # Camada 1: 128 neurônios
        x = self.proj1(x)  # [batch, 128, D, S]
        
        if 'camada1' in self.feedback_cache:
            x = x + 0.1 * self.feedback_cache['camada1']
        
        spikes1 = self.camada1(x)
        self.ativacoes['camada1'] = spikes1.detach()
        
        if erro is not None and 'camada1' in self.engrams:
            self.feedback_cache['camada1'] = self.engrams['camada1'].observar(spikes1, erro)
        
        # Camada 2: 64 neurônios
        x = self.proj2(spikes1)
        
        if 'camada2' in self.feedback_cache:
            x = x + 0.1 * self.feedback_cache['camada2']
        
        spikes2 = self.camada2(x)
        self.ativacoes['camada2'] = spikes2.detach()
        
        if erro is not None and 'camada2' in self.engrams:
            self.feedback_cache['camada2'] = self.engrams['camada2'].observar(spikes2, erro)
        
        # Camada 3: 32 neurônios
        x = self.proj3(spikes2)
        
        if 'camada3' in self.feedback_cache:
            x = x + 0.1 * self.feedback_cache['camada3']
        
        spikes3 = self.camada3(x)
        self.ativacoes['camada3'] = spikes3.detach()
        
        if erro is not None and 'camada3' in self.engrams:
            self.feedback_cache['camada3'] = self.engrams['camada3'].observar(spikes3, erro)
        
        # Camada 4: 10 neurônios (saída)
        x = self.proj4(spikes3)
        spikes4 = self.camada4(x)
        self.ativacoes['saida'] = spikes4.detach()
        
        if erro is not None and 'saida' in self.engrams:
            self.engrams['saida'].observar(spikes4, erro)
        
        return spikes4
    
    def get_ativacoes(self):
        """Retorna ativações para visualização"""
        return self.ativacoes
    
    def get_engram_stats(self):
        """Retorna estatísticas dos engrams"""
        stats = {}
        for nome, engram in self.engrams.items():
            stats[nome] = engram.get_estatisticas()
        return stats
"@ | Out-File -FilePath "pymind/arquiteturas/piramidal.py" -Encoding UTF8
Write-Host "  📄 pymind/arquiteturas/piramidal.py"

# ===== ARQUIVO: pymind/arquiteturas/funil.py =====
@"
"""
Arquitetura Funil para MNIST (rápida)
784 → 49 → 10
"""

import torch
import torch.nn as nn
from ..core.camada import Camada
from ..core.conexoes import ConexaoRegional, ConexaoDensa

class FunilMNIST(nn.Module):
    """
    Arquitetura em funil: 784 pixels → 49 neurônios regionais → 10 classes
    """
    
    def __init__(self, n_dendritos=4, n_sinapses=4):
        super().__init__()
        
        self.n_dendritos = n_dendritos
        self.n_sinapses = n_sinapses
        
        # Conexão regional: cada neurônio vê uma região 4x4 da imagem
        self.regional = ConexaoRegional(
            in_height=28, in_width=28,
            out_neurons=49,  # 7x7 grid
            n_dendritos=n_dendritos,
            n_sinapses=n_sinapses,
            regiao_tamanho=4
        )
        
        # Camada de neurônios regionais
        self.camada_regional = Camada(49, n_dendritos, n_sinapses, nome="regional")
        
        # Conexão para classificação
        self.classif = ConexaoDensa(49, 10, n_dendritos, n_sinapses)
        
        # Camada de saída
        self.camada_saida = Camada(10, n_dendritos, n_sinapses, nome="saida")
    
    def forward(self, x):
        """
        x: [batch, 1, 28, 28]
        """
        # Conexão regional
        x = self.regional(x)  # [batch, 49, D, S]
        
        # Neurônios regionais
        spikes_reg = self.camada_regional(x)  # [batch, 49]
        
        # Conexão para classificação
        x = self.classif(spikes_reg)  # [batch, 10, D, S]
        
        # Camada de saída
        saida = self.camada_saida(x)  # [batch, 10]
        
        return saida
"@ | Out-File -FilePath "pymind/arquiteturas/funil.py" -Encoding UTF8
Write-Host "  📄 pymind/arquiteturas/funil.py"

# ===== ARQUIVO: pymind/arquiteturas/profunda.py =====
@"
"""
Arquitetura Profunda para tarefas complexas
"""

import torch
import torch.nn as nn
from ..core.camada import Camada
from ..core.conexoes import ConexaoDensa

class Profunda(nn.Module):
    """
    Arquitetura profunda genérica
    """
    
    def __init__(self, dims, n_dendritos=4, n_sinapses=4):
        super().__init__()
        
        self.camadas = nn.ModuleList()
        self.projecoes = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            self.projecoes.append(
                ConexaoDensa(dims[i], dims[i+1], n_dendritos, n_sinapses)
            )
            self.camadas.append(
                Camada(dims[i+1], n_dendritos, n_sinapses, nome=f"profunda_{i}")
            )
    
    def forward(self, x):
        for proj, camada in zip(self.projecoes, self.camadas):
            x = proj(x)
            x = camada(x)
        return x
"@ | Out-File -FilePath "pymind/arquiteturas/profunda.py" -Encoding UTF8
Write-Host "  📄 pymind/arquiteturas/profunda.py"

# ===== ARQUIVO: pymind/arquiteturas/autoencoder.py =====
@"
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
"@ | Out-File -FilePath "pymind/arquiteturas/autoencoder.py" -Encoding UTF8
Write-Host "  📄 pymind/arquiteturas/autoencoder.py"

# ===== ARQUIVO: pymind/models/mnist.py =====
@"
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
"@ | Out-File -FilePath "pymind/models/mnist.py" -Encoding UTF8
Write-Host "  📄 pymind/models/mnist.py"

# ===== ARQUIVO: pymind/utils/treino.py =====
@"
"""
Treinador para modelos PyMind
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import os
import json

class Treinador:
    """
    Classe para treinar modelos PyMind
    
    Args:
        modelo: modelo PyMind
        device: dispositivo ('cuda' ou 'cpu')
        learning_rate: taxa de aprendizado
    """
    
    def __init__(self, modelo, device='cuda', learning_rate=0.001):
        self.modelo = modelo
        self.device = device
        self.modelo.to(device)
        
        self.otimizador = torch.optim.Adam(modelo.parameters(), lr=learning_rate)
        self.criterio = nn.CrossEntropyLoss()
        
        self.historico = {
            'train_loss': [],
            'train_acc': [],
            'test_acc': [],
            'epochs': []
        }
    
    def treinar_epoca(self, loader):
        """Treina uma época"""
        self.modelo.train()
        loss_total = 0
        acertos = 0
        total = 0
        
        for imagens, rotulos in loader:
            imagens, rotulos = imagens.to(self.device), rotulos.to(self.device)
            
            # Forward
            saida = self.modelo(imagens)
            loss = self.criterio(saida, rotulos)
            
            # Backward
            self.otimizador.zero_grad()
            loss.backward()
            self.otimizador.step()
            
            # Métricas
            loss_total += loss.item()
            preditos = saida.argmax(dim=1)
            acertos += (preditos == rotulos).sum().item()
            total += rotulos.size(0)
        
        return loss_total / len(loader), 100.0 * acertos / total
    
    def avaliar(self, loader):
        """Avalia o modelo"""
        self.modelo.eval()
        acertos = 0
        total = 0
        
        with torch.no_grad():
            for imagens, rotulos in loader:
                imagens, rotulos = imagens.to(self.device), rotulos.to(self.device)
                saida = self.modelo(imagens)
                preditos = saida.argmax(dim=1)
                acertos += (preditos == rotulos).sum().item()
                total += rotulos.size(0)
        
        return 100.0 * acertos / total
    
    def treinar(self, train_loader, test_loader, epochs=10, 
                save_dir='checkpoints', verbose=True):
        """
        Treina o modelo por múltiplas épocas
        
        Args:
            train_loader: DataLoader de treino
            test_loader: DataLoader de teste
            epochs: número de épocas
            save_dir: diretório para salvar checkpoints
            verbose: mostrar progresso
        
        Returns:
            historico: dicionário com histórico de treino
            melhor_acc: melhor acurácia obtida
        """
        os.makedirs(save_dir, exist_ok=True)
        melhor_acc = 0.0
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🚀 Treinando {self.modelo.__class__.__name__}")
            print(f"📅 Início: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            print(f"💻 Device: {self.device}")
            print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            # Treino
            loss, acc_treino = self.treinar_epoca(train_loader)
            
            # Avaliação
            acc_teste = self.avaliar(test_loader)
            
            # Histórico
            self.historico['train_loss'].append(loss)
            self.historico['train_acc'].append(acc_treino)
            self.historico['test_acc'].append(acc_teste)
            self.historico['epochs'].append(epoch + 1)
            
            if verbose:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Loss: {loss:.4f} | "
                      f"Train: {acc_treino:.2f}% | "
                      f"Test: {acc_teste:.2f}%")
            
            # Salvar melhor modelo
            if acc_teste > melhor_acc:
                melhor_acc = acc_teste
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.modelo.state_dict(),
                    'optimizer_state_dict': self.otimizador.state_dict(),
                    'acc_teste': acc_teste,
                    'config': getattr(self.modelo, 'config', {})
                }
                path = os.path.join(save_dir, f'melhor_{acc_teste:.2f}.pth')
                torch.save(checkpoint, path)
                
                if verbose:
                    print(f"  ✓ Novo recorde: {acc_teste:.2f}% -> {path}")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🏆 Melhor acurácia: {melhor_acc:.2f}%")
            print(f"{'='*60}")
        
        return self.historico, melhor_acc
    
    def salvar_historico(self, path='historico.json'):
        """Salva histórico em JSON"""
        with open(path, 'w') as f:
            json.dump(self.historico, f, indent=2)
"@ | Out-File -FilePath "pymind/utils/treino.py" -Encoding UTF8
Write-Host "  📄 pymind/utils/treino.py"

# ===== ARQUIVO: pymind/utils/visualizacao.py =====
@"
"""
Funções de visualização para PyMind
"""

import matplotlib.pyplot as plt
import torch
import numpy as np

def visualizar_neuronio(neuronio):
    """Visualiza o estado de um neurônio"""
    estado = neuronio.get_estado()
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Filamentos (N)
    ax = axes[0,0]
    im = ax.imshow(estado['N'], cmap='hot', aspect='auto')
    ax.set_title(f'Filamentos (N) - {estado["nome"]}')
    ax.set_xlabel('Sinapses')
    ax.set_ylabel('Dendritos')
    plt.colorbar(im, ax=ax)
    
    # Potencial interno (I)
    ax = axes[0,1]
    im = ax.imshow(estado['I'], cmap='coolwarm', aspect='auto')
    ax.set_title('Potencial Interno (I)')
    ax.set_xlabel('Sinapses')
    ax.set_ylabel('Dendritos')
    plt.colorbar(im, ax=ax)
    
    # Threshold dos dendritos
    ax = axes[1,0]
    ax.bar(range(len(estado['thetas_dendritos'])), estado['thetas_dendritos'])
    ax.set_title('Threshold dos Dendritos')
    ax.set_xlabel('Dendrito')
    ax.set_ylabel('Threshold')
    ax.axhline(y=estado['theta'], color='r', linestyle='--', label='Theta neurônio')
    ax.legend()
    
    # Spike info
    ax = axes[1,1]
    ax.text(0.5, 0.7, f"Spike atual: {estado['ultimo_spike']:.2f}", 
            ha='center', va='center', fontsize=14)
    ax.text(0.5, 0.5, f"Total spikes: {estado['total_spikes']}", 
            ha='center', va='center', fontsize=14)
    ax.text(0.5, 0.3, f"Theta: {estado['theta']:.2f}", 
            ha='center', va='center', fontsize=14)
    ax.set_title('Informações')
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def visualizar_piramide(modelo, imagem, rotulo=None):
    """
    Visualiza as ativações da rede piramidal
    
    Args:
        modelo: modelo PiramidalMNIST
        imagem: tensor [1,28,28] ou [28,28]
        rotulo: rótulo verdadeiro (opcional)
    """
    modelo.eval()
    
    if imagem.dim() == 2:
        imagem = imagem.unsqueeze(0).unsqueeze(0)
    elif imagem.dim() == 3:
        imagem = imagem.unsqueeze(0)
    
    with torch.no_grad():
        saida = modelo(imagem)
        ativacoes = modelo.get_ativacoes()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Imagem original
    ax = axes[0,0]
    ax.imshow(imagem[0,0].cpu(), cmap='gray')
    ax.set_title(f'Imagem Original\nRótulo: {rotulo}' if rotulo else 'Imagem Original')
    ax.axis('off')
    
    # Camada 1 (128 neurônios)
    ax = axes[0,1]
    if 'camada1' in ativacoes:
        ativ1 = ativacoes['camada1'][0].cpu().numpy()
        ax.bar(range(len(ativ1)), ativ1)
        ax.set_title('Camada 1: 128 neurônios')
        ax.set_xlabel('Neurônio')
        ax.set_ylabel('Ativação')
    
    # Camada 2 (64 neurônios)
    ax = axes[0,2]
    if 'camada2' in ativacoes:
        ativ2 = ativacoes['camada2'][0].cpu().numpy()
        ax.bar(range(len(ativ2)), ativ2)
        ax.set_title('Camada 2: 64 neurônios')
        ax.set_xlabel('Neurônio')
        ax.set_ylabel('Ativação')
    
    # Camada 3 (32 neurônios)
    ax = axes[1,0]
    if 'camada3' in ativacoes:
        ativ3 = ativacoes['camada3'][0].cpu().numpy()
        ax.bar(range(len(ativ3)), ativ3)
        ax.set_title('Camada 3: 32 neurônios')
        ax.set_xlabel('Neurônio')
        ax.set_ylabel('Ativação')
    
    # Saída (10 neurônios)
    ax = axes[1,1]
    if 'saida' in ativacoes:
        ativ4 = ativacoes['saida'][0].cpu().numpy()
        cores = ['green' if i == ativ4.argmax() else 'gray' for i in range(10)]
        ax.bar(range(10), ativ4, color=cores)
        ax.set_title('Saída: 10 classes')
        ax.set_xlabel('Classe')
        ax.set_ylabel('Ativação')
        ax.set_xticks(range(10))
    
    # Predição
    ax = axes[1,2]
    pred = saida[0].argmax().item()
    prob = torch.softmax(saida[0], dim=0)[pred].item()
    ax.text(0.5, 0.6, f'Predição: {pred}', ha='center', va='center', fontsize=20)
    ax.text(0.5, 0.4, f'Confiança: {prob:.2f}', ha='center', va='center', fontsize=14)
    if rotulo is not None:
        acerto = pred == rotulo
        cor = 'green' if acerto else 'red'
        ax.text(0.5, 0.2, f'{"✅ ACERTOU" if acerto else "❌ ERROU"}', 
                ha='center', va='center', fontsize=14, color=cor)
    ax.axis('off')
    
    plt.suptitle('Arquitetura Piramidal - Fluxo de Ativação')
    plt.tight_layout()
    return fig


def visualizar_engram(engram, k=10):
    """Visualiza os protótipos do engram"""
    stats = engram.get_estatisticas()
    n = stats['n_prototipos']
    
    if n == 0:
        fig, ax = plt.subplots(1, 1, figsize=(6,4))
        ax.text(0.5, 0.5, 'Engram vazio', ha='center', va='center')
        return fig
    
    k = min(k, n)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(k):
        ax = axes[i]
        proto = engram.prototipos[i].cpu().numpy()
        ax.plot(proto)
        ax.set_title(f'Protótipo {i}\nForça: {engram.forca[i]:.2f}')
        ax.set_xlabel('Dimensão')
    
    for i in range(k, 10):
        axes[i].axis('off')
    
    plt.suptitle(f'Engram - {n} protótipos criados')
    plt.tight_layout()
    return fig
"@ | Out-File -FilePath "pymind/utils/visualizacao.py" -Encoding UTF8
Write-Host "  📄 pymind/utils/visualizacao.py"

# ===== ARQUIVO: examples/treinar_mnist_piramidal.py =====
@"
#!/usr/bin/env python
"""
Exemplo de treino da rede piramidal no MNIST
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import sys
sys.path.append('..')

from pymind import PiramidalMNIST, Treinador
from pymind.utils.visualizacao import visualizar_piramide

def main():
    # Configurações
    config = {
        'n_dendritos': 4,
        'n_sinapses': 4,
        'usar_engram': True,
        'max_prototipos': 100
    }
    
    batch_size = 128
    epochs = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("🧠 PyMind - Treinamento MNIST com Arquitetura Piramidal")
    print("="*60)
    
    # Carregar dados
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_set = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    
    print(f"\n📊 Dados:")
    print(f"  Treino: {len(train_set)} imagens")
    print(f"  Teste: {len(test_set)} imagens")
    print(f"  Batch size: {batch_size}")
    
    # Criar modelo
    modelo = PiramidalMNIST(config)
    total_params = sum(p.numel() for p in modelo.parameters())
    print(f"\n🔧 Modelo criado:")
    print(f"  Arquitetura: 784 → 128 → 64 → 32 → 10")
    print(f"  Parâmetros: {total_params:,}")
    print(f"  Engram: {'Ativado' if config['usar_engram'] else 'Desativado'}")
    
    # Treinar
    treinador = Treinador(modelo, device=device, learning_rate=0.001)
    historico, melhor_acc = treinador.treinar(
        train_loader, test_loader, epochs=epochs)
    
    # Visualizar exemplo
    print("\n🎨 Gerando visualização...")
    imagens, rotulos = next(iter(test_loader))
    fig = visualizar_piramide(modelo, imagens[0], rotulos[0].item())
    fig.savefig('piramidal_exemplo.png')
    print("  Visualização salva em 'piramidal_exemplo.png'")
    
    # Estatísticas do engram
    if hasattr(modelo, 'get_engram_stats'):
        stats = modelo.get_engram_stats()
        print("\n📊 Estatísticas do Engram:")
        for nome, stat in stats.items():
            print(f"  {nome}: {stat['n_prototipos']} protótipos, "
                  f"{stat['total_criacoes']} criações, "
                  f"{stat['total_reforcos']} reforços")
    
    print(f"\n✅ Treino concluído!")
    print(f"  Melhor acurácia: {melhor_acc:.2f}%")

if __name__ == "__main__":
    main()
"@ | Out-File -FilePath "examples/treinar_mnist_piramidal.py" -Encoding UTF8
Write-Host "  📄 examples/treinar_mnist_piramidal.py"

# ===== ARQUIVO: examples/visualizar_neuronio.py =====
@"
#!/usr/bin/env python
"""
Exemplo de visualização de um neurônio
"""

import torch
import sys
sys.path.append('..')

from pymind import NeuronioDendritico
from pymind.utils.visualizacao import visualizar_neuronio
import matplotlib.pyplot as plt

def main():
    # Criar neurônio
    neuronio = NeuronioDendritico(
        n_dendritos=8,
        n_sinapses_por_dendrito=8,
        nome="neuronio_exemplo"
    )
    
    # Simular algumas ativações
    for i in range(10):
        entrada = torch.randn(8, 8) * 0.5
        saida = neuronio(entrada)
        print(f"Entrada {i+1}: spike = {saida.item()}")
    
    # Visualizar
    fig = visualizar_neuronio(neuronio)
    fig.savefig('neuronio_exemplo.png')
    print("\n✅ Visualização salva em 'neuronio_exemplo.png'")
    plt.show()

if __name__ == "__main__":
    main()
"@ | Out-File -FilePath "examples/visualizar_neuronio.py" -Encoding UTF8
Write-Host "  📄 examples/visualizar_neuronio.py"

# ===== ARQUIVO: setup.py =====
@"
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pymind",
    version="0.2.0",
    author="Seu Nome",
    author_email="seu.email@example.com",
    description="PyMind - Rede Neural com Dendritos e Memória Engram",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seuusername/pymind",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
    ],
)
"@ | Out-File -FilePath "setup.py" -Encoding UTF8
Write-Host "  📄 setup.py"

