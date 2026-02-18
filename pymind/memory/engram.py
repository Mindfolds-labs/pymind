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
