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
