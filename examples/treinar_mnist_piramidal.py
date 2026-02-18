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
