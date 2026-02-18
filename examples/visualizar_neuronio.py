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
