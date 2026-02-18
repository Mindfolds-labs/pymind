# PyMind 🧠

**Rede Neural Artificial com Dendritos e Memória Engram**

[![PyPI version](https://badge.fury.io/py/pymind.svg)](https://pypi.org/project/pymind/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Descrição

PyMind é uma biblioteca para criação de redes neurais inspiradas em neurônios biológicos, com suporte a:

- **Dendritos** — cada neurônio possui múltiplos dendritos com thresholds independentes
- **Filamentos (N)** — memória de longo prazo codificada em pesos discretos
- **Potencial interno (I)** — plasticidade sináptica via LTP/LTD
- **Engram** — memória de protótipos com feedback top-down
- **STDP** — plasticidade dependente do tempo dos spikes (vetorizada)
- **Arquiteturas flexíveis** — piramidal, funil, profunda, autoencoder

## Instalação

```bash
pip install pymind
```

## Uso rápido

```python
import torch
from pymind import PiramidalMNIST, Treinador

# Criar modelo
config = {'n_dendritos': 4, 'n_sinapses': 4, 'usar_engram': True}
modelo = PiramidalMNIST(config)

# Treinar
treinador = Treinador(modelo, device='cpu', learning_rate=0.001)
historico, melhor_acc = treinador.treinar(train_loader, test_loader, epochs=10)
```

## Arquiteturas disponíveis

| Classe | Topologia | Uso |
|--------|-----------|-----|
| `PiramidalMNIST` | 784→128→64→32→10 | Classificação MNIST |
| `FunilMNIST` | 784→49→10 | Classificação rápida com campos receptivos |
| `AutoencoderMNIST` | 784→32→784 | Reconstrução de imagens |
| `Profunda` | Configurável | Qualquer tarefa |

## Componentes core

```python
from pymind import NeuronioDendritico, Camada, ConexaoDensa
from pymind import Engram, MemoriaTrabalho
from pymind import Hebbian, STDP, Homeostase
```

## Requisitos

- Python >= 3.9
- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- Matplotlib >= 3.3.0

## Links

- GitHub: https://github.com/Mindfolds-labs/pymind
- Issues: https://github.com/Mindfolds-labs/pymind/issues
