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

__version__ = "1.0.0"
__author__ = "Mindfolds Labs"

__all__ = [
    'NeuronioDendritico', 'Camada',
    'ConexaoDensa', 'ConexaoEsparsa', 'ConexaoRegional',
    'Engram', 'MemoriaTrabalho',
    'Hebbian', 'STDP', 'Homeostase',
    'PiramidalMNIST', 'FunilMNIST', 'Profunda', 'AutoencoderMNIST',
    'MNISTClassifier', 'Treinador',
]
