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
