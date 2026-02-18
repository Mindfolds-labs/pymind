"""
pymind.arquiteturas — Arquiteturas de redes pré-construídas
"""

from .piramidal import PiramidalMNIST
from .funil import FunilMNIST
from .profunda import Profunda
from .autoencoder import AutoencoderMNIST

__all__ = [
    "PiramidalMNIST",
    "FunilMNIST",
    "Profunda",
    "AutoencoderMNIST",
]
