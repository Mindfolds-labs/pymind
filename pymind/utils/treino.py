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
