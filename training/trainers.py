import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List
import numpy as np

class HallucinationTrainer:
    def __init__(self, model, train_loader: DataLoader, val_loader: DataLoader, 
                 optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            
            source_modalities = self._prepare_source_modalities(batch)
            target_modality = batch['target_modality'][0]
            target_data = self._get_target_data(batch, target_modality)
            
            generated = self.model.hallucinate_modality(source_modalities, target_modality)
            
            loss = self.criterion(generated, target_data, source_modalities, target_modality)
            loss.backward()
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                source_modalities = self._prepare_source_modalities(batch)
                target_modality = batch['target_modality'][0]
                target_data = self._get_target_data(batch, target_modality)
                
                generated = self.model.hallucinate_modality(source_modalities, target_modality)
                
                loss = self.criterion(generated, target_data, source_modalities, target_modality)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
        
        return avg_loss
    
    def _prepare_source_modalities(self, batch: Dict) -> Dict[str, torch.Tensor]:
        source_modalities = {}
        
        if 'text' in batch and batch['text'][0]:
            source_modalities['text'] = batch['text'][0].to(self.device)
        if 'image_path' in batch and batch['image_path'][0]:
            source_modalities['image'] = batch['image_path'][0].to(self.device)
        if 'audio_path' in batch and batch['audio_path'][0]:
            source_modalities['audio'] = batch['audio_path'][0].to(self.device)
        if 'video_path' in batch and batch['video_path'][0]:
            source_modalities['video'] = batch['video_path'][0].to(self.device)
        
        return source_modalities
    
    def _get_target_data(self, batch: Dict, target_modality: str) -> torch.Tensor:
        if target_modality == 'text':
            return batch.get('text_target', torch.zeros(1, 100)).to(self.device)
        elif target_modality == 'image':
            return batch.get('image_target', torch.zeros(1, 3, 224, 224)).to(self.device)
        elif target_modality == 'audio':
            return batch.get('audio_target', torch.zeros(1, 110250)).to(self.device)
        else:
            return torch.zeros(1, 512).to(self.device)
    
    def train(self, num_epochs: int, save_path: str = None):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if save_path and val_loss == self.best_val_loss:
                self.save_checkpoint(save_path, epoch)
    
    def save_checkpoint(self, path: str, epoch: int):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, path)