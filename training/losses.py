import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 0.1, gamma: float = 0.01):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, generated: torch.Tensor, target: torch.Tensor, 
                source_modalities: Dict, target_modality: str) -> torch.Tensor:
        
        reconstruction_loss = self._compute_reconstruction_loss(generated, target, target_modality)
        consistency_loss = self._compute_consistency_loss(generated, source_modalities, target_modality)
        perceptual_loss = self._compute_perceptual_loss(generated, target, target_modality)
        
        total_loss = (self.alpha * reconstruction_loss + 
                     self.beta * consistency_loss + 
                     self.gamma * perceptual_loss)
        
        return total_loss
    
    def _compute_reconstruction_loss(self, generated: torch.Tensor, target: torch.Tensor, 
                                   target_modality: str) -> torch.Tensor:
        if target_modality in ['image', 'audio']:
            return self.mse_loss(generated, target)
        elif target_modality == 'text':
            return self.ce_loss(generated.view(-1, generated.size(-1)), target.view(-1))
        else:
            return self.mse_loss(generated, target)
    
    def _compute_consistency_loss(self, generated: torch.Tensor, 
                                source_modalities: Dict, target_modality: str) -> torch.Tensor:
        if not source_modalities:
            return torch.tensor(0.0, device=generated.device)
        
        consistency_loss = 0.0
        num_modalities = len(source_modalities)
        
        for modality, features in source_modalities.items():
            if modality != target_modality:
                modality_consistency = self.l1_loss(
                    generated.mean(), features.mean()
                )
                consistency_loss += modality_consistency
        
        return consistency_loss / max(num_modalities, 1)
    
    def _compute_perceptual_loss(self, generated: torch.Tensor, target: torch.Tensor,
                               target_modality: str) -> torch.Tensor:
        if target_modality == 'image':
            return F.l1_loss(
                self._compute_image_features(generated),
                self._compute_image_features(target)
            )
        else:
            return torch.tensor(0.0, device=generated.device)
    
    def _compute_image_features(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 3:
            images = images.unsqueeze(0)
        
        if images.size(1) == 3:
            return images.mean(dim=[2, 3])
        else:
            return images.mean(dim=1)