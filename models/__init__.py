from .transformers import MultimodalTransformer
from .diffusion_models import DiffusionGenerator
from .attention_networks import CrossModalAttention

__all__ = [
    'MultimodalTransformer',
    'DiffusionGenerator', 
    'CrossModalAttention'
]

# models/transformers.py
import torch
import torch.nn as nn
import math

class MultimodalTransformer(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 6):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.modality_embeddings = nn.Embedding(4, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
    
    def forward(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        modality_tokens = []
        
        modality_ids = {'text': 0, 'image': 1, 'audio': 2, 'video': 3}
        
        for modality, features in modality_features.items():
            if modality in modality_ids:
                modality_id = modality_ids[modality]
                modality_embed = self.modality_embeddings(
                    torch.tensor([modality_id], device=features.device)
                ).expand(features.size(0), -1)
                
                projected_features = features + modality_embed
                modality_tokens.append(projected_features.unsqueeze(1))
        
        if not modality_tokens:
            return torch.zeros(1, self.d_model, device=next(self.parameters()).device)
        
        sequence = torch.cat(modality_tokens, dim=1)
        sequence = self.positional_encoding(sequence.transpose(0, 1))
        
        transformed = self.transformer(sequence)
        pooled = transformed.mean(dim=0)
        
        return pooled

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]