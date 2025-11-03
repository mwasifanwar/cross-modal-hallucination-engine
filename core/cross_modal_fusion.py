import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalFusionNetwork(nn.Module):
    def __init__(self, 
                 text_dim: int = 768,
                 image_dim: int = 2048,
                 audio_dim: int = 128,
                 video_dim: int = 1024,
                 hidden_dim: int = 512,
                 num_heads: int = 8):
        super().__init__()
        
        self.modality_projections = nn.ModuleDict({
            'text': nn.Linear(text_dim, hidden_dim),
            'image': nn.Linear(image_dim, hidden_dim),
            'audio': nn.Linear(audio_dim, hidden_dim),
            'video': nn.Linear(video_dim, hidden_dim)
        })
        
        self.cross_attention = MultiHeadCrossAttention(hidden_dim, num_heads)
        self.fusion_layers = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.hidden_dim = hidden_dim
    
    def forward(self, encoded_modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        projected_modalities = {}
        
        for modality, features in encoded_modalities.items():
            if modality in self.modality_projections:
                projected = self.modality_projections[modality](features)
                projected_modalities[modality] = projected
        
        if not projected_modalities:
            return torch.zeros(1, self.hidden_dim, device=next(self.parameters()).device)
        
        fused_features = self._fuse_modalities(projected_modalities)
        attended_features = self.cross_attention(fused_features, projected_modalities)
        
        final_fusion = self.fusion_layers(attended_features)
        return final_fusion
    
    def _fuse_modalities(self, projected_modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        modality_features = []
        
        for modality in ['text', 'image', 'audio', 'video']:
            if modality in projected_modalities:
                modality_features.append(projected_modalities[modality])
            else:
                modality_features.append(torch.zeros_like(
                    next(iter(projected_modalities.values())), 
                    device=next(self.parameters()).device
                ))
        
        return torch.cat(modality_features, dim=-1)

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim * 4)
        
    def forward(self, fused_features: torch.Tensor, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = fused_features.size(0)
        
        Q = self.query_proj(fused_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        all_keys = []
        all_values = []
        
        for modality in ['text', 'image', 'audio', 'video']:
            if modality in modality_features:
                features = modality_features[modality]
                K = self.key_proj(features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                V = self.value_proj(features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                all_keys.append(K)
                all_values.append(V)
        
        if not all_keys:
            return fused_features
        
        K = torch.cat(all_keys, dim=2)
        V = torch.cat(all_values, dim=2)
        
        attention_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        output = self.out_proj(attended.squeeze(1))
        return output