import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImageDecoder(nn.Module):
    def __init__(self, latent_dim: int = 512, output_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        
        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU()
        )
        
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, latent: torch.Tensor, params: dict = None) -> torch.Tensor:
        size = params.get('size', (256, 256)) if params else (256, 256)
        
        x = self.fc_layers(latent)
        x = x.view(-1, 128, 4, 4)
        
        x = self.deconv_layers(x)
        
        if x.shape[2:] != size:
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        
        return x

class AudioDecoder(nn.Module):
    def __init__(self, latent_dim: int = 512, sample_rate: int = 22050):
        super().__init__()
        self.latent_dim = latent_dim
        self.sample_rate = sample_rate
        
        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU()
        )
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        )
    
    def forward(self, latent: torch.Tensor, params: dict = None) -> torch.Tensor:
        duration = params.get('duration', 5.0) if params else 5.0
        sample_rate = params.get('sample_rate', self.sample_rate) if params else self.sample_rate
        
        num_samples = int(duration * sample_rate)
        
        x = self.fc_layers(latent)
        x = x.unsqueeze(1)
        
        target_length = num_samples // 1024
        if x.size(2) < target_length:
            x = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
        elif x.size(2) > target_length:
            x = x[:, :, :target_length]
        
        x = self.conv_layers(x)
        
        if x.size(2) != num_samples:
            x = F.interpolate(x, size=num_samples, mode='linear', align_corners=False)
        
        return x.squeeze(1)

class TextDecoder(nn.Module):
    def __init__(self, latent_dim: int = 512, vocab_size: int = 30522):
        super().__init__()
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        
        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 768),
            nn.ReLU()
        )
        
        self.transformer_head = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model=768, nhead=8),
            num_layers=3
        )
        
        self.output_proj = nn.Linear(768, vocab_size)
    
    def forward(self, latent: torch.Tensor, params: dict = None) -> torch.Tensor:
        max_length = params.get('max_length', 100) if params else 100
        temperature = params.get('temperature', 1.0) if params else 1.0
        
        batch_size = latent.size(0)
        
        x = self.fc_layers(latent)
        x = x.unsqueeze(0)
        
        memory = x.expand(max_length, batch_size, -1)
        tgt = torch.zeros(max_length, batch_size, 768, device=latent.device)
        
        output = self.transformer_head(tgt, memory)
        logits = self.output_proj(output)
        
        if temperature != 1.0:
            logits = logits / temperature
        
        return logits.permute(1, 0, 2)