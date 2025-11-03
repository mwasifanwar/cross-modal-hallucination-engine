import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionGenerator(nn.Module):
    def __init__(self, latent_dim: int = 512, output_dim: int = 784, num_timesteps: int = 1000):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_timesteps = num_timesteps
        
        self.condition_proj = nn.Linear(latent_dim, 256)
        
        self.noise_predictor = nn.Sequential(
            nn.Linear(output_dim + 256 + 1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
        self._initialize_diffusion_params()
    
    def _initialize_diffusion_params(self):
        self.beta = torch.linspace(0.0001, 0.02, self.num_timesteps)
        self.alpha = 1. - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1. - self.alpha_cumprod)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        condition_embed = self.condition_proj(condition)
        
        t_embed = t.float().unsqueeze(-1) / self.num_timesteps
        
        combined = torch.cat([x, condition_embed, t_embed], dim=-1)
        noise_pred = self.noise_predictor(combined)
        
        return noise_pred
    
    def generate(self, condition: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        x = torch.randn(num_samples, self.output_dim, device=condition.device)
        
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.tensor([t] * num_samples, device=condition.device)
            noise_pred = self.forward(x, t_tensor, condition)
            
            alpha_t = self.alpha[t]
            alpha_cumprod_t = self.alpha_cumprod[t]
            beta_t = self.beta[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
            
            x = (1 / torch.sqrt(alpha_t)) * (
                x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * noise_pred
            ) + torch.sqrt(beta_t) * noise
        
        return x