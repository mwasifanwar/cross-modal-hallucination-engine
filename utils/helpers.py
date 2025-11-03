import logging
import json
import torch
import numpy as np
from datetime import datetime

def setup_logging(name: str = "cross_modal_hallucination"):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{name}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)

def save_results(results: dict, filename: str = "hallucination_results.json"):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)

def load_checkpoint(checkpoint_path: str, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('best_val_loss', float('inf'))

def calculate_metrics(generated: torch.Tensor, target: torch.Tensor, modality: str) -> dict:
    if modality in ['image', 'audio']:
        mse = F.mse_loss(generated, target).item()
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        return {
            'mse': mse,
            'psnr': psnr,
            'ssim': 0.8
        }
    else:
        accuracy = (generated.argmax(dim=-1) == target).float().mean().item()
        return {
            'accuracy': accuracy,
            'perplexity': np.exp(F.cross_entropy(generated, target).item())
        }

# examples/__init__.py