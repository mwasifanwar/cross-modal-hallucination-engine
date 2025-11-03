import torch
import torchvision.transforms as transforms
import torchaudio
from PIL import Image
import numpy as np

class DataPreprocessor:
    def __init__(self):
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.audio_sample_rate = 22050
        self.audio_duration = 5.0
    
    def preprocess_text(self, text: str) -> torch.Tensor:
        return torch.tensor([len(text)], dtype=torch.long)
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        try:
            image = Image.open(image_path).convert('RGB')
            return self.image_transform(image)
        except:
            return torch.zeros(3, 224, 224)
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            if sample_rate != self.audio_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.audio_sample_rate)
                waveform = resampler(waveform)
            
            target_length = int(self.audio_duration * self.audio_sample_rate)
            if waveform.size(1) > target_length:
                waveform = waveform[:, :target_length]
            else:
                padding = target_length - waveform.size(1)
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            return waveform
        except:
            return torch.zeros(1, int(self.audio_duration * self.audio_sample_rate))
    
    def preprocess_video(self, video_path: str) -> torch.Tensor:
        return torch.randn(16, 3, 224, 224)

# training/__init__.py
from .trainers import HallucinationTrainer
from .losses import MultimodalLoss

__all__ = ['HallucinationTrainer', 'MultimodalLoss']