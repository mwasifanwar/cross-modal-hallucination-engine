from .multimodal_dataset import MultimodalDataset
from .preprocessing import DataPreprocessor

__all__ = ['MultimodalDataset', 'DataPreprocessor']

# data/multimodal_dataset.py
import torch
from torch.utils.data import Dataset
import os
import json
from typing import Dict, List, Optional

class MultimodalDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", max_samples: Optional[int] = None):
        self.data_dir = data_dir
        self.split = split
        self.max_samples = max_samples
        
        self.samples = self._load_samples()
        
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
    
    def _load_samples(self) -> List[Dict]:
        annotation_file = os.path.join(self.data_dir, f"{self.split}_annotations.json")
        
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                samples = json.load(f)
        else:
            samples = self._generate_dummy_samples()
        
        return samples
    
    def _generate_dummy_samples(self) -> List[Dict]:
        dummy_samples = []
        for i in range(100):
            sample = {
                'text': f"Sample text description {i}",
                'image_path': f"images/sample_{i}.jpg",
                'audio_path': f"audio/sample_{i}.wav", 
                'video_path': f"videos/sample_{i}.mp4",
                'target_modality': 'image' if i % 2 == 0 else 'audio'
            }
            dummy_samples.append(sample)
        
        return dummy_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        item = {
            'text': sample.get('text', ''),
            'image_path': sample.get('image_path', ''),
            'audio_path': sample.get('audio_path', ''),
            'video_path': sample.get('video_path', ''),
            'target_modality': sample.get('target_modality', 'image')
        }
        
        return item