import torch
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import TextEncoder, ImageEncoder, AudioEncoder

def test_text_encoder():
    encoder = TextEncoder()
    test_text = "This is a sample text for encoding"
    
    embedding = encoder.encode(test_text)
    
    assert embedding.shape == (1, 768)
    assert torch.is_tensor(embedding)

def test_image_encoder():
    encoder = ImageEncoder()
    test_image = torch.randn(1, 3, 224, 224)
    
    features = encoder.encode(test_image)
    
    assert features.shape == (1, 2048)
    assert torch.is_tensor(features)

def test_audio_encoder():
    encoder = AudioEncoder()
    test_audio = torch.randn(1, 22050)
    
    features = encoder.encode(test_audio)
    
    assert features.shape == (1, 128)
    assert torch.is_tensor(features)