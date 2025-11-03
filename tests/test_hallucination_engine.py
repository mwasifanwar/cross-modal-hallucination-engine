import torch
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import CrossModalHallucinationEngine

def test_engine_initialization():
    engine = CrossModalHallucinationEngine()
    
    assert engine.text_encoder is not None
    assert engine.image_encoder is not None
    assert engine.audio_encoder is not None
    assert engine.video_encoder is not None
    assert engine.fusion_network is not None
    assert engine.image_decoder is not None
    assert engine.audio_decoder is not None
    assert engine.text_decoder is not None

def test_modality_encoding():
    engine = CrossModalHallucinationEngine()
    
    test_text = "This is a test"
    text_tensor = engine._preprocess_text(test_text)
    
    test_modalities = {"text": text_tensor}
    encoded = engine._encode_source_modalities(test_modalities)
    
    assert "text" in encoded
    assert encoded["text"].shape[0] == 1

def test_hallucination_process():
    engine = CrossModalHallucinationEngine()
    
    source_modalities = {
        "text": torch.randn(1, 768)
    }
    
    generated = engine.hallucinate_modality(source_modalities, "image")
    
    assert generated is not None
    assert generated.dim() == 4