import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import CrossModalHallucinationEngine

def basic_hallucination_demo():
    print("=== Cross-Modal Hallucination Engine Demo ===")
    print("Created by mwasifanwar")
    
    engine = CrossModalHallucinationEngine()
    
    print("1. Text-to-Image Hallucination")
    text_prompt = "A beautiful sunset over mountains with a lake in the foreground"
    generated_image = engine.text_to_image(text_prompt)
    print(f"Generated image from text: '{text_prompt}'")
    
    print("2. Image-to-Text Hallucination")
    image_path = "sample_image.jpg"
    generated_text = engine.image_to_text(image_path)
    print(f"Generated text from image: {generated_text}")
    
    print("3. Audio-to-Image Hallucination")
    audio_path = "sample_audio.wav"
    generated_image_audio = engine.audio_to_image(audio_path)
    print(f"Generated image from audio")
    
    print("4. Multimodal Hallucination")
    multimodal_result = engine.multimodal_hallucination(
        text="A person playing guitar",
        audio="music_sample.wav",
        target_modality="image"
    )
    print("Generated image from text and audio")
    
    return engine

if __name__ == "__main__":
    engine = basic_hallucination_demo()