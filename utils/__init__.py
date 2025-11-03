from .config import Config
from .helpers import setup_logging, save_results, load_checkpoint

__all__ = ['Config', 'setup_logging', 'save_results', 'load_checkpoint']

# utils/config.py
class Config:
    # Model parameters
    LATENT_DIM = 512
    HIDDEN_DIM = 256
    NUM_HEADS = 8
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    WEIGHT_DECAY = 1e-4
    
    # Data parameters
    IMAGE_SIZE = (256, 256)
    AUDIO_SAMPLE_RATE = 22050
    AUDIO_DURATION = 5.0
    MAX_TEXT_LENGTH = 512
    
    # Generation parameters
    DIFFUSION_STEPS = 1000
    GENERATION_TEMPERATURE = 0.8
    
    # Loss weights
    RECONSTRUCTION_WEIGHT = 1.0
    CONSISTENCY_WEIGHT = 0.1
    PERCEPTUAL_WEIGHT = 0.01
    
    @classmethod
    def to_dict(cls):
        return {key: value for key, value in cls.__dict__.items() 
                if not key.startswith('_') and not callable(value)}