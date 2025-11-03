import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoTokenizer, AutoModel
import torchaudio

class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.output_dim = 768
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state.mean(dim=1)
    
    def encode(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            embeddings = self.forward(inputs['input_ids'], inputs['attention_mask'])
        return embeddings

class ImageEncoder(nn.Module):
    def __init__(self, model_name: str = "resnet50"):
        super().__init__()
        if model_name == "resnet50":
            backbone = models.resnet50(pretrained=True)
            self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.output_dim = 2048
        
    def forward(self, x):
        return self.features(x).squeeze(-1).squeeze(-1)
    
    def encode(self, image_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.forward(image_tensor)
        return features

class AudioEncoder(nn.Module):
    def __init__(self, sample_rate: int = 22050, n_mels: int = 128):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.output_dim = 128
        
    def forward(self, x):
        mel_spec = self._waveform_to_melspectrogram(x)
        features = self.conv_layers(mel_spec.unsqueeze(1))
        return features.squeeze(-1).squeeze(-1)
    
    def _waveform_to_melspectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=2048,
            hop_length=512
        )
        return mel_transform(waveform)
    
    def encode(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.forward(audio_tensor)
        return features

class VideoEncoder(nn.Module):
    def __init__(self, frame_encoder: str = "resnet50", num_frames: int = 16):
        super().__init__()
        self.num_frames = num_frames
        self.frame_encoder = ImageEncoder(frame_encoder)
        
        self.temporal_encoder = nn.LSTM(
            input_size=self.frame_encoder.output_dim,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        self.output_dim = 1024
        
    def forward(self, x):
        batch_size, num_frames, C, H, W = x.shape
        x = x.view(batch_size * num_frames, C, H, W)
        
        frame_features = self.frame_encoder(x)
        frame_features = frame_features.view(batch_size, num_frames, -1)
        
        temporal_features, (hidden, cell) = self.temporal_encoder(frame_features)
        video_features = temporal_features[:, -1, :]
        
        return video_features
    
    def encode(self, video_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.forward(video_tensor)
        return features