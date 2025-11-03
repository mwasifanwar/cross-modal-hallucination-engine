import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import CrossModalHallucinationEngine
from training import HallucinationTrainer, MultimodalLoss
from data import MultimodalDataset
from torch.utils.data import DataLoader

def advanced_generation_demo():
    print("=== Advanced Cross-Modal Generation Demo ===")
    print("Training and Evaluation Pipeline")
    print("Created by mwasifanwar")
    
    engine = CrossModalHallucinationEngine()
    
    train_dataset = MultimodalDataset("data/train", "train")
    val_dataset = MultimodalDataset("data/val", "val")
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    optimizer = torch.optim.Adam(engine.parameters(), lr=0.001)
    criterion = MultimodalLoss()
    
    trainer = HallucinationTrainer(
        model=engine,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    print("Starting training...")
    trainer.train(num_epochs=10, save_path="best_model.pth")
    
    print("Training completed. Generating samples...")
    
    test_samples = [
        {"text": "A cat sitting on a chair", "target_modality": "image"},
        {"audio": "test_audio.wav", "target_modality": "text"},
        {"video": "test_video.mp4", "target_modality": "audio"}
    ]
    
    for i, sample in enumerate(test_samples):
        result = engine.multimodal_hallucination(**sample)
        print(f"Sample {i+1}: Generated {sample['target_modality']}")
    
    return engine, trainer

if __name__ == "__main__":
    engine, trainer = advanced_generation_demo()