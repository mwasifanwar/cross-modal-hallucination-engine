import argparse
from examples.basic_hallucination import basic_hallucination_demo
from examples.advanced_generation import advanced_generation_demo

def main():
    parser = argparse.ArgumentParser(description="Cross-Modal Hallucination Engine")
    parser.add_argument('--mode', type=str, choices=['demo', 'train', 'generate'], default='demo')
    parser.add_argument('--source', type=str, help='Source modality file path')
    parser.add_argument('--target', type=str, help='Target modality type')
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        print("Running Cross-Modal Hallucination Demo")
        basic_hallucination_demo()
    
    elif args.mode == 'train':
        print("Running Advanced Training Pipeline")
        advanced_generation_demo()
    
    elif args.mode == 'generate':
        print(f"Generating {args.target} from {args.source}")
        
        engine = CrossModalHallucinationEngine()
        
        if args.target == "image":
            result = engine.text_to_image(args.source)
            result.save(args.output)
        elif args.target == "text":
            result = engine.image_to_text(args.source)
            with open(args.output, 'w') as f:
                f.write(result)
        elif args.target == "audio":
            result = engine.video_to_audio(args.source)
            torchaudio.save(args.output, result, 22050)

if __name__ == "__main__":
    main()