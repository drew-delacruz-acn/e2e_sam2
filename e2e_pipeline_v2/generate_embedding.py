#!/usr/bin/env python3
"""
Simple script to generate embeddings from an image using CLIP.
"""

import os
import sys
import argparse
import json
import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate image embeddings using CLIP")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output", type=str, default="embedding_result.json", 
                      help="Path to save the output JSON file")
    parser.add_argument("--device", type=str, default=None, 
                      help="Device to use (cuda or cpu)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if image file exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file '{args.image}' does not exist.")
        return 1
    
    # Determine device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    try:
        # Load CLIP model and processor
        print("Loading CLIP model...")
        model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        
        model.to(device)
        model.eval()
        
        # Load and preprocess the image
        print(f"Processing image: {args.image}")
        image = Image.open(args.image).convert('RGB')
        
        # Generate embeddings
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            embedding = model.get_image_features(**inputs)
            
            # Normalize and convert to numpy
            embedding = embedding.cpu().numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)
        
        # Create output dictionary
        output_data = {
            "image_path": str(image_path),
            "embedding_length": len(embedding),
            "embedding_sample": embedding[:5].tolist(),
            "embedding": embedding.tolist()
        }
        
        # Save results to JSON
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Embedding saved to '{args.output}'")
        print(f"Embedding dimensions: {len(embedding)}")
        print(f"Sample (first 5 values): {embedding[:5]}")
        
        return 0
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 