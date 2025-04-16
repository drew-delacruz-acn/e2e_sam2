#!/usr/bin/env python3
"""
Test script for the embedding generation module.
This script takes an image and generates embeddings using different models.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import json

# Add the parent directory to sys.path to import the modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from modules.embedding import EmbeddingGenerator, ModelType


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test embedding generation on an image")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--models", type=str, default="clip", 
                        help="Comma-separated list of model types to use (clip, vit, resnet50)")
    parser.add_argument("--output", type=str, default="embedding_results.json", 
                        help="Path to save the output JSON file")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use (cuda or cpu, default: auto-detect)")
    return parser.parse_args()


def main():
    """Main function to test embedding generation."""
    args = parse_args()
    
    # Check if image file exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file '{args.image}' does not exist.")
        return 1
    
    # Parse model types
    model_types = [model.strip() for model in args.models.split(",")]
    valid_models = ['clip', 'vit', 'resnet50']
    for model in model_types:
        if model not in valid_models:
            print(f"Warning: Invalid model type '{model}'. Valid options are: {', '.join(valid_models)}")
            model_types.remove(model)
    
    if not model_types:
        print(f"Error: No valid model types specified. Valid options are: {', '.join(valid_models)}")
        return 1
    
    print(f"Generating embeddings for '{args.image}' using models: {', '.join(model_types)}")
    
    try:
        # Initialize the embedding generator
        embedding_generator = EmbeddingGenerator(model_types=model_types, device=args.device)
        
        # Load the image
        image = Image.open(args.image).convert('RGB')
        
        # Generate embeddings for each model type
        embeddings = {}
        for model_type in model_types:
            print(f"Generating embedding using {model_type}...")
            embedding = embedding_generator.generate_embedding(image, model_type)
            embeddings[model_type] = embedding.tolist()
            print(f"Generated embedding of length {len(embedding)} for {model_type}")
        
        # Create output dictionary
        output_data = {
            "image_path": str(image_path),
            "embeddings": embeddings
        }
        
        # Save results to JSON
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Embeddings saved to '{args.output}'")
        
        # Print dimensions for each embedding
        print("\nEmbedding dimensions:")
        for model, emb in embeddings.items():
            print(f"  {model}: {len(emb)} dimensions")
            
        # Print a small sample of each embedding (first 5 values)
        print("\nEmbedding samples (first 5 values):")
        for model, emb in embeddings.items():
            print(f"  {model}: {emb[:5]}")
        
        # Unload models to free up memory
        embedding_generator.unload_models()
        
        return 0
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 