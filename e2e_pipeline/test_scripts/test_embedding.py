"""
Test script for the embedding module.
"""
import sys
import os
from pathlib import Path
import argparse
import json

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from e2e_pipeline.modules.embedding import generate_embeddings_for_crops

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test embedding generation module')
    parser.add_argument('--crops_json', type=str, default=None,
                      help='Path to JSON file with crop information')
    parser.add_argument('--model_types', type=str, nargs='+', default=['clip'],
                      choices=['clip', 'vit', 'resnet50'],
                      help='Embedding model types to use')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use (cuda or cpu). Default: auto-detect')
    parser.add_argument('--output_dir', type=str, default='results/test_embedding',
                      help='Directory to save embedding results')
    args = parser.parse_args()
    
    # Check if crops_json exists or try to find a metadata file from a previous segmentation run
    if args.crops_json and os.path.exists(args.crops_json):
        with open(args.crops_json, 'r') as f:
            crop_infos = json.load(f)
        print(f"Loaded {len(crop_infos)} crops from {args.crops_json}")
    else:
        # Look for metadata files from segmentation tests
        test_segmentation_dir = Path("results/test_segmentation")
        if test_segmentation_dir.exists():
            metadata_files = list(test_segmentation_dir.glob("*_metadata.json"))
            if metadata_files:
                with open(metadata_files[0], 'r') as f:
                    crop_infos = json.load(f)
                print(f"Found and loaded {len(crop_infos)} crops from {metadata_files[0]}")
            else:
                print("No metadata files found from previous segmentation tests.")
                print("Please run test_segmentation.py first or provide a valid --crops_json.")
                return
        else:
            print("No segmentation test directory found.")
            print("Please run test_segmentation.py first or provide a valid --crops_json.")
            return
    
    # Set device if not specified
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Configuration for testing
    test_config = {
        "model_types": args.model_types,
        "device": args.device
    }
    
    # Print configuration
    print(f"Testing embedding generation with configuration:")
    print(f"  Number of crops: {len(crop_infos)}")
    print(f"  Model types: {test_config['model_types']}")
    print(f"  Device: {test_config['device']}")
    print(f"  Output directory: {args.output_dir}")
    
    # Run the embedding generation
    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate embeddings
        crops_with_embeddings = generate_embeddings_for_crops(crop_infos, test_config)
        
        print(f"\nSuccess! Generated embeddings for {len(crops_with_embeddings)} crops.")
        
        # Print information about the first few crops
        print("\nEmbedding information:")
        for i, crop in enumerate(crops_with_embeddings[:3]):  # Show first 3 crops
            print(f"Crop {i+1}:")
            print(f"  Path: {crop['crop_path']}")
            print(f"  Query: {crop['query']}")
            print(f"  Score: {crop['score']:.4f}")
            
            for model_type in args.model_types:
                if model_type in crop['embeddings'] and crop['embeddings'][model_type] is not None:
                    print(f"  {model_type} embedding dimension: {len(crop['embeddings'][model_type])}")
                else:
                    print(f"  {model_type} embedding: None")
            print()
        
        if len(crops_with_embeddings) > 3:
            print(f"... and {len(crops_with_embeddings) - 3} more")
        
        # Save the results to JSON
        output_path = output_dir / "embeddings_results.json"
        with open(output_path, 'w') as f:
            json.dump(crops_with_embeddings, f, indent=2)
        
        print(f"Results saved to: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 