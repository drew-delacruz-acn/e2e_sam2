"""
Test script for the serialization module.
"""
import sys
import os
from pathlib import Path
import argparse
import json
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from e2e_pipeline.modules.serialization import save_embeddings_json

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test serialization module')
    parser.add_argument('--embeddings_json', type=str, default=None,
                      help='Path to JSON file with embeddings information')
    parser.add_argument('--output_json', type=str, default='results/test_serialization/serialized_embeddings.json',
                      help='Path to output JSON file')
    parser.add_argument('--generate_mock', action='store_true',
                      help='Generate mock embeddings if no input file is provided')
    args = parser.parse_args()
    
    # Get embeddings data
    if args.embeddings_json and os.path.exists(args.embeddings_json):
        with open(args.embeddings_json, 'r') as f:
            embeddings_list = json.load(f)
        print(f"Loaded {len(embeddings_list)} items with embeddings from {args.embeddings_json}")
    elif args.generate_mock:
        # Generate mock embeddings
        print("Generating mock embeddings data...")
        embeddings_list = []
        for i in range(5):
            embedding_item = {
                "crop_path": f"mock_data/crop_{i}.jpg",
                "original_image": "mock_data/original.jpg",
                "bbox": [100 + i*10, 100 + i*10, 200 + i*10, 200 + i*10],
                "query": f"object_{i}",
                "score": 0.95 - i*0.05,
                "embeddings": {
                    "clip": np.random.rand(512).tolist(),
                    "vit": np.random.rand(768).tolist()
                }
            }
            embeddings_list.append(embedding_item)
        print(f"Generated {len(embeddings_list)} mock embedding items")
    else:
        # Look for embedding files from previous tests
        test_embedding_dir = Path("results/test_embedding")
        if test_embedding_dir.exists():
            embedding_files = list(test_embedding_dir.glob("*.json"))
            if embedding_files:
                with open(embedding_files[0], 'r') as f:
                    embeddings_list = json.load(f)
                print(f"Found and loaded {len(embeddings_list)} items from {embedding_files[0]}")
            else:
                print("No embedding files found from previous tests.")
                print("Please run test_embedding.py first, provide a valid --embeddings_json, or use --generate_mock.")
                return
        else:
            print("No embedding test directory found.")
            print("Please run test_embedding.py first, provide a valid --embeddings_json, or use --generate_mock.")
            return
    
    # Configuration for testing
    serialization_config = {
        "output_json": args.output_json
    }
    
    # Print configuration
    print(f"Testing serialization with configuration:")
    print(f"  Number of items: {len(embeddings_list)}")
    print(f"  Output JSON: {serialization_config['output_json']}")
    
    # Run the serialization
    try:
        # Save embeddings to JSON
        output_path = save_embeddings_json(embeddings_list, serialization_config)
        
        print(f"\nSuccess! Saved embeddings to {output_path}")
        
        # Verify the file was created and can be loaded
        with open(output_path, 'r') as f:
            loaded_data = json.load(f)
        
        print(f"Verification successful!")
        print(f"Loaded {len(loaded_data)} items from the saved JSON.")
        
        # Print details of the first item
        if loaded_data:
            first_item = loaded_data[0]
            print("\nFirst item details:")
            print(f"  Crop path: {first_item['crop_path']}")
            print(f"  Original image: {first_item['original_image']}")
            print(f"  Query: {first_item['query']}")
            print(f"  Score: {first_item['score']}")
            print(f"  Bounding box: {first_item['bbox']}")
            
            print("\nEmbedding information:")
            for model_name, embedding in first_item['embeddings'].items():
                if embedding:
                    print(f"  {model_name} embedding dimension: {len(embedding)}")
                else:
                    print(f"  {model_name} embedding: None")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 