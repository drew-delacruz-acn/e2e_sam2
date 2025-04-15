"""
Module for serializing embeddings and metadata to JSON.
"""
import os
import json
from pathlib import Path
import numpy as np

def save_embeddings_json(embeddings_list, serialization_config):
    """
    Save the embeddings and metadata to a JSON file.
    
    Args:
        embeddings_list (list): List of dictionaries containing crop information and embeddings with keys:
            - crop_path: Path to the cropped image
            - original_image: Path to the original image
            - bbox: Bounding box coordinates [x1, y1, x2, y2]
            - query: The text query that detected this object
            - score: The confidence score of the detection
            - embeddings: Dictionary mapping model names to embeddings
        serialization_config (dict): Configuration for serialization with keys:
            - output_json: Path to output JSON file
            
    Returns:
        str: Path to the saved JSON file
    """
    # Get output path
    output_json = serialization_config.get("output_json", "results/crop_embeddings.json")
    
    # Ensure directory exists
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process embeddings to make them JSON-compatible
    processed_embeddings = []
    for item in embeddings_list:
        # Make a copy of the item
        processed_item = item.copy()
        
        # Process embeddings if needed
        embeddings = processed_item.get("embeddings", {})
        for model_name, embedding in embeddings.items():
            if embedding is not None and isinstance(embedding, np.ndarray):
                # Convert numpy arrays to lists
                embeddings[model_name] = embedding.tolist()
        
        processed_item["embeddings"] = embeddings
        processed_embeddings.append(processed_item)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(processed_embeddings, f, indent=2)
    
    print(f"Saved {len(processed_embeddings)} crop embeddings to {output_path}")
    return str(output_path)

# Test function for this module
def test_serialization():
    """Test function for serialization module"""
    from pathlib import Path
    import random
    
    # Create a sample embeddings list
    test_embeddings = []
    for i in range(3):
        # Create a sample embedding item
        embedding_item = {
            "crop_path": f"results/crops/sample_crop_{i}.jpg",
            "original_image": "data/sample_image.jpg",
            "bbox": [100 + i*10, 100 + i*10, 200 + i*10, 200 + i*10],
            "query": "sample",
            "score": 0.9 - i*0.1,
            "embeddings": {
                "clip": np.random.rand(512).tolist(),  # Mock CLIP embedding
                "vit": np.random.rand(768).tolist()    # Mock ViT embedding
            }
        }
        test_embeddings.append(embedding_item)
    
    # Configuration for testing
    test_config = {
        "output_json": "results/test_serialization/test_embeddings.json"
    }
    
    try:
        # Save embeddings
        output_path = save_embeddings_json(test_embeddings, test_config)
        
        print(f"Test successful! Saved embeddings to {output_path}")
        
        # Verify the saved file
        with open(output_path, 'r') as f:
            loaded_data = json.load(f)
        
        print(f"Loaded {len(loaded_data)} items from the saved JSON.")
        print(f"First item:")
        print(f"  Crop: {loaded_data[0]['crop_path']}")
        print(f"  Query: {loaded_data[0]['query']}")
        print(f"  Score: {loaded_data[0]['score']}")
        print(f"  Models: {', '.join(loaded_data[0]['embeddings'].keys())}")
        print(f"  CLIP embedding length: {len(loaded_data[0]['embeddings']['clip'])}")
        
        return True
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_serialization() 