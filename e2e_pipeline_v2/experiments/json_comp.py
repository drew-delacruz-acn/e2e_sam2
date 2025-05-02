import os
import json
from pathlib import Path

def load_embeddings(results_dir, model_type="resnet50"):
    """
    Load embeddings from the results directory.
    
    Args:
        results_dir: Path to the results directory
        model_type: Type of embeddings to load ("resnet50", "vit", or "clip")
        
    Returns:
        Dictionary mapping image names to their segment embeddings
    """
    # Validate model type
    if model_type not in ["resnet50", "vit", "clip"]:
        raise ValueError(f"Invalid model type: {model_type}. Must be 'resnet50', 'vit', or 'clip'")
    
    results_path = Path(results_dir)
    all_embeddings = {}
    
    # First check if there's a processing summary
    summary_path = results_path / "processing_summary.json"
    if summary_path.exists():
        # Use the summary to find embedding files
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        for img_data in summary.get("processed_images", []):
            embeddings_file = img_data.get("embeddings_file")
            if embeddings_file and os.path.exists(embeddings_file):
                with open(embeddings_file, 'r') as f:
                    img_embeddings = json.load(f)
                
                image_name = Path(img_embeddings["original_image"]).stem
                all_embeddings[image_name] = {}
                
                # Extract the requested embeddings for each segment
                for segment_id, segment_data in img_embeddings["segments"].items():
                    if model_type in segment_data["embeddings"]:
                        all_embeddings[image_name][segment_id] = {
                            "embedding": segment_data["embeddings"][model_type],
                            "path": segment_data["path"]
                        }
    else:
        # Scan directory structure to find embedding files
        for img_dir in results_path.iterdir():
            if img_dir.is_dir():
                embeddings_dir = img_dir / "embeddings"
                if embeddings_dir.exists():
                    for embed_file in embeddings_dir.glob("*_embeddings.json"):
                        with open(embed_file, 'r') as f:
                            img_embeddings = json.load(f)
                        
                        image_name = img_dir.name
                        all_embeddings[image_name] = {}
                        
                        # Extract the requested embeddings for each segment
                        for segment_id, segment_data in img_embeddings["segments"].items():
                            if model_type in segment_data["embeddings"]:
                                all_embeddings[image_name][segment_id] = {
                                    "embedding": segment_data["embeddings"][model_type],
                                    "path": segment_data["path"]
                                }
    
    return all_embeddings


# Use the function
embeddings = load_embeddings("path/to/results", "vit")

# Now you have access to all embeddings
for image_name, segments in embeddings.items():
    for segment_id, data in segments.items():
        embedding = data["embedding"]
        segment_path = data["path"]