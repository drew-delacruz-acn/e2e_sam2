"""
Test script for the full end-to-end pipeline.
"""
import sys
import os
from pathlib import Path
import argparse
import yaml
import json

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from e2e_pipeline.main import main as run_pipeline

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test full pipeline')
    parser.add_argument('--config', type=str, default=None,
                      help='Path to config YAML file')
    parser.add_argument('--mode', type=str, choices=['video', 'image'], default=None,
                      help='Mode: video or image')
    parser.add_argument('--video_path', type=str, default=None,
                      help='Path to video file (for video mode)')
    parser.add_argument('--image_path', type=str, default=None,
                      help='Path to image file (for image mode)')
    parser.add_argument('--queries', type=str, nargs='+', default=None,
                      help='Text queries for object detection')
    parser.add_argument('--results_dir', type=str, default=None,
                      help='Directory to save results')
    args = parser.parse_args()
    
    config_path = args.config
    
    # If config file not provided, create one from arguments
    if not config_path:
        # Use default config path
        config_path = "e2e_pipeline/test_scripts/test_config.yaml"
        
        # Create a config dictionary
        config = {}
        
        # Set mode
        config["mode"] = args.mode or "image"
        
        # Set paths based on mode
        if config["mode"] == "video":
            config["video"] = {
                "video_path": args.video_path or "data/sample_video.mp4",
                "frame_sampling": {
                    "method": "scene",
                    "params": {
                        "threshold": 30.0
                    }
                }
            }
        else:  # image mode
            config["image"] = {
                "image_path": args.image_path or "/Users/andrewdelacruz/e2e_sam2/data/thor_hammer.jpeg"
            }
        
        # Set segmentation config
        config["segmentation"] = {
            "queries": args.queries or ["hammer"],
            "detection_threshold": 0.1,
            "segmentation_threshold": 0.5,
            "results_dir": args.results_dir or "results/test_pipeline"
        }
        
        # Set embedding config
        config["embedding"] = {
            "model_types": ["clip"],
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        
        # Set serialization config
        config["serialization"] = {
            "output_json": os.path.join(config["segmentation"]["results_dir"], "crop_embeddings.json")
        }
        
        # Save the config
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Created config file at {config_path}")
    
    # Print the configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Running pipeline with configuration:")
    print(f"  Mode: {config['mode']}")
    if config['mode'] == 'video':
        print(f"  Video path: {config['video']['video_path']}")
        print(f"  Frame sampling method: {config['video']['frame_sampling']['method']}")
    else:
        print(f"  Image path: {config['image']['image_path']}")
    print(f"  Queries: {config['segmentation']['queries']}")
    print(f"  Detection threshold: {config['segmentation']['detection_threshold']}")
    print(f"  Results directory: {config['segmentation']['results_dir']}")
    
    # Run the pipeline
    try:
        output_path = run_pipeline(config_path)
        
        print(f"\nPipeline completed successfully!")
        print(f"Results saved to: {output_path}")
        
        # Load and print some of the results
        with open(output_path, 'r') as f:
            results = json.load(f)
        
        print(f"\nProcessed {len(results)} objects:")
        for i, result in enumerate(results[:3]):  # Show first 3 results
            print(f"Object {i+1}:")
            print(f"  Query: {result['query']}")
            print(f"  Score: {result['score']:.4f}")
            print(f"  Crop path: {result['crop_path']}")
            print(f"  Embedding models: {', '.join(result['embeddings'].keys())}")
            print()
        
        if len(results) > 3:
            print(f"... and {len(results) - 3} more")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import torch  # Import here to avoid importing before command-line arguments are parsed
    main() 