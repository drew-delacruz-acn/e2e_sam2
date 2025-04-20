#!/usr/bin/env python3
"""
Script to process ground truth images and generate embeddings.
This script takes a directory of ground truth images and generates embeddings
for each image using the specified models.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from e2e_pipeline_v2.modules.embedding.generator import EmbeddingGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('ground_truth_process')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process ground truth images and generate embeddings")
    
    parser.add_argument("--input_dir", type=str, required=True,
                      help="Directory containing ground truth images")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save generated embeddings")
    parser.add_argument("--config", type=str, default="e2e_pipeline_v2/config/config.yaml",
                      help="Path to the configuration file")
    parser.add_argument("--models", type=str, nargs="+", default=["clip", "vit", "resnet50"],
                      choices=["clip", "vit", "resnet50"],
                      help="Embedding models to use")
    parser.add_argument("--create_mapping", action="store_true",
                      help="Create a ground truth mapping file")
    parser.add_argument("--device", type=str, default="cpu",
                      help="Device to use (cuda or cpu)")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def process_ground_truth(
    input_dir: str,
    output_dir: str,
    embedding_generator: EmbeddingGenerator,
    models: list
) -> dict:
    """Process ground truth images and generate embeddings."""
    processed_objects = {}
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image in input directory
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        image_path = os.path.join(input_dir, filename)
        object_name = os.path.splitext(filename)[0]
        logger.info(f"Processing ground truth image: {filename}")
        
        try:
            # Read and convert image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not read image: {image_path}")
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Generate embeddings for each model
            embeddings = {}
            for model_type in models:
                try:
                    embedding = embedding_generator.generate_embedding(image, model_type)
                    
                    # Convert embedding to list for JSON serialization
                    if isinstance(embedding, torch.Tensor):
                        embedding = embedding.cpu().numpy().tolist()
                    elif isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()
                        
                    embeddings[model_type] = embedding
                    logger.info(f"Generated {model_type} embedding for {object_name}")
                    
                except Exception as e:
                    logger.error(f"Error generating {model_type} embedding for {object_name}: {str(e)}")
                    continue
            
            # Save embeddings
            output_path = os.path.join(output_dir, f"{object_name}_embeddings.json")
            with open(output_path, 'w') as f:
                json.dump({
                    "object_name": object_name,
                    "image_path": image_path,
                    "embeddings": embeddings
                }, f, indent=2)
            
            processed_objects[object_name] = {
                "image_path": image_path,
                "embedding_path": output_path
            }
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            continue
    
    return processed_objects

def create_gt_mapping(processed_objects: dict, output_path: str):
    """Create a mapping file for ground truth objects."""
    # Extract object names and create default mapping
    object_classes = {
        obj_name: {
            "class": obj_name.split('_')[0] if '_' in obj_name else obj_name,
            "similarity_threshold": 0.75  # Default threshold
        }
        for obj_name in processed_objects.keys()
    }
    
    # Save mapping
    with open(output_path, 'w') as f:
        json.dump({
            "object_classes": object_classes,
            "processed_objects": processed_objects
        }, f, indent=2)
    
    logger.info(f"Created ground truth mapping file: {output_path}")

def main():
    args = parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator(
        model_types=args.models,
        device=args.device
    )
    
    try:
        # Process ground truth images
        processed_objects = process_ground_truth(
            args.input_dir,
            args.output_dir,
            embedding_generator,
            args.models
        )
        
        # Create mapping file if requested
        if args.create_mapping:
            mapping_path = os.path.join(args.output_dir, "ground_truth_mapping.json")
            create_gt_mapping(processed_objects, mapping_path)
        
        # Print summary
        logger.info("\n=== Processing Summary ===")
        logger.info(f"Processed {len(processed_objects)} ground truth objects")
        logger.info(f"Generated embeddings saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 