#!/usr/bin/env python3
"""
Script to generate embeddings for existing detection/segmentation results.
This avoids running the detection and segmentation again which has MPS compatibility issues.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import glob

from e2e_pipeline_v2.modules.embedding import EmbeddingGenerator, ModelType

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate embeddings for existing detection/segmentation results")
    parser.add_argument("--results_dir", type=str, default="results",
                      help="Directory with detection/segmentation results")
    parser.add_argument("--image", type=str, required=True,
                      help="Path to the original image")
    parser.add_argument("--models", type=str, nargs="+", default=["clip", "vit", "resnet50"],
                      choices=["clip", "vit", "resnet50"],
                      help="Embedding models to use")
    parser.add_argument("--output_dir", type=str, default="/Users/andrewdelacruz/e2e_sam2/results",
                      help="Directory to save embedding results")
    parser.add_argument("--device", type=str, default="cpu",
                      help="Device to use (cuda or cpu)")
    return parser.parse_args()

def apply_mask_to_image(image, mask):
    """Apply mask to image, keeping original pixels inside mask and setting others to zero.
    
    Args:
        image: RGB image as numpy array (H, W, 3)
        mask: Binary mask as numpy array (H, W)
        
    Returns:
        Masked image with original pixels where mask is True, zero elsewhere
    """
    # Ensure mask is boolean
    if mask.dtype != bool:
        mask = mask.astype(bool)
    
    # Create copy of image
    masked_image = np.zeros_like(image)
    
    # Apply mask
    masked_image[mask] = image[mask]
    
    return masked_image

def save_image(image, path):
    """Save image to disk.
    
    Args:
        image: RGB image as numpy array
        path: Path to save the image
    """
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save image
    if isinstance(image, np.ndarray):
        # Convert from RGB to BGR for OpenCV
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    elif isinstance(image, Image.Image):
        image.save(path)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

def main():
    args = parse_args()
    
    # Check if image file exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' does not exist.")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Processing image: {args.image}")
    print(f"Embedding models: {', '.join(args.models)}")
    print(f"Using device: {args.device}")
    
    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator(
        model_types=args.models,
        device=args.device
    )
    
    # Find mask images in the results directory
    mask_pattern = os.path.join(args.results_dir, "*_mask.png")
    mask_files = glob.glob(mask_pattern)
    
    if not mask_files:
        print(f"No mask files found in {args.results_dir} matching pattern {mask_pattern}")
        return 1
    
    print(f"Found {len(mask_files)} mask files")
    
    # Load original image
    original_image = cv2.imread(args.image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Get image name for output files
    image_name = os.path.splitext(os.path.basename(args.image))[0]
    
    # Process each mask file
    all_embeddings = []
    
    for mask_file in mask_files:
        mask_basename = os.path.basename(mask_file)
        mask_id = os.path.splitext(mask_basename)[0]  # Remove .png
        print(f"\nProcessing mask: {mask_id}")
        
        # Load the mask
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        mask_bool = mask > 127  # Convert to boolean mask
        
        # Create a unique ID for this mask
        result_id = f"{image_name}_{mask_id}"
        
        # Create masked image (full image with mask applied)
        masked_full_image = apply_mask_to_image(original_image, mask_bool)
        
        # Get the tight crop around the mask
        # Find bounding box of the mask
        y_indices, x_indices = np.where(mask_bool)
        if len(y_indices) == 0 or len(x_indices) == 0:
            print(f"  Warning: Mask {mask_id} is empty, skipping")
            continue
            
        top, bottom = y_indices.min(), y_indices.max()
        left, right = x_indices.min(), x_indices.max()
        
        # Extract the crop
        crop = original_image[top:bottom+1, left:right+1]
        
        # Create masked crop
        crop_mask = mask_bool[top:bottom+1, left:right+1]
        masked_crop = apply_mask_to_image(crop, crop_mask)
        
        # Save the images
        crop_path = os.path.join(args.output_dir, f"{result_id}_crop.png")
        masked_crop_path = os.path.join(args.output_dir, f"{result_id}_masked_crop.png")
        masked_full_path = os.path.join(args.output_dir, f"{result_id}_masked_full.png")
        
        save_image(crop, crop_path)
        save_image(masked_crop, masked_crop_path)
        save_image(masked_full_image, masked_full_path)
        
        # Generate embeddings for the different image versions
        embedding_results = {}
        
        # Process with example model first for verbose output
        print("  Processing crop...")
        try:
            crop_embedding = embedding_generator.generate_embedding(
                image=crop, 
                model_type=args.models[0]  # Use first model for example output
            )
            print(f"    Generated {args.models[0]} embedding of length {len(crop_embedding)}")
            print(f"    Sample: {crop_embedding[:3]}...")
        except Exception as e:
            print(f"    Error: {str(e)}")
        
        print("  Processing masked crop...")
        try:
            masked_crop_embedding = embedding_generator.generate_embedding(
                image=masked_crop, 
                model_type=args.models[0]  # Use first model for example output
            )
            print(f"    Generated {args.models[0]} embedding of length {len(masked_crop_embedding)}")
            print(f"    Sample: {masked_crop_embedding[:3]}...")
        except Exception as e:
            print(f"    Error: {str(e)}")
        
        print("  Processing masked full image...")
        try:
            masked_full_embedding = embedding_generator.generate_embedding(
                image=masked_full_image, 
                model_type=args.models[0]  # Use first model for example output
            )
            print(f"    Generated {args.models[0]} embedding of length {len(masked_full_embedding)}")
            print(f"    Sample: {masked_full_embedding[:3]}...")
        except Exception as e:
            print(f"    Error: {str(e)}")
        
        # Generate embeddings for all models
        for model_type in args.models:
            print(f"  Generating all embeddings with {model_type}...")
            try:
                # Generate embeddings
                crop_emb = embedding_generator.generate_embedding(crop, model_type).tolist()
                masked_crop_emb = embedding_generator.generate_embedding(masked_crop, model_type).tolist()
                masked_full_emb = embedding_generator.generate_embedding(masked_full_image, model_type).tolist()
                
                # Add to results
                embedding_results[model_type] = {
                    "crop": crop_emb,
                    "masked_crop": masked_crop_emb,
                    "masked_full": masked_full_emb
                }
            except Exception as e:
                print(f"    Error with {model_type}: {str(e)}")
        
        # Save embeddings
        embeddings_path = os.path.join(args.output_dir, f"{result_id}_embeddings.json")
        with open(embeddings_path, 'w') as f:
            json.dump({
                "image_path": args.image,
                "mask_path": mask_file,
                "files": {
                    "crop": crop_path,
                    "masked_crop": masked_crop_path,
                    "masked_full": masked_full_path
                },
                "embeddings": embedding_results
            }, f, indent=2)
        
        print(f"  Saved embeddings to {embeddings_path}")
        
        # Add to all embeddings
        all_embeddings.append({
            "id": result_id,
            "embedding_path": embeddings_path
        })
    
    # Save summary of all embeddings
    summary_path = os.path.join(args.output_dir, f"{image_name}_embeddings_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            "image_path": args.image,
            "model_types": args.models,
            "num_embeddings": len(all_embeddings),
            "embeddings": all_embeddings
        }, f, indent=2)
    
    print(f"\nSaved summary to {summary_path}")
    print(f"Total number of processed masks: {len(all_embeddings)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 