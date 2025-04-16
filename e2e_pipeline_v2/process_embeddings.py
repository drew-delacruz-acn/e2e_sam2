#!/usr/bin/env python3
"""
Script to process detection and segmentation results through the embedding module.
Generates embeddings for crops and masked images.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

from e2e_pipeline_v2.modules.embedding import EmbeddingGenerator, ModelType
from e2e_pipeline_v2.pipeline import DetectionSegmentationPipeline

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process detection and segmentation results through embedding module")
    parser.add_argument("--config", type=str, default="e2e_pipeline_v2/config.yaml",
                      help="Path to the configuration file")
    parser.add_argument("--image", type=str, required=True,
                      help="Path to the input image")
    parser.add_argument("--queries", type=str, nargs="+", default=None,
                      help="Text queries for detection (overrides config)")
    parser.add_argument("--models", type=str, nargs="+", default=["clip", "vit", "resnet50"],
                      choices=["clip", "vit", "resnet50"],
                      help="Embedding models to use")
    parser.add_argument("--output_dir", type=str, default="/Users/andrewdelacruz/e2e_sam2/results",
                      help="Directory to save results")
    parser.add_argument("--device", type=str, default="cpu",
                      help="Device to use (cuda or cpu)")
    parser.add_argument("--force_cpu", action="store_true",
                      help="Force CPU usage for all operations")
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

def format_time(seconds):
    """Format time in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds %= 60
        return f"{int(minutes)} minutes and {seconds:.2f} seconds"
    else:
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        return f"{int(hours)} hours, {int(minutes)} minutes and {seconds:.2f} seconds"

def main():
    start_time = time.time()
    args = parse_args()
    
    # Track input images count
    image_count = 0
    detection_count = 0
    
    # Check if image file exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' does not exist.")
        return 1
    
    # Increment image count
    image_count += 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine the best available device
    if args.force_cpu:
        device = "cpu"
        print("Forcing CPU usage as requested")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.backends.mps.enabled = False
    else:
        # Check for CUDA
        if torch.cuda.is_available():
            device = "cuda"
            print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        # Check for MPS (Apple Metal)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            print("Using Apple Metal Performance Shaders (MPS)")
        else:
            device = "cpu"
            print("No GPU detected. Using CPU")
    
    # Override device if explicitly specified
    if args.device != "cpu":
        device = args.device
    
    print(f"Processing image: {args.image}")
    print(f"Embedding models: {', '.join(args.models)}")
    print(f"Using device: {device}")
    
    try:
        # Initialize pipeline with modified config
        pipeline = DetectionSegmentationPipeline(args.config)
        
        # Modify detection config to force CPU if needed
        if args.force_cpu and hasattr(pipeline.detector, 'config'):
            pipeline.detector.config['force_cpu'] = True
        
        pipeline_start = time.time()
        print("Running detection and segmentation pipeline...")
        
        # Run detection and segmentation
        results = pipeline.run(
            image_path=args.image,
            text_queries=args.queries,
            visualize=True,
            save_results=True,
            generate_embeddings=False  # We'll handle embedding generation ourselves
        )
        
        pipeline_end = time.time()
        pipeline_time = pipeline_end - pipeline_start
        print(f"Pipeline completed in {format_time(pipeline_time)}")
        
        # Check if detection and segmentation were successful
        if not results or "success" in results and not results["success"]:
            print(f"Error: Detection or segmentation failed: {results.get('error', 'Unknown error')}")
            return 1
        
        # Initialize embedding generator
        embedding_generator = EmbeddingGenerator(
            model_types=args.models,
            device=device
        )
        
        # Process each detection
        all_embeddings = []
        
        # Load original image
        original_image = cv2.imread(args.image)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Get metadata about the image
        image_name = os.path.splitext(os.path.basename(args.image))[0]
        
        # Get detection results from the pipeline
        detection_results = results["detection"]
        
        # Check if we have any detections
        if not detection_results or not detection_results.get("boxes", []):
            print("No objects detected in the image.")
            end_time = time.time()
            print(f"\nProcessing completed in {format_time(end_time - start_time)}")
            print(f"Processed {image_count} images with {detection_count} detections")
            return 0
        
        # Track how many detections we have
        if detection_results.get("boxes"):
            detection_count += len(detection_results["boxes"])
        
        embedding_start = time.time()
        print("Generating embeddings for detections...")
        
        # Process detections
        for i, (box, label, score) in enumerate(zip(
            detection_results["boxes"],
            detection_results["labels"],
            detection_results["scores"]
        )):
            # Create a unique ID for this detection
            detection_id = f"{image_name}_{label}_{i:02d}_{score:.2f}"
            
            # Convert box coordinates if needed
            if isinstance(box, torch.Tensor):
                box = box.cpu().numpy().astype(int)
            elif isinstance(box, list):
                box = np.array(box).astype(int)
            
            # Ensure the box is within image boundaries
            h, w = original_image.shape[:2]
            x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
            
            # Get the crop
            crop = original_image[y1:y2, x1:x2]
            
            # Skip if crop is empty
            if crop.size == 0:
                print(f"Warning: Empty crop for detection {detection_id}, skipping")
                continue
            
            # Save the crop
            crop_path = os.path.join(args.output_dir, f"{detection_id}_crop.png")
            save_image(crop, crop_path)
            
            # Generate embeddings for the crop
            embedding_results = {}
            
            print(f"\nGenerating embeddings for detection {detection_id}...")
            print("  Processing crop...")
            
            # Generate all embeddings for all models and save to file
            for model_type in args.models:
                print(f"  Generating embeddings with {model_type}...")
                try:
                    # Generate embeddings
                    crop_emb = embedding_generator.generate_embedding(crop, model_type)
                    
                    # Convert tensors to lists for JSON serialization
                    if isinstance(crop_emb, torch.Tensor):
                        crop_emb = crop_emb.cpu().detach().numpy().tolist()
                    elif hasattr(crop_emb, 'tolist'):
                        crop_emb = crop_emb.tolist()
                    
                    # Add to results
                    embedding_results[model_type] = {
                        "crop": crop_emb
                    }
                    print(f"    Generated {model_type} embedding of length {len(crop_emb)}")
                    print(f"    Sample: {crop_emb[:3]}...")
                except Exception as e:
                    print(f"    Error with {model_type}: {str(e)}")
            
            # Save embeddings
            embeddings_path = os.path.join(args.output_dir, f"{detection_id}_embeddings.json")
            
            # Ensure all values are JSON serializable
            detection_info = {
                "box": box.tolist() if isinstance(box, np.ndarray) else [int(b) if isinstance(b, (np.integer, torch.Tensor)) else b for b in box],
                "label": label if not isinstance(label, (np.ndarray, torch.Tensor)) else label.item() if hasattr(label, 'item') else str(label),
                "score": float(score) if isinstance(score, (np.number, torch.Tensor)) else score
            }
            
            with open(embeddings_path, 'w') as f:
                json.dump({
                    "image_path": args.image,
                    "detection": detection_info,
                    "files": {
                        "crop": crop_path
                    },
                    "embeddings": embedding_results
                }, f, indent=2)
            
            print(f"  Saved embeddings to {embeddings_path}")
            
            # Add to all embeddings
            all_embeddings.append({
                "id": detection_id,
                "embedding_path": embeddings_path
            })
        
        embedding_end = time.time()
        embedding_time = embedding_end - embedding_start
        print(f"Embedding generation completed in {format_time(embedding_time)}")
        
        # Save summary of all embeddings
        summary_path = os.path.join(args.output_dir, f"{image_name}_embeddings_summary.json")
        with open(summary_path, 'w') as f:
            json.dump({
                "image_path": args.image,
                "model_types": args.models,
                "num_embeddings": len(all_embeddings),
                "embeddings": all_embeddings,
                "processing_time": {
                    "pipeline_time": pipeline_time,
                    "embedding_time": embedding_time,
                    "total_time": time.time() - start_time
                }
            }, f, indent=2)
        
        print(f"\nSaved summary to {summary_path}")
        print(f"Total number of processed items: {len(all_embeddings)}")
        
        # Print final timing information
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n== Performance Summary ==")
        print(f"Total execution time: {format_time(total_time)}")
        print(f"Pipeline processing: {format_time(pipeline_time)} ({pipeline_time/total_time*100:.1f}%)")
        print(f"Embedding generation: {format_time(embedding_time)} ({embedding_time/total_time*100:.1f}%)")
        print(f"Processed {image_count} images with {detection_count} detections")
        print(f"Average time per detection: {format_time(embedding_time/max(1, detection_count))}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Print timing even if there was an error
        end_time = time.time()
        print(f"\nExecution failed after {format_time(end_time - start_time)}")
        print(f"Processed {image_count} images with {detection_count} detections")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 