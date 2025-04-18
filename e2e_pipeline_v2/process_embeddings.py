#!/usr/bin/env python3
"""
Script to process detection and segmentation results through the embedding module.
Generates embeddings for crops and masked images.
"""

import os
import sys
import json
import time
import logging
import traceback
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

from e2e_pipeline_v2.modules.embedding import EmbeddingGenerator, ModelType
from e2e_pipeline_v2.pipeline import DetectionSegmentationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('embedding_process')

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
    parser.add_argument("--debug", action="store_true",
                      help="Enable detailed debug logging")
    parser.add_argument("--quiet", action="store_true", 
                      help="Show minimal information (only summary and errors)")
    parser.add_argument("--direct_object", action="store_true",
                      help="Process image as a single object, skipping detection pipeline")
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

def log_tensor_info(tensor, name="tensor"):
    """Log detailed information about a tensor."""
    if logger.level > logging.DEBUG:
        return  # Skip detailed tensor logging if not in debug mode
    
    if not isinstance(tensor, torch.Tensor):
        logger.debug(f"{name} is not a tensor but {type(tensor)}")
        return
    
    try:
        logger.debug(f"{name} - Type: {tensor.dtype}, Shape: {tensor.shape}, Device: {tensor.device}")
        logger.debug(f"{name} - Min: {tensor.min().item()}, Max: {tensor.max().item()}, Mean: {tensor.mean().item()}")
        logger.debug(f"{name} - Contains NaN: {torch.isnan(tensor).any().item()}, Contains Inf: {torch.isinf(tensor).any().item()}")
    except Exception as e:
        logger.warning(f"Could not log complete tensor info for {name}: {str(e)}")
        
def log_system_info():
    """Log detailed system and PyTorch information."""
    logger.info("=== System Information ===")
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("MPS (Metal Performance Shaders) is available")
    
    logger.info(f"NumPy Version: {np.__version__}")
    logger.info(f"OpenCV Version: {cv2.__version__}")
    
    # Only do detailed dtype testing in debug mode
    if logger.level <= logging.DEBUG:
        # Log available dtypes in PyTorch
        logger.debug("=== PyTorch Data Types ===")
        dtypes = [
            torch.float, torch.float16, torch.float32, torch.float64,
            torch.int, torch.int8, torch.int16, torch.int32, torch.int64,
            torch.uint8, torch.bool
        ]
        
        # Check for bfloat16 support
        if hasattr(torch, 'bfloat16'):
            dtypes.append(torch.bfloat16)
            logger.debug("BFloat16 is available in this PyTorch version")
        else:
            logger.debug("BFloat16 is NOT available in this PyTorch version")
        
        for dtype in dtypes:
            try:
                x = torch.tensor([1.0], dtype=dtype)
                logger.debug(f"Dtype {dtype} - test successful on CPU")
                
                if torch.cuda.is_available():
                    try:
                        x_cuda = x.cuda()
                        logger.debug(f"Dtype {dtype} - test successful on CUDA")
                    except Exception as e:
                        logger.warning(f"Dtype {dtype} - CUDA test failed: {str(e)}")
                
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    try:
                        x_mps = x.to('mps')
                        logger.debug(f"Dtype {dtype} - test successful on MPS")
                    except Exception as e:
                        logger.warning(f"Dtype {dtype} - MPS test failed: {str(e)}")
                        
            except Exception as e:
                logger.warning(f"Dtype {dtype} - test failed: {str(e)}")

def main():
    start_time = time.time()
    args = parse_args()
    
    # Track input images count
    image_count = 0
    detection_count = 0
    
    # Set up debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        # Add file handler for debug logs
        debug_file_handler = logging.FileHandler('embedding_debug.log')
        debug_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(debug_file_handler)
        
        # Log system information only in debug mode
        log_system_info()
    elif args.quiet:
        # In quiet mode, only show warnings and higher in console
        logger.setLevel(logging.WARNING)
    else:
        # In normal mode, show INFO and higher in console
        logger.setLevel(logging.INFO)
        
    # Helper function to print important summary info even in quiet mode
    def print_summary(message):
        """Print a message regardless of logging level"""
        if args.quiet:
            # Print directly for important summary info in quiet mode
            print(message)
        else:
            # Use logger for normal mode
            logger.info(message)
            
    # Check if image file exists
    if not os.path.exists(args.image):
        logger.error(f"Error: Image file '{args.image}' does not exist.")
        return 1
    
    # Increment image count
    image_count += 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine the best available device
    if args.force_cpu:
        device = "cpu"
        logger.info("Forcing CPU usage as requested")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.backends.mps.enabled = False
    else:
        # Check for CUDA
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        # Check for MPS (Apple Metal)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Apple Metal Performance Shaders (MPS)")
        else:
            device = "cpu"
            logger.info("No GPU detected. Using CPU")
    
    # Override device if explicitly specified
    if args.device != "cpu":
        device = args.device
        
    logger.info(f"Processing image: {args.image}")
    logger.info(f"Embedding models: {', '.join(args.models)}")
    logger.info(f"Using device: {device}")
    
    try:
        # Initialize embedding generator
        logger.info(f"Initializing embedding generator with models: {args.models}")
        try:
            embedding_generator = EmbeddingGenerator(
                model_types=args.models,
                device=device
            )
        except Exception as e:
            logger.error(f"Error initializing embedding generator: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 1

        if args.direct_object:
            # Direct object mode - skip detection pipeline
            logger.info(f"Processing image as direct object: {args.image}")
            
            # Load image directly
            original_image = cv2.imread(args.image)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            logger.info(f"Original image shape: {original_image.shape}, dtype: {original_image.dtype}")
            
            # Get metadata about the image
            image_name = os.path.splitext(os.path.basename(args.image))[0]
            
            # Generate embeddings directly
            embedding_results = {}
            embedding_start = time.time()
            
            logger.info("Generating embeddings for direct object...")
            
            for model_type in args.models:
                logger.info(f"  Generating embeddings with {model_type}...")
                try:
                    # Generate embeddings
                    emb = embedding_generator.generate_embedding(original_image, model_type)
                    
                    # Convert tensors to lists for JSON serialization
                    if isinstance(emb, torch.Tensor):
                        emb = emb.cpu().numpy().tolist()
                    
                    # Add to results
                    embedding_results[model_type] = {
                        "crop": emb
                    }
                    logger.info(f"    Generated {model_type} embedding of length {len(emb)}")
                    
                except Exception as e:
                    logger.error(f"    Error with {model_type}: {str(e)}")
                    logger.error(f"    Traceback: {traceback.format_exc()}")
            
            # Save embeddings
            embeddings_path = os.path.join(args.output_dir, f"{image_name}_embeddings.json")
            try:
                with open(embeddings_path, 'w') as f:
                    json.dump({
                        "image_path": args.image,
                        "embeddings": embedding_results
                    }, f, indent=2)
                logger.info(f"  Saved embeddings to {embeddings_path}")
            except Exception as e:
                logger.error(f"Error saving embeddings to JSON: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")

        else:
            # Initialize pipeline with modified config
            logger.info("Initializing detection and segmentation pipeline...")
            pipeline = DetectionSegmentationPipeline(args.config)
            
            pipeline_start = time.time()
            logger.info("Running detection and segmentation pipeline...")
            
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
            logger.info(f"Pipeline completed in {format_time(pipeline_time)}")
            
            # Check if detection and segmentation were successful
            if not results or "success" in results and not results["success"]:
                logger.error(f"Error: Detection or segmentation failed: {results.get('error', 'Unknown error')}")
                return 1
            
            # Get detection results from the pipeline
            detection_results = results["detection"]
            
            # Log detection results structure
            logger.debug(f"Detection results keys: {detection_results.keys()}")
            
            # Check if we have any detections
            if not detection_results or not detection_results.get("boxes", []):
                logger.warning("No objects detected in the image.")
                end_time = time.time()
                logger.info(f"\nProcessing completed in {format_time(end_time - start_time)}")
                logger.info(f"Processed {image_count} images with {detection_count} detections")
                return 0
            
            # Track how many detections we have
            num_detections = len(detection_results["boxes"])
            detection_count += num_detections
            logger.info(f"Found {num_detections} detections in the image")
            
            embedding_start = time.time()
            logger.info("Generating embeddings for detections...")
            
            # Process detections
            for i, (box, label, score) in enumerate(zip(
                detection_results["boxes"],
                detection_results["labels"],
                detection_results["scores"]
            )):
                # Create a unique ID for this detection
                detection_id = f"daggers_{label}_{i:02d}_{score:.2f}"
                logger.info(f"\nProcessing detection {i+1}/{num_detections}: {detection_id}")
                
                # Convert box coordinates if needed
                original_box_type = type(box)
                logger.debug(f"Original box type: {original_box_type}")
                
                if isinstance(box, torch.Tensor):
                    logger.debug(f"Box tensor - Type: {box.dtype}, Shape: {box.shape}, Device: {box.device}")
                    box = box.cpu().numpy().astype(int)
                elif isinstance(box, list):
                    logger.debug(f"Box list - Length: {len(box)}, Values: {box}")
                    box = np.array(box).astype(int)
                
                logger.debug(f"Converted box: {box}, Type: {type(box)}")
                
                # Ensure the box is within image boundaries
                h, w = original_image.shape[:2]
                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
                logger.debug(f"Box coordinates after boundary check: [{x1}, {y1}, {x2}, {y2}]")
                
                # Get the crop
                logger.debug(f"Extracting crop from coordinates: y={y1}:{y2}, x={x1}:{x2}")
                crop = original_image[y1:y2, x1:x2]
                logger.debug(f"Crop shape: {crop.shape}, dtype: {crop.dtype}")
                
                # Skip if crop is empty
                if crop.size == 0:
                    logger.warning(f"Warning: Empty crop for detection {detection_id}, skipping")
                    continue
                
                # Save the crop
                crop_path = os.path.join(args.output_dir, f"{detection_id}_crop.png")
                save_image(crop, crop_path)
                logger.debug(f"Saved crop to: {crop_path}")
                
                # Generate embeddings for the crop
                embedding_results = {}
                
                logger.info(f"Generating embeddings for detection {detection_id}...")
                
                # Generate all embeddings for all models and save to file
                for model_type in args.models:
                    logger.info(f"  Generating embeddings with {model_type}...")
                    try:
                        # Generate embeddings
                        emb = embedding_generator.generate_embedding(crop, model_type)
                        
                        # Convert tensors to lists for JSON serialization
                        if isinstance(emb, torch.Tensor):
                            emb = emb.cpu().numpy().tolist()
                        
                        # Add to results
                        embedding_results[model_type] = {
                            "crop": emb
                        }
                        logger.info(f"    Generated {model_type} embedding of length {len(emb)}")
                        
                    except Exception as e:
                        logger.error(f"    Error with {model_type}: {str(e)}")
                        logger.error(f"    Traceback: {traceback.format_exc()}")
                
                # Save embeddings
                embeddings_path = os.path.join(args.output_dir, f"{detection_id}_embeddings.json")
                try:
                    with open(embeddings_path, 'w') as f:
                        json.dump({
                            "image_path": args.image,
                            "detection": {
                                "box": box.tolist() if isinstance(box, np.ndarray) else [int(b) if isinstance(b, (np.integer, torch.Tensor)) else b for b in box],
                                "label": label if not isinstance(label, (np.ndarray, torch.Tensor)) else label.item() if hasattr(label, 'item') else str(label),
                                "score": float(score) if isinstance(score, (np.number, torch.Tensor)) else score
                            },
                            "files": {
                                "crop": crop_path
                            },
                            "embeddings": embedding_results
                        }, f, indent=2)
                    logger.info(f"  Saved embeddings to {embeddings_path}")
                except Exception as e:
                    logger.error(f"Error saving embeddings to JSON: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")

        # Print final timing information
        end_time = time.time()
        total_time = end_time - start_time
        
        print_summary("\n== Performance Summary ==")
        print_summary(f"Total execution time: {format_time(total_time)}")
        if not args.direct_object:
            print_summary(f"Pipeline processing: {format_time(pipeline_time)} ({pipeline_time/total_time*100:.1f}%)")
        print_summary(f"Processed {image_count} images with {detection_count} detections")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 