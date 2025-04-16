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
    
    # Set up debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        # Add file handler for debug logs
        debug_file_handler = logging.FileHandler('embedding_debug.log')
        debug_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(debug_file_handler)
        
        # Log system information only in debug mode
        log_system_info()
    else:
        # In non-debug mode, only show INFO and higher in console
        logger.setLevel(logging.INFO)
    
    # Track input images count
    image_count = 0
    detection_count = 0
    
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
        # Initialize pipeline with modified config
        logger.info("Initializing detection and segmentation pipeline...")
        pipeline = DetectionSegmentationPipeline(args.config)
        
        # Modify detection config to force CPU if needed
        if args.force_cpu and hasattr(pipeline.detector, 'config'):
            pipeline.detector.config['force_cpu'] = True
        
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
        
        # Initialize embedding generator
        logger.info(f"Initializing embedding generator with models: {args.models}")
        try:
            embedding_generator = EmbeddingGenerator(
                model_types=args.models,
                device=device
            )
            
            # Log model details
            logger.info("=== Embedding Model Details ===")
            for model_name, model in embedding_generator.models.items():
                logger.info(f"Model: {model_name}")
                
                # Log model device
                for name, module in model.named_modules():
                    if hasattr(module, 'weight') and hasattr(module.weight, 'device'):
                        logger.info(f"  Module {name} is on device {module.weight.device}")
                        
                        # Log parameter info for first few layers
                        if name.startswith('0') or name.startswith('1') or name.startswith('encoder'):
                            if hasattr(module, 'weight'):
                                weight = module.weight
                                logger.info(f"  Weight - Shape: {weight.shape}, Type: {weight.dtype}")
                            
                            if hasattr(module, 'bias') and module.bias is not None:
                                bias = module.bias
                                logger.info(f"  Bias - Shape: {bias.shape}, Type: {bias.dtype}")
                
        except Exception as e:
            logger.error(f"Error initializing embedding generator: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 1
        
        # Process each detection
        all_embeddings = []
        
        # Load original image
        logger.info(f"Loading original image: {args.image}")
        original_image = cv2.imread(args.image)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        logger.info(f"Original image shape: {original_image.shape}, dtype: {original_image.dtype}")
        
        # Get metadata about the image
        image_name = os.path.splitext(os.path.basename(args.image))[0]
        
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
            detection_id = f"{image_name}_{label}_{i:02d}_{score:.2f}"
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
                    # Convert crop to tensor for debugging
                    if args.debug:
                        logger.debug(f"Converting crop to tensor for debugging")
                        with torch.no_grad():
                            # Convert numpy array to tensor
                            crop_tensor = torch.from_numpy(crop.transpose(2, 0, 1)).float()
                            logger.debug(f"Crop tensor - Shape: {crop_tensor.shape}, Type: {crop_tensor.dtype}")
                            
                            # Move to device
                            crop_tensor = crop_tensor.to(device)
                            logger.debug(f"Crop tensor moved to device: {crop_tensor.device}")
                    
                    # Generate embeddings
                    logger.debug(f"Calling generate_embedding for {model_type}")
                    crop_emb = embedding_generator.generate_embedding(crop, model_type)
                    
                    # Log embedding tensor info
                    if isinstance(crop_emb, torch.Tensor):
                        log_tensor_info(crop_emb, f"{model_type}_embedding")
                    
                    # Convert tensors to lists for JSON serialization
                    logger.debug(f"Converting embedding to list for JSON serialization")
                    if isinstance(crop_emb, torch.Tensor):
                        logger.debug(f"  Embedding is a tensor on device {crop_emb.device} with dtype {crop_emb.dtype}")
                        try:
                            # First try simple conversion
                            crop_emb = crop_emb.cpu().numpy().tolist()
                        except Exception as e1:
                            logger.warning(f"  Error in simple tensor conversion: {str(e1)}")
                            try:
                                # Try detached conversion
                                crop_emb = crop_emb.detach().cpu().numpy().tolist()
                            except Exception as e2:
                                logger.warning(f"  Error in detached tensor conversion: {str(e2)}")
                                try:
                                    # Try float32 conversion for bfloat16 tensors
                                    logger.debug("  Attempting float32 conversion for potential bfloat16 tensor")
                                    crop_emb = crop_emb.float().cpu().numpy().tolist()
                                except Exception as e3:
                                    logger.error(f"  All tensor conversion methods failed: {str(e3)}")
                                    # Use a Python list conversion as last resort
                                    crop_emb = [float(x) for x in crop_emb.flatten().tolist()]
                    elif hasattr(crop_emb, 'tolist'):
                        crop_emb = crop_emb.tolist()
                    
                    # Add to results
                    embedding_results[model_type] = {
                        "crop": crop_emb
                    }
                    logger.info(f"    Generated {model_type} embedding of length {len(crop_emb)}")
                    logger.debug(f"    Sample: {crop_emb[:3]}...")
                except Exception as e:
                    logger.error(f"    Error with {model_type}: {str(e)}")
                    logger.error(f"    Traceback: {traceback.format_exc()}")
            
            # Save embeddings
            embeddings_path = os.path.join(args.output_dir, f"{detection_id}_embeddings.json")
            
            # Ensure all values are JSON serializable
            logger.debug("Preparing detection info for JSON serialization")
            detection_info = {
                "box": box.tolist() if isinstance(box, np.ndarray) else [int(b) if isinstance(b, (np.integer, torch.Tensor)) else b for b in box],
                "label": label if not isinstance(label, (np.ndarray, torch.Tensor)) else label.item() if hasattr(label, 'item') else str(label),
                "score": float(score) if isinstance(score, (np.number, torch.Tensor)) else score
            }
            
            try:
                with open(embeddings_path, 'w') as f:
                    json.dump({
                        "image_path": args.image,
                        "detection": detection_info,
                        "files": {
                            "crop": crop_path
                        },
                        "embeddings": embedding_results
                    }, f, indent=2)
                logger.info(f"  Saved embeddings to {embeddings_path}")
            except Exception as e:
                logger.error(f"Error saving embeddings to JSON: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Add to all embeddings
            all_embeddings.append({
                "id": detection_id,
                "embedding_path": embeddings_path
            })
        
        embedding_end = time.time()
        embedding_time = embedding_end - embedding_start
        logger.info(f"Embedding generation completed in {format_time(embedding_time)}")
        
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
        
        logger.info(f"\nSaved summary to {summary_path}")
        logger.info(f"Total number of processed items: {len(all_embeddings)}")
        
        # Print final timing information
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"\n== Performance Summary ==")
        logger.info(f"Total execution time: {format_time(total_time)}")
        logger.info(f"Pipeline processing: {format_time(pipeline_time)} ({pipeline_time/total_time*100:.1f}%)")
        logger.info(f"Embedding generation: {format_time(embedding_time)} ({embedding_time/total_time*100:.1f}%)")
        logger.info(f"Processed {image_count} images with {detection_count} detections")
        logger.info(f"Average time per detection: {format_time(embedding_time/max(1, detection_count))}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Print timing even if there was an error
        end_time = time.time()
        logger.error(f"\nExecution failed after {format_time(end_time - start_time)}")
        logger.error(f"Processed {image_count} images with {detection_count} detections")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 