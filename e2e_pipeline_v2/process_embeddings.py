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

# Monkey patch torch to better handle BFloat16 errors
# This needs to be done before importing any model code
def patch_torch_bfloat16():
    """Apply patches to handle BFloat16 issues in PyTorch."""
    
    # Original tensor creation method
    original_tensor = torch.Tensor
    original_from_numpy = torch.from_numpy
    
    # Override tensor creation to prevent BFloat16
    def safe_tensor(*args, **kwargs):
        if 'dtype' in kwargs and kwargs['dtype'] == torch.bfloat16:
            print("Warning: BFloat16 tensor requested. Converting to Float32 for compatibility.")
            kwargs['dtype'] = torch.float32
        return original_tensor(*args, **kwargs)
    
    def safe_from_numpy(ndarray, *args, **kwargs):
        if 'dtype' in kwargs and kwargs['dtype'] == torch.bfloat16:
            print("Warning: BFloat16 tensor requested from NumPy array. Converting to Float32.")
            kwargs['dtype'] = torch.float32
        return original_from_numpy(ndarray, *args, **kwargs)
    
    # Apply the patches
    torch.Tensor = safe_tensor
    torch.from_numpy = safe_from_numpy
    
    # Disable BFloat16 in amp if available
    if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
        original_autocast = torch.cuda.amp.autocast
        
        def safe_autocast(*args, **kwargs):
            if 'dtype' in kwargs and kwargs['dtype'] == torch.bfloat16:
                print("Warning: BFloat16 autocast requested. Using Float32 or Float16 instead.")
                # Use float16 on CUDA, float32 elsewhere
                kwargs['dtype'] = torch.float16 if torch.cuda.is_available() else torch.float32
            return original_autocast(*args, **kwargs)
        
        torch.cuda.amp.autocast = safe_autocast
    
    # Patch tensor conversion methods to prevent BFloat16
    original_to = torch.Tensor.to
    
    def safe_to(self, *args, **kwargs):
        if len(args) > 0 and args[0] == torch.bfloat16:
            print("Warning: Conversion to BFloat16 detected. Using Float32 instead.")
            args = list(args)
            args[0] = torch.float32
            args = tuple(args)
        if 'dtype' in kwargs and kwargs['dtype'] == torch.bfloat16:
            print("Warning: Conversion to BFloat16 detected. Using Float32 instead.")
            kwargs['dtype'] = torch.float32
        return original_to(self, *args, **kwargs)
    
    torch.Tensor.to = safe_to
    
    # If ops module is available, try to patch low-level ops
    if hasattr(torch, 'ops'):
        # Save a reference to the actual _op_table if it exists
        if hasattr(torch.ops, '_op_table'):
            op_table = torch.ops._op_table
            # This is a dirty but effective approach to prevent BFloat16 ops
            for op_name in dir(op_table):
                if not op_name.startswith('__'):
                    op = getattr(op_table, op_name)
                    for func_name in dir(op):
                        if not func_name.startswith('__'):
                            try:
                                func = getattr(op, func_name)
                                # Skip non-callable attributes
                                if not callable(func):
                                    continue
                                # Save original function
                                setattr(op, f"_original_{func_name}", func)
                                # Create wrapped function
                                def make_safe_op(orig_func):
                                    def safe_op(*args, **kwargs):
                                        # Convert BFloat16 tensors in args to Float32
                                        fixed_args = []
                                        for arg in args:
                                            if isinstance(arg, torch.Tensor) and arg.dtype == torch.bfloat16:
                                                fixed_args.append(arg.to(dtype=torch.float32))
                                            else:
                                                fixed_args.append(arg)
                                        # Call original function with fixed args
                                        return orig_func(*fixed_args, **kwargs)
                                    return safe_op
                                # Replace function with safe version
                                setattr(op, func_name, make_safe_op(func))
                            except:
                                # Skip any operations that can't be patched
                                pass

# Apply the patch before importing model code
patch_torch_bfloat16()

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
    parser.add_argument("--force_fp32", action="store_true",
                      help="Force FP32 precision for all operations (helps with compatibility)")
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

def convert_tensor_precision(tensor, device, dtype=None):
    """Convert a tensor to the appropriate precision based on the device.
    
    Args:
        tensor: Input tensor
        device: Device to use (cuda, mps, cpu)
        dtype: Optional dtype to force (overrides automatic selection)
        
    Returns:
        Tensor with appropriate precision
    """
    # If not a tensor, return as is
    if not isinstance(tensor, torch.Tensor):
        return tensor
    
    # If dtype is explicitly specified, use it
    if dtype is not None:
        return tensor.to(device=device, dtype=dtype)
    
    # Select precision based on device
    if device == 'cuda':
        # Use float16 for CUDA (NVIDIA GPUs)
        try:
            return tensor.to(device=device, dtype=torch.float16)
        except Exception as e:
            print(f"Warning: Could not convert to float16: {str(e)}")
            return tensor.to(device=device, dtype=torch.float32)
    elif device == 'mps':
        # Use float32 for MPS (Apple Silicon)
        return tensor.to(device=device, dtype=torch.float32)
    else:
        # Use float32 for CPU
        return tensor.to(device=device, dtype=torch.float32)

def safe_generate_embedding(embedding_generator, image, model_type, device, force_fp32=False):
    """Safely generate embeddings with fallback to more compatible precision.
    
    Args:
        embedding_generator: The embedding generator instance
        image: Input image
        model_type: Model type to use
        device: Device to use
        force_fp32: Whether to force FP32 precision
        
    Returns:
        Generated embedding
    """
    # For CLIP models, we need special handling because they often use BFloat16
    is_clip_model = model_type.lower() == "clip"
    
    for attempt in range(3):  # Try up to 3 times with different approaches
        try:
            if attempt == 0:
                # First attempt: Use the normal approach (or FP32 if forced)
                if force_fp32 or is_clip_model:
                    # Force FP32 mode right away for CLIP or if requested
                    with torch.cuda.amp.autocast(enabled=False):
                        embedding = embedding_generator.generate_embedding(image, model_type)
                        # Convert to FP32 if on GPU
                        if device != 'cpu' and isinstance(embedding, torch.Tensor):
                            embedding = embedding.to(dtype=torch.float32)
                else:
                    # Try with default precision for the device
                    embedding = embedding_generator.generate_embedding(image, model_type)
            
            elif attempt == 1:
                # Second attempt: Always force FP32 precision
                print(f"Retry #{attempt}: Forcing FP32 precision...")
                with torch.cuda.amp.autocast(enabled=False):
                    embedding = embedding_generator.generate_embedding(image, model_type)
                    # Ensure the embedding is in FP32
                    if isinstance(embedding, torch.Tensor):
                        embedding = embedding.to(device='cpu', dtype=torch.float32)
            
            else:
                # Last resort: Use CPU for everything
                print(f"Retry #{attempt}: Forcing CPU + FP32 precision...")
                # Temporarily move to CPU
                original_device = embedding_generator.device
                embedding_generator.device = 'cpu'
                
                with torch.cuda.amp.autocast(enabled=False):
                    embedding = embedding_generator.generate_embedding(image, model_type)
                    # Ensure the embedding is in FP32
                    if isinstance(embedding, torch.Tensor):
                        embedding = embedding.to(dtype=torch.float32)
                
                # Restore device
                embedding_generator.device = original_device
            
            return embedding
            
        except RuntimeError as e:
            err_msg = str(e)
            # Check for precision-related errors
            if "unsupported scalar type" in err_msg or "BFloat16" in err_msg:
                print(f"Precision error detected: {err_msg}")
                if attempt < 2:
                    print(f"Trying fallback approach #{attempt+1}...")
                else:
                    raise RuntimeError(f"Failed to generate embedding after all fallback attempts: {err_msg}")
            else:
                # Re-raise other types of errors
                raise
        except Exception as e:
            # For non-RuntimeErrors, just re-raise
            raise

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
        if hasattr(torch.backends, 'mps'):
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
    
    # Preemptively disable bfloat16 for CLIP models (they often use it internally)
    print("Configuring PyTorch for maximum compatibility...")
    # Force torch to use float32 for matmul
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('highest')
    
    # Set appropriate precision for the device
    if device == 'cuda':
        if args.force_fp32:
            precision = "FP32 (forced)"
            # Disable automatic mixed precision
            if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = False
            if hasattr(torch.cuda, 'amp'):
                torch.cuda.amp.autocast(enabled=False)
        else:
            precision = "FP16"
            # Allow TF32 on Ampere+ GPUs for better performance
            if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
    elif device == 'mps':
        precision = "FP32"
    else:
        precision = "FP32"
        
    print(f"Processing image: {args.image}")
    print(f"Embedding models: {', '.join(args.models)}")
    print(f"Using device: {device} with {precision} precision")
    
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
                    # Generate embeddings with safe fallback
                    crop_emb = safe_generate_embedding(
                        embedding_generator, 
                        crop, 
                        model_type, 
                        device, 
                        force_fp32=args.force_fp32
                    )
                    
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
                    print(f"    Skipping this model for detection {detection_id}")
            
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
                },
                "device_info": {
                    "device": device,
                    "precision": precision
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
        print(f"Device: {device} with {precision} precision")
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