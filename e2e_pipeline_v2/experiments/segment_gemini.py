
# Standard library imports
import os
import json
import argparse
import logging
import traceback
from pathlib import Path
import sys
import time
import gc  # For garbage collection
from PIL import Image

# Add root directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Third-party imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

###########################################
# LOGGING CONFIGURATION
###########################################

def setup_logging():
    """
    Set up logging configuration for tracking script execution.
    
    Returns:
        logger: Configured logging object for use throughout the script
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("segment_embeddings.log")
        ]
    )
    # Set debug level for root logger to get more detailed information
    logging.getLogger().setLevel(logging.DEBUG)
    return logging.getLogger("segment_embeddings")


# Initialize logger
logger = setup_logging()

###########################################
# DEVICE AND MEMORY MANAGEMENT
###########################################

def get_device():
    """
    Determine the appropriate device for computation (CUDA GPU, MPS, or CPU).
    
    Returns:
        device: PyTorch device object for running computations
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA GPU for computation")
        # Log CUDA device details
        cuda_id = torch.cuda.current_device()
        logger.info(f"CUDA Device ID: {cuda_id}")
        logger.info(f"CUDA Device Name: {torch.cuda.get_device_name(cuda_id)}")
        logger.info(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(cuda_id) / 1024**2:.2f} MB")
        logger.info(f"CUDA Memory Reserved: {torch.cuda.memory_reserved(cuda_id) / 1024**2:.2f} MB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon). Note that SAM2 might give different outputs on MPS.")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU. This may be slower than GPU acceleration.")
    
    logger.info(f"Selected device: {device}")
    return device


def log_memory():
    """
    Log current memory usage for debugging memory issues.
    
    This function checks memory usage on the selected device and
    performs garbage collection to free unused memory.
    """
    device = get_device()
    if device.type == 'cuda':
        cuda_id = torch.cuda.current_device()
        logger.debug(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(cuda_id) / 1024**2:.2f} MB")
        logger.debug(f"CUDA Memory Reserved: {torch.cuda.memory_reserved(cuda_id) / 1024**2:.2f} MB")
    
    # Force garbage collection to clean up memory
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        logger.debug(f"After GC - CUDA Memory Allocated: {torch.cuda.memory_allocated(cuda_id) / 1024**2:.2f} MB")

###########################################
# SAM2 MODEL INITIALIZATION
###########################################

def initialize_sam2():
    """
    Initialize the SAM2 (Segment Anything Model 2) model.
    
    This function loads the SAM2 model configuration and weights from the
    specified paths and builds the model on the appropriate device.
    
    Returns:
        sam2: Initialized SAM2 model
        
    Raises:
        FileNotFoundError: If config or checkpoint files are not found
        Exception: If model initialization fails
    """
    logger.info("Building SAM2 model...")
    logger.info(f'Starting path: {os.getcwd()}')
    
    start_time = time.time()
    device = get_device()
    
    try:
        logger.debug("Loading SAM2 model configuration from config file")
        config_file = "configs/sam2.1/sam2.1_hiera_l.yaml"
        checkpoint_path = "checkpoints/sam2.1_hiera_large.pt"
        
        if not os.path.exists(config_file):
            logger.error(f"Config file not found: {config_file}")
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        logger.debug(f"Building SAM2 model with config: {config_file}, checkpoint: {checkpoint_path}, device: {device}")
        sam2 = build_sam2(
            config_file=config_file,
            checkpoint_path=checkpoint_path,
            device=device,
            apply_post_processing=False
        )
        
        logger.debug("SAM2 model built successfully")
        log_memory()
        
        elapsed_time = time.time() - start_time
        logger.info(f"SAM2 model built successfully in {elapsed_time:.2f} seconds")
        
        # Verify the model was loaded correctly
        logger.debug(f"SAM2 model type: {type(sam2)}")
        
        return sam2
    except Exception as e:
        logger.error(f"Failed to initialize SAM2 model: {e}")
        logger.error(traceback.format_exc())
        raise



def create_mask_generator(model, device):
    """
    Create the automatic mask generator with device-specific settings.
    
    This function configures a SAM2AutomaticMaskGenerator with appropriate
    parameters based on the computing device.
    
    Args:
        model: Initialized SAM2 model
        device: PyTorch device (cuda, mps, or cpu)
        
    Returns:
        generator: Configured SAM2AutomaticMaskGenerator
        
    Raises:
        Exception: If mask generator creation fails
    """
    logger.info(f"Creating automatic mask generator for {device}...")
    
    try:
        # Device-specific configurations
        if device.type == 'cuda':
            logger.info("Using CUDA-specific settings for mask generator")
            logger.debug("Parameters: points_per_side=32, pred_iou_thresh=0.86, stability_score_thresh=0.92")
            generator = SAM2AutomaticMaskGenerator(
                model=model,
                # points_per_side=32,
                # pred_iou_thresh=0.86,
                # stability_score_thresh=0.92,
                # crop_n_layers=0,  # Disable cropping to prevent IndexError
                # crop_n_points_downscale_factor=2,
                # min_mask_region_area=100,
                # output_mode="binary_mask"  # Explicitly set output mode
            )
        else:
            # For MPS and CPU
            logger.info(f"Using {device.type}-specific settings for mask generator")
            logger.debug("Parameters: points_per_side=32, pred_iou_thresh=0.86, stability_score_thresh=0.92")
            generator = SAM2AutomaticMaskGenerator(
                model=model,
                # points_per_side=32,
                # pred_iou_thresh=0.86,
                # stability_score_thresh=0.92,
                # crop_n_layers=0,  # Disable cropping to avoid IndexError
                # crop_n_points_downscale_factor=2,
                # min_mask_region_area=100,
                # output_mode="binary_mask"
            )
        
        logger.debug(f"Mask generator created successfully: {type(generator)}")
        return generator
    except Exception as e:
        logger.error(f"Failed to create mask generator: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    device = get_device()
    logger.info(f"Device: {device}")
    sam2 = initialize_sam2()
    mask_generator = create_mask_generator(sam2, device)

    image = Image.open("data/tesseract.png")
    image = np.array(image.convert("RGB"))
    predictor = mask_generator.generate(image)
    print(predictor)
    print("done")