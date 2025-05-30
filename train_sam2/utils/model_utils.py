"""
Model utilities for SAM2 training and inference.
"""
import os
import torch
import logging
import traceback
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def setup_device():
    """Set up computation device (CUDA, MPS, or CPU)"""
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            # Use bfloat16 for CUDA
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # Turn on tfloat32 for Ampere GPUs
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS device")
            print(
                "\nSupport for MPS devices is preliminary. SAM2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )
        else:
            device = torch.device("cpu")
            print("Using CPU device")
        return device
    except Exception as e:
        print(f"Error setting up device: {str(e)}")
        raise


def load_sam2_model(model_cfg, checkpoint_path, device, is_state_dict=None):
    """
    Load SAM2 model from checkpoint with proper error handling.
    
    Args:
        model_cfg: Path to model configuration file
        checkpoint_path: Path to model checkpoint
        device: Torch device to load model on
        is_state_dict: Boolean indicating if checkpoint is a state dict. 
                      If None, will try to auto-detect.
    
    Returns:
        SAM2ImagePredictor: Loaded model predictor
    """
    try:
        print(f"Loading model from {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not os.path.exists(model_cfg):
            raise FileNotFoundError(f"Model config not found: {model_cfg}")

        # Auto-detect if it's likely a state dict based on file extension
        if is_state_dict is None:
            is_state_dict = checkpoint_path.endswith('.torch') or checkpoint_path.endswith('.pt')
        
        try:
            if is_state_dict:
                # Load from state dict (fine-tuned model)
                print(f"Loading as state dict: {checkpoint_path}")
                # First load the base model with architecture
                model = build_sam2(model_cfg, None, device=device)
                # Then load the state dict
                checkpoint = torch.load(checkpoint_path, map_location=device)
                if isinstance(checkpoint, dict):
                    if "model" in checkpoint:
                        model.load_state_dict(checkpoint["model"])
                    elif "state_dict" in checkpoint:
                        model.load_state_dict(checkpoint["state_dict"])
                    else:
                        model.load_state_dict(checkpoint)
                else:
                    model.load_state_dict(checkpoint)
            else:
                # Load regular checkpoint
                print(f"Loading as regular checkpoint: {checkpoint_path}")
                model = build_sam2(model_cfg, checkpoint_path, device=device)
        except Exception as first_error:
            print(f"First loading attempt failed: {str(first_error)}")
            print("Trying alternative loading method...")
            
            # Try the alternative approach
            model = build_sam2(model_cfg, None, device=device)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Try different ways the checkpoint might be structured
            if isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    model.load_state_dict(checkpoint["model"])
                elif "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            print("Alternative loading method successful!")

        predictor = SAM2ImagePredictor(model)
        print("Model loaded successfully")
        return predictor
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print(traceback.format_exc())
        raise


def save_checkpoint(model, optimizer, metrics, save_path, iteration=None):
    """
    Save model checkpoint with training metadata.
    
    Args:
        model: The model to save
        optimizer: The optimizer state
        metrics: Dictionary of training metrics
        save_path: Path to save the checkpoint
        iteration: Current training iteration
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
        'metrics': metrics,
        'iteration': iteration
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


def setup_optimizer(model, lr=1e-5, weight_decay=4e-5):
    """Set up optimizer for training"""
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Set up mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()
    
    return optimizer, scaler 