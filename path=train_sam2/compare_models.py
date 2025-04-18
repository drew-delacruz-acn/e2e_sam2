import logging
import os
import torch
import traceback

def load_model(checkpoint_path, model_cfg, device):
    """Load SAM2 model from checkpoint"""
    try:
        logging.info(f"Loading model from {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not os.path.exists(model_cfg):
            raise FileNotFoundError(f"Model config not found: {model_cfg}")

        # Determine if it's likely a state dict based on file extension
        is_state_dict = checkpoint_path.endswith('.torch') or checkpoint_path.endswith('.pt')
        
        try:
            if is_state_dict:
                # Load from state dict (fine-tuned model)
                logging.info(f"Loading as state dict: {checkpoint_path}")
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
                logging.info(f"Loading as regular checkpoint: {checkpoint_path}")
                model = build_sam2(model_cfg, checkpoint_path, device=device)
        except Exception as first_error:
            logging.warning(f"First loading attempt failed: {str(first_error)}")
            logging.info("Trying alternative loading method...")
            
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
            
            logging.info("Alternative loading method successful!")

        predictor = SAM2ImagePredictor(model)
        logging.info("Model loaded successfully")
        return predictor
        
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        logging.error(traceback.format_exc())
        raise 