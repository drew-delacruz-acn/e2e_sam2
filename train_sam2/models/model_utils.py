import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def load_sam2_model(model_cfg, checkpoint_path, device):
    """
    Load SAM2 model from checkpoint
    
    Args:
        model_cfg: Path to model configuration file
        checkpoint_path: Path to model checkpoint
        device: Device to load model on (cuda, mps, or cpu)
    
    Returns:
        Loaded SAM2 model
    """
    sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
    return sam2_model

def get_predictor(model, device):
    """
    Create a SAM2 image predictor from model
    
    Args:
        model: SAM2 model
        device: Device for computation
    
    Returns:
        SAM2ImagePredictor instance
    """
    predictor = SAM2ImagePredictor(model)
    return predictor

def setup_optimizer(model, lr=1e-5, weight_decay=4e-5):
    """
    Setup optimizer for model training
    
    Args:
        model: Model to optimize
        lr: Learning rate
        weight_decay: Weight decay for regularization
    
    Returns:
        Optimizer and scaler for mixed precision training
    """
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Setup mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    return optimizer, scaler

def save_model(model, path, iteration):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        path: Directory to save checkpoint in
        iteration: Current iteration number
    """
    import os
    save_path = os.path.join(path, f"model_iter_{iteration}.pt")
    torch.save(model.state_dict(), save_path)
    
    # Also save latest
    latest_path = os.path.join(path, "model_latest.pt")
    torch.save(model.state_dict(), latest_path)
    
    print(f"Saved model at iteration {iteration} to {save_path}")

def load_model_checkpoint(model, checkpoint_path):
    """
    Load model from checkpoint
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
    
    Returns:
        Model with loaded weights
    """
    model.load_state_dict(torch.load(checkpoint_path))
    return model 