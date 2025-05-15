import os
import torch

def set_env_variables():
    """Set environment variables needed for the pipeline"""
    # if using Apple MPS, fall back to CPU for unsupported ops
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def get_frame_names(frames_dir):
    """
    Get sorted list of frame filenames
    
    Args:
        frames_dir: Directory containing video frames
        
    Returns:
        Sorted list of frame filenames
    """
    frame_names = [
        p for p in os.listdir(frames_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    return frame_names

def get_unique_frame_numbers(tracking_objects):
    """
    Extract unique frame numbers from tracking objects
    
    Args:
        tracking_objects: List of tracking objects
        
    Returns:
        List of unique frame numbers
    """
    all_frames = []
    for obj_class in tracking_objects:
        for occurrence in obj_class['frameOccurences']:
            if occurrence['frameNum'] not in all_frames:
                all_frames.append(occurrence['frameNum'])
    return sorted(all_frames)

def configure_device(device):
    """
    Configure torch device settings
    
    Args:
        device: PyTorch device object
        
    Returns:
        Device object with configured settings
    """
    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    
    return device 