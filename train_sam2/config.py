import os
import torch

# Path configurations
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/davis-2017/DAVIS/")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
SAM2_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "sam2.1_hiera_large.pt")
MODEL_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs/sam2.1/sam2.1_hiera_l.yaml")
SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Training configurations
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 4e-5
MAX_ITERATIONS = 10000
SAVE_INTERVAL = 1000

# Inference configurations
MAX_IMAGE_SIZE = 1024
NUM_INFERENCE_POINTS = 30

# Device configuration
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    if device.type == "cuda":
        # use bfloat16 for CUDA
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

# Environment setup
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # if using Apple MPS, fall back to CPU for unsupported ops 