"""
Configuration package for SAM2 training and inference.
"""

from .default_config import *

__all__ = [
    'DATA_DIR',
    'CHECKPOINT_DIR', 
    'SAM2_CHECKPOINT',
    'MODEL_CONFIG',
    'SAVE_DIR',
    'LEARNING_RATE',
    'WEIGHT_DECAY',
    'MAX_ITERATIONS',
    'SAVE_INTERVAL',
    'MAX_IMAGE_SIZE',
    'NUM_INFERENCE_POINTS',
    'get_device'
] 