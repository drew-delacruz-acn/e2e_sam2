"""
Configuration settings for the pipeline.
"""

# Default configuration
DEFAULT_CONFIG = {
    "detection": {
        "model_name": "google/owlv2-base-patch16-ensemble",
        "threshold": 0.02,
        "iou_threshold": 0.00
    },
    "segmentation": {
        "model_config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "checkpoint": "checkpoints/sam2.1_hiera_large.pt",
        "score_threshold": 0.02
    },
    "paths": {
        "results_dir": "results"
    }
}

def get_config(config_override=None):
    """Get configuration with optional overrides"""
    config = DEFAULT_CONFIG.copy()
    if config_override:
        # Deep update the config
        for k, v in config_override.items():
            if isinstance(v, dict) and k in config:
                config[k].update(v)
            else:
                config[k] = v
    return config 