"""
e2e_pipeline_v2 modules package.
This package provides modular components for the end-to-end pipeline.
"""

# Import main functions from each module
from e2e_pipeline_v2.modules.frame_sampling import sample_frames_from_video
from e2e_pipeline_v2.modules.detection import ObjectDetector

# Add more imports as modules are implemented
# from e2e_pipeline_v2.modules.segmentation import segment_image
# from e2e_pipeline_v2.modules.embedding import generate_embeddings_for_crops
# from e2e_pipeline_v2.modules.serialization import save_embeddings_json

__all__ = [
    'sample_frames_from_video',
    'ObjectDetector',
    # Add more exports as modules are implemented
    # 'segment_image',
    # 'generate_embeddings_for_crops',
    # 'save_embeddings_json'
]
