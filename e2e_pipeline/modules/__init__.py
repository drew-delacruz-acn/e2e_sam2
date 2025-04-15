"""
Module initialization for the e2e_pipeline modules package.
"""
from e2e_pipeline.modules.frame_sampling import sample_frames_from_video
from e2e_pipeline.modules.segmentation import segment_image
from e2e_pipeline.modules.embedding import generate_embeddings_for_crops
from e2e_pipeline.modules.serialization import save_embeddings_json

__all__ = [
    'sample_frames_from_video',
    'segment_image',
    'generate_embeddings_for_crops',
    'save_embeddings_json'
] 