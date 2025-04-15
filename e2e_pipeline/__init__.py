"""
Initialization file for the e2e_pipeline package.
"""
from e2e_pipeline.modules import (
    sample_frames_from_video,
    segment_image,
    generate_embeddings_for_crops,
    save_embeddings_json
)

__all__ = [
    'sample_frames_from_video',
    'segment_image',
    'generate_embeddings_for_crops',
    'save_embeddings_json'
] 