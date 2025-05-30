"""
Embedding module for generating and managing embeddings.
"""

from enum import Enum, auto
from typing import List, Union, Optional

# Define model types as an enum
class ModelType(str, Enum):
    """Enum for supported embedding model types."""
    CLIP = "clip"
    VIT = "vit"
    RESNET50 = "resnet50"

# Import the main classes
from e2e_pipeline_v2.modules.embedding.generator import EmbeddingGenerator

__all__ = [
    'EmbeddingGenerator',
    'ModelType',
] 