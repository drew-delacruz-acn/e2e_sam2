# test_embedding_extractor.py
from src.embedding_extractor import EmbeddingExtractor
from PIL import Image
import numpy as np

# Create extractor
extractor = EmbeddingExtractor()

# Test on a sample image
image = Image.open("test_image.jpg")
crop = image.crop((100, 100, 300, 300))  # Example crop

# Extract embedding
embedding = extractor.extract(crop)
print(f"Embedding shape: {embedding.shape}")
print(f"First few values: {embedding[:5]}")