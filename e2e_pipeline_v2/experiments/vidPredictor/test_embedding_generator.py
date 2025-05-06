# test_embedding_extractor.py
from src.embedding_extractor import EmbeddingExtractor
from PIL import Image
import numpy as np

# Create extractor
extractor = EmbeddingExtractor()

# Test on a sample image
image = Image.open("/home/ubuntu/code/drew/test_data/frames/Scenes 001-020__220D-2-_20230815190723523/subset/21.jpg")
crop = image.crop((6, 69, 773, 474))  # Example crop

# Extract embedding
embedding = extractor.extract(crop)
print(f"Embedding shape: {embedding.shape}")
print(f"First few values: {embedding[:5]}")