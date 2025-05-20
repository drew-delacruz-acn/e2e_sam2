#!/usr/bin/env python
# example_supcon.py - Example usage of supervised contrastive learning pipeline
import pandas as pd
import numpy as np
import sys
import os
import logging
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contrastiveLearning.pipeline import run_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('example_supcon')

def create_sample_data(num_classes=5, samples_per_class=5, embedding_dim=256):
    """Create a sample DataFrame with embeddings and class IDs for testing."""
    data = []
    for class_id in range(num_classes):
        # Create slightly different embeddings for each class
        base_embedding = np.random.randn(embedding_dim)
        for _ in range(samples_per_class):
            # Add small noise to the base embedding
            embedding = base_embedding + 0.1 * np.random.randn(embedding_dim)
            # Normalize the embedding
            embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
            data.append({
                "class": class_id,
                "embedding": embedding
            })
    return pd.DataFrame(data)

def main():
    """Main function to demonstrate the pipeline."""
    # Create sample data
    logger.info("Creating sample data...")
    df = create_sample_data()
    
    # Run the pipeline with mean prototypes
    logger.info("Running pipeline with mean prototypes...")
    mean_prototypes = run_pipeline(df, method="mean")
    logger.info(f"Generated {len(mean_prototypes)} mean prototypes.")
    
    # Run the pipeline with medoid prototypes
    logger.info("Running pipeline with medoid prototypes...")
    medoid_prototypes = run_pipeline(df, method="medoid")
    logger.info(f"Generated {len(medoid_prototypes)} medoid prototypes.")
    
    # Print example prototype
    logger.info("\nExample prototype (mean):")
    example_class = mean_prototypes["class"].iloc[0]
    example_emb = mean_prototypes["representative_embedding"].iloc[0]
    logger.info(f"Class: {example_class}")
    logger.info(f"Embedding shape: {example_emb.shape}")
    logger.info(f"Embedding norm: {np.linalg.norm(example_emb):.6f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 