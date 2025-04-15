"""
Generate embeddings for images using different models.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
from PIL import Image
from src.models.model_loader import ModelLoader, ModelType
from src.preprocessing.image_processor import ImageProcessor

class EmbeddingGenerator:
    """Generates and manages embeddings for images using different models."""
    
    def __init__(self, model_loader: ModelLoader, image_processor: ImageProcessor):
        """Initialize the embedding generator.
        
        Args:
            model_loader: ModelLoader instance for accessing models
            image_processor: ImageProcessor instance for preprocessing images
        """
        self.model_loader = model_loader
        self.image_processor = image_processor
        
    def generate_embedding(self, image_path: Path, model_type: ModelType) -> np.ndarray:
        """Generate embedding for a single image using specified model.
        
        Args:
            image_path: Path to the image
            model_type: Type of model to use for embedding generation
            
        Returns:
            numpy array containing the embedding
        """
        model = self.model_loader.get_model(model_type)
        
        with torch.no_grad():
            if model_type in ['clip', 'vit']:
                processor = self.model_loader.get_processor(model_type)
                
                # Load image using PIL instead of passing path string
                image = Image.open(image_path).convert('RGB')
                
                if model_type == 'clip':
                    inputs = processor(images=image, return_tensors="pt")
                else:  # vit
                    inputs = processor(images=image, return_tensors="pt")
                
                inputs = {k: v.to(self.model_loader.device) for k, v in inputs.items()}
                
                if model_type == 'clip':
                    embedding = model.get_image_features(**inputs)
                else:  # vit
                    outputs = model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :]  # Use CLS token
            else:
                image_tensor = self.image_processor.preprocess_image(image_path, model_type)
                image_tensor = image_tensor.to(self.model_loader.device)
                embedding = model(image_tensor.unsqueeze(0))
                
            # Convert to numpy and flatten
            embedding = embedding.cpu().numpy().flatten()
            
            # Normalize the embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
    def generate_embeddings_batch(self, 
                                image_paths: List[Path], 
                                model_types: List[ModelType]
                                ) -> Dict[ModelType, Dict[Path, np.ndarray]]:
        """Generate embeddings for multiple images using multiple models.
        
        Args:
            image_paths: List of paths to images
            model_types: List of model types to use
            
        Returns:
            Dictionary mapping model types to dictionaries mapping image paths to embeddings
        """
        results: Dict[ModelType, Dict[Path, np.ndarray]] = {
            model_type: {} for model_type in model_types
        }
        
        for model_type in model_types:
            for image_path in image_paths:
                embedding = self.generate_embedding(image_path, model_type)
                results[model_type][image_path] = embedding
                
        return results

def main():
    """Main function to generate embeddings for all images."""
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_loader = ModelLoader(device)
    image_processor = ImageProcessor()
    generator = EmbeddingGenerator(model_loader, image_processor)
    
    # Paths
    data_dir = Path('data')
    raw_dir = data_dir / 'raw_images'
    processed_dir = data_dir / 'processed'
    results_dir = Path('results')
    
    # Ensure directories exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image paths
    image_paths = list(raw_dir.glob('*.[jp][pn][g]'))  # jpg, jpeg, png
    
    # Generate embeddings for all models
    model_types: List[ModelType] = ['resnet50', 'vit', 'clip']
    embeddings = generator.generate_embeddings_batch(image_paths, model_types)
    
    # Save embeddings
    for model_type in model_types:
        save_path = results_dir / f'{model_type}_embeddings.npz'
        np.savez(
            save_path,
            **{str(path): emb for path, emb in embeddings[model_type].items()}
        )
        print(f"Saved embeddings for {model_type} to {save_path}")

if __name__ == '__main__':
    main() 