"""
Embedding generator module for creating embeddings from images using different models.
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Union, Optional
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel, ViTFeatureExtractor, ViTModel

from e2e_pipeline_v2.modules.embedding import ModelType

class EmbeddingGenerator:
    """Generates and manages embeddings for images using different models."""
    
    def __init__(self, 
                model_types: Optional[List[Union[ModelType, str]]] = None,
                device: Optional[str] = None):
        """Initialize the embedding generator.
        
        Args:
            model_types: List of model types to load. Default is ['clip']
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        # Set default model types if not provided
        if model_types is None:
            model_types = [ModelType.CLIP]
        
        # Convert string model types to enum values
        self.model_types = [ModelType(mt) if isinstance(mt, str) else mt for mt in model_types]
        
        # Determine device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing EmbeddingGenerator with device: {self.device}")
        
        # Initialize dictionaries for models and processors
        self.models = {}
        self.processors = {}
        
        # Load requested models
        for model_type in self.model_types:
            self._load_model(model_type)
    
    def _load_model(self, model_type: ModelType):
        """Load a specific model.
        
        Args:
            model_type: Type of model to load
        """
        print(f"Loading {model_type.value} model...")
        
        if model_type == ModelType.CLIP:
            # Load CLIP model
            model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
            processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
            
            model.to(self.device)
            model.eval()
            
            self.models[model_type] = model
            self.processors[model_type] = processor
        
        elif model_type == ModelType.VIT:
            # Load ViT model
            processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
            model = ViTModel.from_pretrained('google/vit-base-patch16-224')
            
            model.to(self.device)
            model.eval()
            
            self.models[model_type] = model
            self.processors[model_type] = processor
        
        elif model_type == ModelType.RESNET50:
            # Load ResNet50 model
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
            # Remove the final classification layer
            model = torch.nn.Sequential(*list(model.children())[:-1])
            
            model.to(self.device)
            model.eval()
            
            # Create preprocessing transforms for ResNet50
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            self.models[model_type] = model
            self.processors[model_type] = preprocess
    
    def generate_embedding(self, 
                          image: Union[str, Image.Image, np.ndarray], 
                          model_type: Union[ModelType, str] = ModelType.CLIP) -> np.ndarray:
        """Generate embedding for a single image.
        
        Args:
            image: Image to process (path string, PIL Image, or numpy array)
            model_type: Type of model to use
            
        Returns:
            Numpy array containing the normalized embedding
        """
        # Convert string model type to enum if needed
        if isinstance(model_type, str):
            model_type = ModelType(model_type)
        
        # Ensure model is loaded
        if model_type not in self.models:
            self._load_model(model_type)
        
        # Get model and processor
        model = self.models[model_type]
        processor = self.processors[model_type]
        
        # Convert image to PIL if necessary
        if isinstance(image, str):
            # It's a file path
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # It's a numpy array
            pil_image = Image.fromarray(np.uint8(image)).convert('RGB')
        elif isinstance(image, Image.Image):
            # It's already a PIL image
            pil_image = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Generate embedding based on model type
        with torch.no_grad():
            if model_type == ModelType.CLIP:
                inputs = processor(images=pil_image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                embedding = model.get_image_features(**inputs)
            
            elif model_type == ModelType.VIT:
                inputs = processor(images=pil_image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :]  # Use CLS token
            
            elif model_type == ModelType.RESNET50:
                # Apply preprocessing
                img_tensor = processor(pil_image).unsqueeze(0).to(self.device)
                embedding = model(img_tensor)
            
            # Normalize and convert to numpy
            embedding = embedding.cpu().numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
    
    def generate_embeddings_batch(self, 
                                 images: List[Union[str, Image.Image, np.ndarray]],
                                 model_types: Optional[List[Union[ModelType, str]]] = None) -> Dict[ModelType, List[np.ndarray]]:
        """Generate embeddings for multiple images using multiple models.
        
        Args:
            images: List of images to process
            model_types: List of model types to use (defaults to self.model_types)
            
        Returns:
            Dictionary mapping model types to lists of embeddings
        """
        # Use default model types if not specified
        if model_types is None:
            model_types = self.model_types
        else:
            # Convert string model types to enum values
            model_types = [ModelType(mt) if isinstance(mt, str) else mt for mt in model_types]
        
        # Initialize results dictionary
        results = {mt: [] for mt in model_types}
        
        # Generate embeddings for each image and model type
        for image in images:
            for model_type in model_types:
                try:
                    embedding = self.generate_embedding(image, model_type)
                    results[model_type].append(embedding)
                except Exception as e:
                    print(f"Error generating embedding with {model_type.value}: {str(e)}")
                    results[model_type].append(None)
        
        return results 