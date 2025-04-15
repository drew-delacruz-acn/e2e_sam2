"""
Model loader for loading and managing different embedding models.
"""

import torch
import torchvision.models as models
from transformers import ViTFeatureExtractor, ViTModel, CLIPModel, CLIPProcessor
from typing import Dict, Any, Literal

ModelType = Literal['resnet50', 'vit', 'clip']

class ModelLoader:
    """Handles loading and management of different embedding models."""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize the model loader.
        
        Args:
            device: Device to load models on ('cuda' or 'cpu')
        """
        self.device = device
        self.models: Dict[str, Any] = {}
        self.processors: Dict[str, Any] = {}
        
    def load_model(self, model_type: ModelType) -> None:
        """Load a specific model.
        
        Args:
            model_type: Type of model to load ('resnet50', 'vit', or 'clip')
        """
        if model_type in self.models:
            return  # Model already loaded
            
        if model_type == 'resnet50':
            # Use the new weights parameter instead of pretrained
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
            # Remove the final classification layer
            self.models[model_type] = torch.nn.Sequential(*list(model.children())[:-1])
            
        elif model_type == 'vit':
            # Use ViTFeatureExtractor instead of ViTImageProcessor for older version
            processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
            model = ViTModel.from_pretrained('google/vit-base-patch16-224')
            self.models[model_type] = model
            self.processors[model_type] = processor
            
        elif model_type == 'clip':
            model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
            processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
            self.models[model_type] = model
            self.processors[model_type] = processor
            
        self.models[model_type].to(self.device)
        self.models[model_type].eval()
        
    def get_model(self, model_type: ModelType) -> Any:
        """Get a loaded model.
        
        Args:
            model_type: Type of model to get
            
        Returns:
            The requested model
            
        Raises:
            KeyError: If model hasn't been loaded
        """
        if model_type not in self.models:
            self.load_model(model_type)
        return self.models[model_type]
    
    def get_processor(self, model_type: ModelType) -> Any:
        """Get a model's processor if it exists.
        
        Args:
            model_type: Type of model to get processor for
            
        Returns:
            The processor for the model if it exists
            
        Raises:
            KeyError: If model type has no processor
        """
        if model_type not in self.processors:
            self.load_model(model_type)
        return self.processors[model_type] 