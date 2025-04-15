"""
Image preprocessing utilities for different models.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from typing import Dict, Any
from src.models.model_loader import ModelType

class ImageProcessor:
    """Handles image preprocessing for different models."""
    
    def __init__(self):
        """Initialize the image processor with transforms for each model."""
        self.transforms: Dict[str, Any] = {
            'resnet50': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]),
            'vit': transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])
            # Note: CLIP uses its own processor
        }
        
    def preprocess_image(self, image_path: Path, model_type: ModelType) -> torch.Tensor:
        """Preprocess an image for a specific model.
        
        Args:
            image_path: Path to the image
            model_type: Type of model the image is being preprocessed for
            
        Returns:
            Preprocessed image as a torch tensor
            
        Raises:
            ValueError: If model_type is 'clip' (should use CLIP's processor instead)
        """
        if model_type == 'clip':
            raise ValueError("CLIP images should be processed using the CLIP processor")
            
        # Load and convert image
        image = Image.open(image_path).convert('RGB')
        
        # Apply appropriate transforms
        return self.transforms[model_type](image)
        
    def save_processed_image(self, image_path: Path, output_dir: Path) -> Path:
        """Save a processed copy of the image.
        
        Args:
            image_path: Path to the original image
            output_dir: Directory to save processed image in
            
        Returns:
            Path to the saved processed image
        """
        # Load and convert image
        image = Image.open(image_path).convert('RGB')
        
        # Basic processing (resize to standard size)
        processed = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ])(image)
        
        # Create output path
        output_path = output_dir / image_path.name
        
        # Save processed image
        processed.save(output_path)
        
        return output_path 