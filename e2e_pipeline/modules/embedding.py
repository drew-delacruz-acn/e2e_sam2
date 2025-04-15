"""
Module for generating embeddings for cropped images.
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image

# Add the root directory to path to fix import issues
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Custom wrapper for embedding generation
class ModelTypeEnum:
    CLIP = 'clip'
    VIT = 'vit'
    RESNET50 = 'resnet50'

class SimpleImageProcessor:
    """Simple image processor that converts PIL images to tensors for models"""
    
    def preprocess_image(self, image_path, model_type):
        """Process an image for a specific model"""
        from torchvision import transforms
        
        image = Image.open(image_path).convert('RGB')
        
        # Different preprocessing for different models
        if model_type == ModelTypeEnum.CLIP:
            # CLIP expects images normalized with its specific values
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])
            ])
        elif model_type == ModelTypeEnum.VIT:
            # ViT preprocessing
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])
        else:  # Default or ResNet
            # Standard ImageNet normalization
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        
        return preprocess(image)

class SimpleModelLoader:
    """Simple model loader that loads models from transformers and torchvision"""
    
    def __init__(self, device):
        self.device = device
        self.models = {}
        self.processors = {}
    
    def get_model(self, model_type):
        """Get a model of the specified type"""
        if model_type in self.models:
            return self.models[model_type]
        
        if model_type == ModelTypeEnum.CLIP:
            from transformers import CLIPVisionModel
            model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            model.to(self.device)
            self.models[model_type] = model
        elif model_type == ModelTypeEnum.VIT:
            from transformers import ViTModel
            model = ViTModel.from_pretrained("google/vit-base-patch16-224")
            model.to(self.device)
            self.models[model_type] = model
        else:  # Default or ResNet
            from torchvision.models import resnet50, ResNet50_Weights
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            model.fc = torch.nn.Identity()  # Remove classification layer
            model.to(self.device)
            self.models[model_type] = model
        
        return self.models[model_type]
    
    def get_processor(self, model_type):
        """Get a processor for the specified model type"""
        if model_type in self.processors:
            return self.processors[model_type]
        
        if model_type == ModelTypeEnum.CLIP:
            from transformers import CLIPProcessor
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.processors[model_type] = processor
        elif model_type == ModelTypeEnum.VIT:
            from transformers import ViTImageProcessor
            processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
            self.processors[model_type] = processor
        
        return self.processors[model_type]

class SimpleEmbeddingGenerator:
    """Simplified embedding generator that doesn't rely on external imports"""
    
    def __init__(self, model_loader, image_processor):
        self.model_loader = model_loader
        self.image_processor = image_processor
    
    def generate_embedding(self, image_path, model_type):
        """Generate embedding for a single image using the specified model"""
        model = self.model_loader.get_model(model_type)
        
        with torch.no_grad():
            if model_type in [ModelTypeEnum.CLIP, ModelTypeEnum.VIT]:
                processor = self.model_loader.get_processor(model_type)
                
                # Load image using PIL
                image = Image.open(image_path).convert('RGB')
                
                if model_type == ModelTypeEnum.CLIP:
                    inputs = processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(self.model_loader.device) for k, v in inputs.items()}
                    
                    # For CLIP, we need to get the output and then extract the pooler_output
                    outputs = model(**inputs)
                    # Get the image embeddings from the pooler output
                    embedding = outputs.pooler_output
                else:  # vit
                    inputs = processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(self.model_loader.device) for k, v in inputs.items()}
                    
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

def generate_embeddings_for_crops(crop_infos, embedding_config):
    """
    Generate embeddings for cropped images using specified models.
    
    Args:
        crop_infos (list): List of dictionaries containing crop information with keys:
            - crop_path: Path to the cropped image
            - original_image: Path to the original image
            - bbox: Bounding box coordinates [x1, y1, x2, y2]
            - query: The text query that detected this object
            - score: The confidence score of the detection
        embedding_config (dict): Configuration for embedding generation with keys:
            - model_types: List of model types to use (e.g., "clip", "vit")
            - device: Device to use for inference ("cuda" or "cpu")
            
    Returns:
        list: List of dictionaries containing crop information and embeddings with keys:
            - crop_path: Path to the cropped image
            - original_image: Path to the original image
            - bbox: Bounding box coordinates [x1, y1, x2, y2]
            - query: The text query that detected this object
            - score: The confidence score of the detection
            - embeddings: Dictionary mapping model names to embeddings
    """
    # Get configuration
    model_types = embedding_config.get("model_types", ["clip"])
    device = embedding_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize our simplified embedding generator
    model_loader = SimpleModelLoader(device)
    image_processor = SimpleImageProcessor()
    generator = SimpleEmbeddingGenerator(model_loader, image_processor)
    
    # Prepare result list
    result_crops = []
    
    # Process each crop
    for crop_info in crop_infos:
        crop_path = crop_info["crop_path"]
        
        # Generate embeddings for each model type
        embeddings = {}
        for model_type in model_types:
            try:
                # Generate embedding
                embedding = generator.generate_embedding(Path(crop_path), model_type)
                embeddings[model_type] = embedding.tolist()  # Convert to list for JSON serialization
            except Exception as e:
                print(f"Error generating {model_type} embedding for {crop_path}: {e}")
                embeddings[model_type] = None
        
        # Add embeddings to crop info
        crop_info_with_embedding = crop_info.copy()
        crop_info_with_embedding["embeddings"] = embeddings
        
        result_crops.append(crop_info_with_embedding)
    
    print(f"Generated embeddings for {len(result_crops)} crops using models: {', '.join(model_types)}")
    return result_crops

# Test function for this module
def test_embedding_generation():
    """Test function for embedding generation module"""
    from pathlib import Path
    import json
    
    # Create a sample crop_info
    test_crops_dir = Path("results/test_embedding")
    test_crops_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if there are existing crop_infos from a previous segmentation test
    test_segmentation_dir = Path("results/test_segmentation")
    metadata_files = list(test_segmentation_dir.glob("*_metadata.json")) if test_segmentation_dir.exists() else []
    
    if metadata_files:
        # Use existing crop_infos from a previous test
        with open(metadata_files[0], 'r') as f:
            crop_infos = json.load(f)
        print(f"Using {len(crop_infos)} existing crops from {metadata_files[0]}")
    else:
        # Create a dummy crop_info with a sample image
        test_image_path = "/Users/andrewdelacruz/e2e_sam2/data/thor_hammer.jpeg"  # Update with your test image
        if not os.path.exists(test_image_path):
            print(f"Test image {test_image_path} not found. Please provide a valid image path.")
            return False
        
        sample_crop_path = str(test_crops_dir / "sample_crop.jpg")
        img = Image.open(test_image_path)
        img.save(sample_crop_path)
        
        crop_infos = [{
            "crop_path": sample_crop_path,
            "original_image": test_image_path,
            "bbox": [100, 100, 300, 300],
            "query": "sample",
            "score": 0.95
        }]
    
    # Configuration for testing
    test_config = {
        "model_types": ["clip"],  # Start with just one model for testing
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    try:
        # Generate embeddings
        crops_with_embeddings = generate_embeddings_for_crops(crop_infos, test_config)
        
        print(f"Test successful! Generated embeddings for {len(crops_with_embeddings)} crops.")
        
        # Print first crop's embedding shape for each model
        if crops_with_embeddings:
            first_crop = crops_with_embeddings[0]
            print(f"First crop: {first_crop['crop_path']}")
            print(f"Query: {first_crop['query']}")
            for model, emb in first_crop['embeddings'].items():
                if emb:
                    print(f"{model} embedding shape: {len(emb)}")
        
        # Save results to JSON
        out_path = test_crops_dir / "test_embeddings.json"
        with open(out_path, 'w') as f:
            json.dump(crops_with_embeddings, f, indent=2)
        print(f"Results saved to {out_path}")
        
        return True
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_embedding_generation() 