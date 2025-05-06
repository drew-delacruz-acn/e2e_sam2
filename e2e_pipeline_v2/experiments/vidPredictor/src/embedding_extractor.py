# embedding_extractor.py
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class EmbeddingExtractor:
    def __init__(self, device=None):
        """Initialize embedding extractor using ResNet features"""
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
                
        self.device = device
        
        # Load model and remove classification head
        model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(model.children())[:-1])
        self.model.eval()
        self.model = self.model.to(device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract(self, image_crop):
        """Extract embedding from image crop"""
        # Convert to PIL
        if isinstance(image_crop, np.ndarray):
            image_crop = Image.fromarray(image_crop)
            
        # Apply preprocessing
        tensor = self.transform(image_crop).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(tensor)
            
        # Return flattened feature vector
        return features.cpu().numpy().flatten()