# owlv2_detector.py
import torch
from PIL import Image
import numpy as np
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import matplotlib.pyplot as plt

class OWLv2Detector:
    def __init__(self, model_name="google/owlv2-base-patch16", device=None):
        """Initialize OWLv2 detector"""
        # Set up device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                # Fall back to CPU for unsupported ops on MPS
                import os
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            else:
                device = "cpu"
        
        self.device = device
        print(f"OWLv2 using device: {device}")
        
        # Load model and processor
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(device)
    
    def detect(self, image, text_queries, threshold=0.1):
        """
        Detect objects in an image based on text queries
        
        Args:
            image: PIL.Image or numpy array
            text_queries: List of text strings to search for
            threshold: Detection confidence threshold
            
        Returns:
            List of detection dictionaries with box, score, label, and text
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
            
        # Process inputs
        inputs = self.processor(text=text_queries, images=image_pil, return_tensors="pt").to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process outputs
        target_sizes = torch.Tensor([image_pil.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold
        )
        
        # Extract and format results
        detections = []
        if len(results[0]["boxes"]) > 0:
            boxes = results[0]["boxes"]
            scores = results[0]["scores"]
            labels = results[0]["labels"]
            
            for box, score, label in zip(boxes, scores, labels):
                box_coords = box.tolist()
                label_idx = label.item()
                score_val = score.item()
                
                detections.append({
                    "box": box_coords,
                    "score": score_val,
                    "label": label_idx,
                    "text": text_queries[label_idx]
                })
                
                print(f"Detected {text_queries[label_idx]} with confidence {score_val:.3f} at {[round(c, 1) for c in box_coords]}")
        
        return detections
    
    def visualize_detections(self, image, detections):
        """Visualize detections on the image"""
        # Convert PIL Image to numpy array if needed
        if not isinstance(image, np.ndarray):
            image_np = np.array(image)
        else:
            image_np = image
            
        # Create plot
        plt.figure(figsize=(10, 10))
        plt.imshow(image_np)
        
        # Draw boxes
        for detection in detections:
            box = detection["box"]
            text = detection["text"]
            score = detection["score"]
            
            # Draw box
            x0, y0, x1, y1 = box
            plt.gca().add_patch(
                plt.Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor='red', linewidth=2)
            )
            
            # Draw label
            plt.text(
                x0, y0-10, f"{text}: {score:.2f}", 
                bbox=dict(facecolor='white', alpha=0.8),
                color='red', fontsize=12
            )
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()