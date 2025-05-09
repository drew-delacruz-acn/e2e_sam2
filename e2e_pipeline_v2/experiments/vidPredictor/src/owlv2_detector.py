# owlv2_detector.py (updated version)
import torch
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from transformers import Owlv2Processor, Owlv2ForObjectDetection

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
            Dictionary with boxes, scores, and labels lists
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
        
        # Prepare results in the format expected by the pipeline
        if len(results[0]["boxes"]) > 0:
            boxes = results[0]["boxes"]
            scores = results[0]["scores"]
            labels = results[0]["labels"]
            
            # For individual logging and debugging
            for box, score, label in zip(boxes, scores, labels):
                box_coords = box.tolist()
                label_idx = label.item()
                score_val = score.item()
                print(f"Detected {text_queries[label_idx]} with confidence {score_val:.3f} at {[round(c, 1) for c in box_coords]}")
            
            # Return in the format expected by the pipeline
            return {
                "boxes": boxes,
                "scores": scores,
                "labels": [text_queries[label.item()] for label in labels]
            }
        else:
            # Return empty results in expected format
            return {
                "boxes": [],
                "scores": [],
                "labels": []
            }
    
    def visualize_detections(self, image, detections, output_dir=None, frame_idx=None, show=True):
        """
        Visualize detections on the image and optionally save to directory
        
        Args:
            image: Image to visualize (PIL.Image or numpy array)
            detections: List of detections from detect() method
            output_dir: Directory to save visualizations (None to skip saving)
            frame_idx: Frame index for naming saved files
            show: Whether to display the visualization with plt.show()
        
        Returns:
            Path to saved visualization (if output_dir is provided)
        """
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
        
        # Save to file if output directory is provided
        save_path = None
        if output_dir is not None:
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Determine filename
            if frame_idx is not None:
                filename = f"frame_{frame_idx:04d}_detections.jpg"
            else:
                # Use timestamp if frame_idx not provided
                import time
                filename = f"detection_{int(time.time())}.jpg"
            
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved detection visualization to {save_path}")
        
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close()
            
        return save_path