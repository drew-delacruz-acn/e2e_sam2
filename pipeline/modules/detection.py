"""
Object detection module using Owlv2.
"""
from PIL import Image
import torch
import numpy as np
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers import OwlViTProcessor, OwlViTForObjectDetection

from .utils import calculate_iou

class ObjectDetector:
    """Object detection module using Owlv2"""
    
    def __init__(self, config):
        """
        Initialize the detector with configuration
        Args:
            config: Configuration dictionary
        """
        self.config = config
        # Use CPU for better compatibility
        self.device = torch.device("cpu")
        print(f"ObjectDetector initialized with device: {self.device}")
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to('cpu')
        # self.processor = Owlv2Processor.from_pretrained(config["detection"]["model_name"])
        # self.model = Owlv2ForObjectDetection.from_pretrained(config["detection"]["model_name"])
    
    def _get_device(self):
        """Determine the appropriate device for computation"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def detect(self, image_path, text_queries):
        """
        Detect objects in an image based on text queries
        Args:
            image_path: Path to the image file
            text_queries: List of text queries for detection
        Returns:
            Dictionary with detection results
        """
        image = Image.open(image_path).convert("RGB")
        print(f"Image shape: {image.size[0]}x{image.size[1]}")
        
        inputs = self.processor(text=text_queries, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.02,

            # threshold=self.config["detection"]["threshold"]

        )
        
        # Extract and filter results
        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        filtered_boxes, filtered_scores, filtered_labels = self._filter_boxes(boxes, scores, labels)

        # Print detection results
        for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
            box_list = [round(i, 2) for i in box.tolist()]
            print(f"Detected {text_queries[label]} with confidence {round(score.item(), 3)} at location {box_list}")
        
        return {
            "image": image,
            "boxes": filtered_boxes,
            "scores": filtered_scores, 
            "labels": filtered_labels,
            "text_queries": text_queries
        }
    
    def _filter_boxes(self, boxes, scores, labels):
        """
        Filter out overlapping boxes using IoU
        Args:
            boxes: Bounding boxes
            scores: Confidence scores
            labels: Label indices
        Returns:
            Filtered boxes, scores, and labels
        """
        filtered_indices = []
        box_areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]

        for i in range(len(boxes)):
            keep = True
            for j in range(len(filtered_indices)):
                idx = filtered_indices[j]
                iou = calculate_iou(boxes[i], boxes[idx])
                print(iou)
                
                # If boxes are touching (iou > 0)
                if iou > 0:
                    # Keep the larger box
                    if box_areas[i] > box_areas[idx]:
                        filtered_indices[j] = i  # Replace smaller box with larger one
                        keep = False
                        break
                    else:
                        keep = False
                        break
            
            if keep:
                filtered_indices.append(i)

        # Create filtered lists
        filtered_boxes = [boxes[i] for i in filtered_indices]
        filtered_scores = [scores[i] for i in filtered_indices]
        filtered_labels = [labels[i] for i in filtered_indices]
        
        return filtered_boxes, filtered_scores, filtered_labels 