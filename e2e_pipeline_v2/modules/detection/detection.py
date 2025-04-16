"""
Object detection module using OwlViT or OwlViT-2.
"""
from PIL import Image
import torch
import numpy as np
import os
import json
import uuid
import time
from pathlib import Path
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers import OwlViTProcessor, OwlViTForObjectDetection

from e2e_pipeline_v2.modules.detection.utils import calculate_iou, ensure_dir, save_crops, visualize_detections

# Hardcoded model paths
OWLVIT_MODEL_PATH = "google/owlvit-base-patch32"
OWLV2_MODEL_PATH = "google/owlv2-base-patch16"

class ObjectDetector:
    """Object detection module supporting both OwlViT and OwlViT-2"""
    
    def __init__(self, config):
        """
        Initialize the detector with configuration
        
        Args:
            config: Configuration dictionary with:
                - model_type: "owlvit" or "owlvit2"
                - model_name: specific model variant (optional)
                - threshold: detection confidence threshold
        """
        self.config = config
        self.device = self._get_device()
        
        # Set default threshold
        self.threshold = config.get("threshold", 0.1)
        
        # Initialize model based on type
        model_type = config.get("model_type", "owlvit").lower()
        
        print(f"Initializing {model_type} detector on {self.device}")
        
        if model_type == "owlvit":
            # Always use the hardcoded OwlViT model path
            model_name = OWLVIT_MODEL_PATH
            self.processor = OwlViTProcessor.from_pretrained(model_name)
            self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device)
            self.model_type = "owlvit"
            self.model_name = model_name
            print(f"Using OwlViT model: {model_name}")
        elif model_type == "owlvit2":
            # Always use the hardcoded OwlV2 model path
            model_name = OWLV2_MODEL_PATH
            self.processor = Owlv2Processor.from_pretrained(model_name)
            self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(self.device)
            self.model_type = "owlvit2"
            self.model_name = model_name
            print(f"Using OwlV2 model: {model_name}")
        else:
            raise ValueError(f"Unknown model type: {model_type}. Must be 'owlvit' or 'owlvit2'")
    
    def _get_device(self):
        """
        Determine the appropriate device for computation
        
        Returns:
            torch.device: The device to use
        """
        if self.config.get("force_cpu", False):
            return torch.device("cpu")
        elif torch.cuda.is_available() and not self.config.get("force_cpu", False):
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and not self.config.get("force_cpu", False):
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
        # Load image
        image = Image.open(image_path).convert("RGB")
        print(f"Image shape: {image.size[0]}x{image.size[1]}")
        
        # Process inputs
        inputs = self.processor(text=text_queries, images=image, return_tensors="pt").to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process outputs
        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.threshold
        )
        
        # Extract and filter results
        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        
        # Apply non-maximum suppression
        filtered_boxes, filtered_scores, filtered_labels = self._filter_boxes(boxes, scores, labels)
        
        # Print detection results
        for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
            box_list = [round(i, 2) for i in box.tolist()]
            print(f"Detected {text_queries[label]} with confidence {round(score.item(), 3)} at location {box_list}")
        
        return {
            "image": image,
            "image_path": image_path,
            "boxes": filtered_boxes,
            "scores": filtered_scores, 
            "labels": filtered_labels,
            "text_queries": text_queries,
            "model_type": self.model_type,
            "model_name": self.model_name
        }
    
    def _filter_boxes(self, boxes, scores, labels, iou_threshold=0.5):
        """
        Filter out overlapping boxes using non-maximum suppression
        
        Args:
            boxes: Bounding boxes
            scores: Confidence scores
            labels: Label indices
            iou_threshold: IoU threshold for filtering
            
        Returns:
            Filtered boxes, scores, and labels
        """
        if len(boxes) == 0:
            return [], [], []
            
        # Convert to lists for easier manipulation
        boxes_list = [box for box in boxes]
        scores_list = [score.item() for score in scores]
        labels_list = [label.item() for label in labels]
        
        # Sort by score (descending)
        indices = sorted(range(len(scores_list)), key=lambda i: scores_list[i], reverse=True)
        boxes_list = [boxes_list[i] for i in indices]
        scores_list = [scores_list[i] for i in indices]
        labels_list = [labels_list[i] for i in indices]
        
        # Apply NMS
        filtered_indices = []
        for i in range(len(boxes_list)):
            keep = True
            for j in filtered_indices:
                iou = calculate_iou(boxes_list[i], boxes_list[j])
                if iou > iou_threshold:
                    keep = False
                    break
            if keep:
                filtered_indices.append(i)
        
        # Create filtered lists
        filtered_boxes = [boxes_list[i] for i in filtered_indices]
        filtered_scores = [scores_list[i] for i in filtered_indices]
        filtered_labels = [labels_list[i] for i in filtered_indices]
        
        # Convert back to tensors
        filtered_boxes = [torch.tensor(box) if not isinstance(box, torch.Tensor) else box for box in filtered_boxes]
        filtered_scores = [torch.tensor(score) if not isinstance(score, torch.Tensor) else score for score in filtered_scores]
        filtered_labels = [torch.tensor(label) if not isinstance(label, torch.Tensor) else label for label in filtered_labels]
        
        return filtered_boxes, filtered_scores, filtered_labels
    
    def save_results(self, detection_results, output_dir, save_options=None):
        """
        Save detection results using an improved hierarchical structure
        
        Args:
            detection_results: Results from detect()
            output_dir: Base directory to save results
            save_options: Dictionary with saving options
                - save_original: Whether to save the original image
                - save_visualizations: Whether to save visualization
                - save_crops: Whether to save individual crops
                - save_metadata: Whether to save metadata JSON
                - min_score: Minimum score for saving crops
                
        Returns:
            Dictionary with paths to saved results
        """
        # Default save options
        if save_options is None:
            save_options = {
                "save_original": True,
                "save_visualizations": True,
                "save_crops": True,
                "save_metadata": True,
                "min_score": 0.0
            }
        
        # Extract results
        image = detection_results["image"]
        image_path = detection_results["image_path"]
        boxes = detection_results["boxes"]
        scores = detection_results["scores"]
        labels = detection_results["labels"]
        text_queries = detection_results["text_queries"]
        model_type = detection_results["model_type"]
        model_name = detection_results["model_name"].split("/")[-1]
        
        # Get image name without extension
        image_name = os.path.basename(image_path)
        image_name = os.path.splitext(image_name)[0]
        
        # Create output directories
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Create main image directory
        image_dir = os.path.join(output_dir, f"{image_name}-{timestamp}")
        ensure_dir(image_dir)
        
        # Paths for different components
        original_path = os.path.join(image_dir, "original.jpg")
        viz_dir = os.path.join(image_dir, "visualizations")
        detection_dir = os.path.join(image_dir, "detections", model_name)
        metadata_path = os.path.join(image_dir, "metadata.json")
        
        # Create subdirectories
        ensure_dir(viz_dir)
        ensure_dir(detection_dir)
        
        # Dictionary to store results
        results = {
            "base_dir": image_dir,
            "model_type": model_type,
            "model_name": model_name,
            "timestamp": timestamp
        }
        
        # 1. Save original image
        if save_options["save_original"]:
            image.save(original_path)
            results["original_image"] = original_path
        
        # 2. Save visualization
        if save_options["save_visualizations"] and boxes:
            viz_path = os.path.join(viz_dir, f"{model_type}_detections.png")
            visualization_path = visualize_detections(
                image, boxes, scores, labels, text_queries, output_path=viz_path
            )
            results["visualization"] = visualization_path
        
        # 3. Save individual crops
        if save_options["save_crops"] and boxes:
            crop_paths = []
            detection_metadata = []
            
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                # Skip if below minimum score
                if score < save_options["min_score"]:
                    continue
                    
                # Generate unique ID
                detection_id = str(uuid.uuid4())[:8]
                label_text = text_queries[label]
                
                # Create a filename with the label, score, and unique ID
                filename = f"{label_text}_{score:.2f}_{detection_id}.png"
                crop_path = os.path.join(detection_dir, filename)
                
                # Save the crop
                crop_image = image.crop([int(c) for c in box])
                crop_image.save(crop_path)
                crop_paths.append(crop_path)
                
                # Add metadata for this detection
                detection_metadata.append({
                    "id": detection_id,
                    "label": label_text,
                    "score": float(score),
                    "box": [float(c) for c in box],
                    "crop_path": crop_path
                })
            
            results["crops"] = crop_paths
            results["detections"] = detection_metadata
        
        # 4. Save metadata
        if save_options["save_metadata"]:
            metadata = {
                "image_path": image_path,
                "image_size": [image.size[0], image.size[1]],
                "model_type": model_type,
                "model_name": model_name,
                "text_queries": text_queries,
                "timestamp": timestamp,
                "num_detections": len(boxes),
                "detection_threshold": self.threshold,
                "detections": results.get("detections", [])
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            results["metadata"] = metadata_path
        
        return results 