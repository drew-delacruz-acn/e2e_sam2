#!/usr/bin/env python3

import os
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
import argparse
from transformers import Owlv2Processor, Owlv2ForObjectDetection

class OWLv2Detector:
    def __init__(self, model_name="google/owlv2-base-patch16", device=None):
        """
        Initialize OWLv2 detector
        
        Args:
            model_name: Name of the OWLv2 model
            device: Device to run on ('cuda', 'mps', 'cpu')
        """
        # Set up device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        print(f"Using device: {device}")
        
        # Initialize OWLv2 model
        print(f"Loading OWLv2 model: {model_name}")
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(device)
    
    def detect(self, image, text_queries, threshold=0.1):
        """
        Detect objects in an image
        
        Args:
            image: PIL Image or numpy array
            text_queries: List of text queries for detection
            threshold: Detection confidence threshold
            
        Returns:
            List of detections (boxes, scores, labels)
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
        
        # Process inputs for OWLv2
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
        
        # Extract results
        boxes = results[0]["boxes"]
        scores = results[0]["scores"]
        labels = results[0]["labels"]
        
        detections = []
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
    
    def visualize_detections(self, image, detections, figsize=(12, 10)):
        """
        Visualize detections on an image
        
        Args:
            image: PIL Image or numpy array
            detections: List of detections from detect()
            figsize: Figure size for visualization
        """
        # Convert PIL Image to numpy array if needed
        if not isinstance(image, np.ndarray):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Create figure and axes
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(image_np)
        
        # Define colors for different objects
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        
        # Draw bounding boxes and labels
        for i, detection in enumerate(detections):
            box = detection["box"]
            score = detection["score"]
            label = detection["text"]
            
            # Get coordinates
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Choose color (cycle through colors)
            color = colors[i % len(colors)]
            
            # Draw rectangle
            rect = patches.Rectangle(
                (x1, y1), width, height, 
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            plt.text(
                x1, y1-10, f"{label}: {score:.2f}", 
                color=color, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8)
            )
        
        plt.title(f"OWLv2 Detections: {len(detections)} objects found")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def load_video_frame(video_path=None, frames_dir=None, frame_pattern="*.jpg", frame_idx=0):
    """
    Load a specific frame from a video or directory
    
    Args:
        video_path: Path to video file
        frames_dir: Path to directory with frames
        frame_pattern: Pattern for frame files
        frame_idx: Index of frame to load
        
    Returns:
        Selected frame as numpy array
    """
    if video_path and os.path.exists(video_path):
        # Load frame from video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Set position to the requested frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            raise ValueError(f"Could not read frame {frame_idx} from video")
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()
        
        return frame_rgb
        
    elif frames_dir and os.path.isdir(frames_dir):
        # Load frame from directory
        frame_files = sorted(glob.glob(os.path.join(frames_dir, frame_pattern)))
        
        if not frame_files:
            raise ValueError(f"No frames found in {frames_dir} with pattern {frame_pattern}")
        
        if frame_idx >= len(frame_files):
            raise ValueError(f"Frame index {frame_idx} out of range (only {len(frame_files)} frames available)")
        
        frame = np.array(Image.open(frame_files[frame_idx]).convert("RGB"))
        return frame
    
    else:
        raise ValueError("Either video_path or frames_dir must be provided")


def main():
    parser = argparse.ArgumentParser(description="OWLv2 Video Object Detection")
    parser.add_argument("--model", type=str, default="google/owlv2-base-patch16", help="OWLv2 model name")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--frames-dir", type=str, help="Directory containing video frames")
    parser.add_argument("--frame-pattern", type=str, default="*.jpg", help="Pattern for frame files")
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame index to process")
    parser.add_argument("--text-prompt", type=str, required=True, help="Text prompt(s) for detection, comma-separated")
    parser.add_argument("--threshold", type=float, default=0.1, help="Detection confidence threshold")
    parser.add_argument("--device", type=str, choices=["cuda", "mps", "cpu"], help="Device for inference")
    
    args = parser.parse_args()
    
    # Validate input
    if not args.video and not args.frames_dir:
        parser.error("Either --video or --frames-dir must be provided")
    
    # Parse text prompts
    text_queries = [q.strip() for q in args.text_prompt.split(",")]
    print(f"Using text queries: {text_queries}")
    
    # Load selected frame
    try:
        frame = load_video_frame(
            args.video, 
            args.frames_dir, 
            args.frame_pattern, 
            args.frame_idx
        )
        print(f"Loaded frame {args.frame_idx} with shape {frame.shape}")
    except Exception as e:
        print(f"Error loading frame: {e}")
        return
    
    # Initialize detector
    detector = OWLv2Detector(args.model, args.device)
    
    # Run detection
    detections = detector.detect(frame, text_queries, args.threshold)
    
    # Visualize results
    if detections:
        detector.visualize_detections(frame, detections)
    else:
        print(f"No objects matching {text_queries} found in frame {args.frame_idx}")


if __name__ == "__main__":
    main()