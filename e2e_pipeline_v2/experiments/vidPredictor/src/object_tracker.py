# object_tracker.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import matplotlib.pyplot as plt

class ObjectTracker:
    def __init__(self, iou_weight=0.5, emb_weight=0.5, match_threshold=0.4, 
                 min_iou_threshold=0.1, min_emb_threshold=0.5,
                 new_instance_max_iou=0.1):
        """Initialize simple two-tier object tracker
        
        Args:
            iou_weight: Weight for IoU in combined score calculation
            emb_weight: Weight for embedding similarity in combined score calculation
            match_threshold: Minimum combined score to consider a match
            min_iou_threshold: Minimum IoU required for a potential match
            min_emb_threshold: Minimum embedding similarity required for a potential match
            new_instance_max_iou: Maximum IoU to consider a new instance of the same class
        """
        self.tracked_objects = {}
        self.next_obj_id = 1
        self.iou_weight = iou_weight
        self.emb_weight = emb_weight
        self.match_threshold = match_threshold
        self.min_iou_threshold = min_iou_threshold
        self.min_emb_threshold = min_emb_threshold
        self.new_instance_max_iou = new_instance_max_iou
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Check if boxes overlap
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        # Calculate areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        union = area1 + area2 - intersection
        
        return intersection / union
    
    def update_tracks(self, frame, frame_idx, detections, embedding_extractor, output_dir=None):
        """Update object tracks with new detections
        
        Args:
            frame: Current video frame
            frame_idx: Current frame index
            detections: List of detections from OWLv2
            embedding_extractor: Feature extractor for appearance matching
            output_dir: Optional output directory for visualizations
            
        Returns:
            Dictionary mapping object IDs to their current boxes
        """
        current_objects = {}
        
        # First frame initialization
        if not self.tracked_objects:

            for detection in detections:
                obj_id = self.next_obj_id
                self.next_obj_id += 1
                box = detection["box"]
                
                # Extract crop and embedding
                x1, y1, x2, y2 = [int(c) for c in box]
                if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
                    # Skip invalid boxes
                    continue
                    
                crop = frame[y1:y2, x1:x2]
                embedding = embedding_extractor.extract(crop)
                
                # Initialize object
                self.tracked_objects[obj_id] = {
                    "class": detection["text"],
                    "embedding": embedding,
                    "trajectory": [(frame_idx, box)],
                    "last_seen": frame_idx
                }
                
                current_objects[obj_id] = box
                print(f"Initialized object {obj_id} ({detection['text']})")
                
            return current_objects
        
        # For existing tracks, compute matching scores
        match_scores = []
        
        for i, detection in enumerate(detections):
            box = detection["box"]
            label = detection["text"]
            conf = detection["score"]
            
            # Extract crop and embedding
            x1, y1, x2, y2 = [int(c) for c in box]
            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
                # Skip invalid boxes
                continue
                
            crop = frame[y1:y2, x1:x2]
            det_embedding = embedding_extractor.extract(crop)
            
            # Group existing objects by class for better instance handling
            class_objects = {}
            for obj_id, obj_data in self.tracked_objects.items():
                if obj_data["class"] == label:
                    if obj_data["class"] not in class_objects:
                        class_objects[obj_data["class"]] = []
                    class_objects[obj_data["class"]].append((obj_id, obj_data))
            
            # Skip if no objects of this class are being tracked
            if label not in class_objects:
                continue
                
            # Compare with all existing objects of the same class
            for obj_id, obj_data in class_objects[label]:
                # Skip if not seen recently (within 30 frames)
                if frame_idx - obj_data["last_seen"] > 30:
                    continue
                
                # Calculate IoU with last known position
                last_box = obj_data["trajectory"][-1][1]
                iou = self.calculate_iou(box, last_box)
                
                # Calculate embedding similarity
                emb_sim = cosine_similarity([det_embedding], [obj_data["embedding"]])[0][0]
                
                # Combined score using weighted average of IoU and embedding similarity
                combined_score = self.iou_weight * iou + self.emb_weight * emb_sim
                
                # Apply minimum thresholds for both IoU and embedding similarity
                # Only consider matches where both metrics meet minimum requirements
                if (iou >= self.min_iou_threshold and emb_sim >= self.min_emb_threshold):
                    match_scores.append((obj_id, i, combined_score, iou, emb_sim))
        
        # Sort by combined score
        match_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Assign matches
        matched_detections = set()
        matched_objects = set()
        
        for obj_id, det_idx, score, iou, emb_sim in match_scores:
            # Skip if already matched
            if obj_id in matched_objects or det_idx in matched_detections:
                continue
                
            # Only consider good matches
            if score > self.match_threshold:
                matched_objects.add(obj_id)
                matched_detections.add(det_idx)
                
                detection = detections[det_idx]
                box = detection["box"]
                
                # Extract crop and update embedding
                x1, y1, x2, y2 = [int(c) for c in box]
                if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
                    continue
                    
                crop = frame[y1:y2, x1:x2]
                det_embedding = embedding_extractor.extract(crop)
                
                # Update with moving average
                self.tracked_objects[obj_id]["embedding"] = (
                    0.7 * self.tracked_objects[obj_id]["embedding"] + 
                    0.3 * det_embedding
                )
                
                # Update trajectory
                self.tracked_objects[obj_id]["trajectory"].append((frame_idx, box))
                self.tracked_objects[obj_id]["last_seen"] = frame_idx
                
                current_objects[obj_id] = box
                print(f"Updated object {obj_id} ({detection['text']}) with score {score:.2f} (IoU: {iou:.2f}, Emb: {emb_sim:.2f})")
        
        # Handle new detections - either new objects or new instances of existing classes
        for i, detection in enumerate(detections):
            if i in matched_detections:
                continue
                
            box = detection["box"]
            label = detection["text"]
            
            # Extract crop and embedding
            x1, y1, x2, y2 = [int(c) for c in box]
            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
                continue
                
            crop = frame[y1:y2, x1:x2]
            det_embedding = embedding_extractor.extract(crop)
            
            # Check if this is a new instance of an existing class
            is_new_instance = True
            max_emb_sim = 0.0
            max_iou = 0.0
            
            # Look at all existing objects of the same class
            for obj_id, obj_data in self.tracked_objects.items():
                if obj_data["class"] == label:
                    # Calculate IoU with the most recent position
                    last_box = obj_data["trajectory"][-1][1]
                    iou = self.calculate_iou(box, last_box)
                    
                    # Calculate embedding similarity
                    emb_sim = cosine_similarity([det_embedding], [obj_data["embedding"]])[0][0]
                    
                    # Track maximum values
                    max_emb_sim = max(max_emb_sim, emb_sim)
                    max_iou = max(max_iou, iou)
                    
                    # If significant overlap or very high similarity with any existing instance,
                    # this might not be a new instance
                    if iou > self.new_instance_max_iou and emb_sim > self.min_emb_threshold:
                        is_new_instance = False
                        break
            
            # Create a new object if this is a new instance
            if is_new_instance:
                obj_id = self.next_obj_id
                self.next_obj_id += 1
                
                # Initialize object
                self.tracked_objects[obj_id] = {
                    "class": label,
                    "embedding": det_embedding,
                    "trajectory": [(frame_idx, box)],
                    "last_seen": frame_idx
                }
                
                current_objects[obj_id] = box
                if max_emb_sim > 0:
                    print(f"Created new instance {obj_id} of ({label}) - distinct from existing instances (max IoU: {max_iou:.2f}, max Emb: {max_emb_sim:.2f})")
                else:
                    print(f"Created new object {obj_id} ({label})")
        
        # Create tracking visualization if output_dir provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            
            plt.figure(figsize=(10, 10))
            plt.imshow(frame)
            plt.title(f"Frame {frame_idx} - Tracking")
            
            # Draw boxes for all current objects
            for obj_id, box in current_objects.items():
                obj_class = self.tracked_objects[obj_id]["class"]
                color = f"C{obj_id % 10}"
                
                # Draw box
                x1, y1, x2, y2 = box
                plt.gca().add_patch(
                    plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, 
                                 edgecolor=color, linewidth=2)
                )
                
                # Draw label
                plt.text(
                    x1, y1-10, f"{obj_id}: {obj_class}", 
                    bbox=dict(facecolor='white', alpha=0.8),
                    color=color, fontsize=12
                )
            
            plt.axis('off')
            plt.tight_layout()
            
            # Save visualization
            vis_path = os.path.join(output_dir, f"track_frame_{frame_idx:04d}.jpg")
            plt.savefig(vis_path, bbox_inches='tight', dpi=150)
            plt.close()
        
        return current_objects