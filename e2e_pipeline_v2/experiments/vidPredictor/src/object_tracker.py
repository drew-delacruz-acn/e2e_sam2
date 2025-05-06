# object_tracker.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ObjectTracker:
    def __init__(self):
        """Initialize simple two-tier object tracker"""
        self.tracked_objects = {}
        self.next_obj_id = 1
    
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
    
    def update_tracks(self, frame, frame_idx, detections, embedding_extractor):
        """Update object tracks with new detections
        
        Args:
            frame: Current video frame
            frame_idx: Current frame index
            detections: List of detections from OWLv2
            embedding_extractor: Feature extractor for appearance matching
            
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
            det_box = detection["box"]
            
            # Extract crop and embedding
            x1, y1, x2, y2 = [int(c) for c in det_box]
            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
                # Skip invalid boxes
                continue
                
            crop = frame[y1:y2, x1:x2]
            det_embedding = embedding_extractor.extract(crop)
            
            # Compare with all existing objects
            for obj_id, obj_data in self.tracked_objects.items():
                # Skip if not seen recently
                if frame_idx - obj_data["last_seen"] > 30:
                    continue
                    
                # Calculate IoU with last known position
                last_box = obj_data["trajectory"][-1][1]
                iou = self.calculate_iou(det_box, last_box)
                
                # Calculate embedding similarity
                emb_sim = cosine_similarity([det_embedding], [obj_data["embedding"]])[0][0]
                
                # Combined score
                combined_score = 0.5 * iou + 0.5 * emb_sim
                
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
            if score > 0.4:  # Threshold
                matched_objects.add(obj_id)
                matched_detections.add(det_idx)
                
                detection = detections[det_idx]
                box = detection["box"]
                
                # Extract crop and update embedding
                x1, y1, x2, y2 = [int(c) for c in box]
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
                print(f"Updated object {obj_id} ({detection['text']}) with score {score:.2f}")
        
        # Handle new objects
        for i, detection in enumerate(detections):
            if i not in matched_detections:
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
                print(f"Created new object {obj_id} ({detection['text']})")
        
        return current_objects