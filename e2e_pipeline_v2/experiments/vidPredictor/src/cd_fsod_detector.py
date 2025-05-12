import os
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union


class CDFSODDetector:
    """
    Detector class for CD-FSOD detections with continuity-based tracking.
    
    This detector loads CD-FSOD detections from JSON files and implements a continuity-based
    approach to tracking objects across frames. It detects both first appearances and
    reappearances of objects after a significant gap.
    """
    
    def __init__(
        self, 
        json_dir: str, 
        confidence_threshold: float = 0.2,
        iou_threshold: float = 0.5,  # Kept for backward compatibility but not used
        min_gap_frames: int = 10
    ):
        """
        Initialize the CD-FSOD detector.
        
        Args:
            json_dir: Directory containing JSON files with CD-FSOD detections
            confidence_threshold: Minimum confidence score for detections
            iou_threshold: No longer used - kept for backward compatibility
            min_gap_frames: Minimum number of frames an object must be absent to count as reappearance
        """
        self.json_dir = json_dir
        self.confidence_threshold = confidence_threshold
        self.min_gap_frames = min_gap_frames
        
        # Load all JSON files and process them
        self.detections_by_frame = self._load_detections()
        
        # Track objects and their appearances
        self.first_appearances = {}    # Maps frame_idx -> list of first appearances
        self.reappearances = {}        # Maps frame_idx -> list of reappearances
        self.object_tracks = {}        # Maps object_id -> list of (frame_idx, detection) tuples
        
        # Process detections to identify first appearances and reappearances
        self._process_detections()
    
    def _load_detections(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        Load all JSON files from the specified directory.
        
        Returns:
            Dictionary mapping frame indices to lists of detections
        """
        detections_by_frame = {}
        
        # List JSON files in the directory
        json_files = [f for f in os.listdir(self.json_dir) if f.endswith('.json')]
        
        # Sort files by frame index
        json_files.sort(key=lambda f: int(os.path.splitext(f)[0]))
        
        # Load each file
        for json_file in json_files:
            frame_idx = int(os.path.splitext(json_file)[0])
            file_path = os.path.join(self.json_dir, json_file)
            
            with open(file_path, 'r') as f:
                detections = json.load(f)
                
                # Filter detections by confidence threshold
                filtered_detections = [
                    d for d in detections 
                    if d.get('confidence', 0) >= self.confidence_threshold
                ]
                
                detections_by_frame[frame_idx] = filtered_detections
        
        return detections_by_frame
    
    def _process_detections(self):
        """
        Process all detections to identify first appearances and reappearances using continuity tracking.
        """
        # Initialize data structures for all frames
        frames = sorted(self.detections_by_frame.keys())
        for frame_idx in frames:
            self.first_appearances[frame_idx] = []
            self.reappearances[frame_idx] = []
        
        # Track when we last saw each label class
        # Maps label -> (last_frame_idx, is_active)
        last_seen = {}
        
        # Assign a single object_id for each label class
        # We're using simple continuity tracking now, not trying to differentiate 
        # between multiple instances of the same class
        label_to_id = {}
        
        # Process frames in order
        for frame_idx in frames:
            frame_detections = self.detections_by_frame[frame_idx]
            
            # Collect all object labels in this frame
            current_labels = set(d['label'] for d in frame_detections)
            
            # Check each detection to see if it's a first appearance or reappearance
            for detection in frame_detections:
                label = detection['label']
                
                # If we've never seen this label before
                if label not in last_seen:
                    # Assign an object_id for this label
                    label_to_id[label] = f"{label}_0"
                    
                    # Mark as a first appearance
                    detection['object_id'] = label_to_id[label]
                    self.first_appearances[frame_idx].append(detection)
                    
                    # Initialize the object track
                    self.object_tracks[label_to_id[label]] = [(frame_idx, detection)]
                    
                    # Record that we've seen this label
                    last_seen[label] = (frame_idx, True)  # Active status
                
                # If we've seen this label before
                else:
                    last_frame_idx, is_active = last_seen[label]
                    frame_gap = frame_idx - last_frame_idx
                    
                    # If the object is currently inactive and it's been gone for at least min_gap_frames
                    if not is_active and frame_gap >= self.min_gap_frames:
                        # Mark as a reappearance
                        detection['object_id'] = label_to_id[label]
                        self.reappearances[frame_idx].append(detection)
                        
                        # Update the object track
                        self.object_tracks[label_to_id[label]].append((frame_idx, detection))
                        
                        # Mark as active again
                        last_seen[label] = (frame_idx, True)
                    
                    # If the object is active or it hasn't been gone long enough
                    else:
                        # Just update the object track if it's the same object
                        if label in label_to_id:
                            detection['object_id'] = label_to_id[label]
                            self.object_tracks[label_to_id[label]].append((frame_idx, detection))
                        
                        # Update the last seen info
                        last_seen[label] = (frame_idx, True)
            
            # Update active status for objects not seen in this frame
            for label in last_seen:
                if label not in current_labels:
                    last_frame_idx, _ = last_seen[label]
                    last_seen[label] = (last_frame_idx, False)
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        Note: This method is kept for backward compatibility but is no longer used.
        
        Args:
            box1: First box coordinates [x1, y1, x2, y2]
            box2: Second box coordinates [x1, y1, x2, y2]
            
        Returns:
            IoU value between 0 and 1
        """
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Check if boxes overlap
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate box areas
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate union area
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area
        
        return iou
    
    def detect(
        self, 
        image: Union[np.ndarray, Dict], 
        text_queries: List[str], 
        threshold: Optional[float] = None
    ) -> Dict[str, Union[np.ndarray, List[str]]]:
        """
        Detect objects in the given frame that either first appear or reappear after a gap.
        
        Args:
            image: The image/frame to detect objects in or a dictionary with frame data
            text_queries: List of object classes to detect
            threshold: Optional confidence threshold (overrides the default)
            
        Returns:
            Dictionary containing 'boxes', 'labels', and 'scores' for detected objects
        """
        # Extract frame index from filename or metadata
        frame_idx = self._extract_frame_idx(image)
        
        if frame_idx is None:
            # If we can't determine the frame index, return empty results
            return {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "labels": [],
                "scores": np.zeros(0, dtype=np.float32)
            }
        
        # IMPORTANT: Only include first appearances and reappearances, not all detections in the frame
        # This ensures we're only detecting objects when they first appear or reappear after a gap
        frame_detections = []
        
        # Add first appearances for this frame
        if frame_idx in self.first_appearances:
            frame_detections.extend(self.first_appearances[frame_idx])
            
        # Add reappearances for this frame
        if frame_idx in self.reappearances:
            frame_detections.extend(self.reappearances[frame_idx])
        
        # We deliberately DO NOT include other detections from self.detections_by_frame[frame_idx]
        # as those would include continuing objects which we want to ignore
        
        # Apply label filtering using text queries
        filtered_detections = []
        for detection in frame_detections:
            label = detection["label"]
            
            # Check if this label matches any of the requested queries
            if not text_queries or "all" in text_queries or label in text_queries:
                filtered_detections.append(detection)
        
        # Prepare output in the format expected by the pipeline
        if filtered_detections:
            boxes = np.array([d["coordinates"] for d in filtered_detections])
            labels = [d["label"] for d in filtered_detections]
            scores = np.array([d["confidence"] for d in filtered_detections])
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = []
            scores = np.zeros(0, dtype=np.float32)
        
        return {
            "boxes": boxes,
            "labels": labels,
            "scores": scores
        }
    
    def _extract_frame_idx(self, image: Union[np.ndarray, Dict]) -> Optional[int]:
        """
        Extract frame index from image metadata or filename.
        
        In a real implementation, this would extract the frame index from metadata
        or from the filename pattern. For testing, we'll create a mock implementation
        that works with the expected test images.
        
        Args:
            image: The input image or image data dictionary
            
        Returns:
            Frame index or None if it cannot be determined
        """
        # Handle dictionary input format (used by the pipeline)
        if isinstance(image, dict) and 'frame_idx' in image:
            return image['frame_idx']
        
        # Handle MockImage format with frame_idx property
        if hasattr(image, 'frame_idx'):
            return image.frame_idx
        
        # Return None if we can't determine the frame index
        return None 