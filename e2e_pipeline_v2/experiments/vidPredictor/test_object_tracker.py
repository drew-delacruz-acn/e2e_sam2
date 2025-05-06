# test_object_tracker.py
from src.frame_loader import load_frames_from_directory
from src.owlv2_detector import OWLv2Detector
from src.embedding_extractor import EmbeddingExtractor
from src.object_tracker import ObjectTracker
import os

# Create output directory
output_dir = "./tracking_results"
os.makedirs(output_dir, exist_ok=True)

# Load frames
frames, _ = load_frames_from_directory("./video_frames")

# Initialize components
detector = OWLv2Detector()
embedding_extractor = EmbeddingExtractor()
tracker = ObjectTracker(iou_weight=0.5, emb_weight=0.5, match_threshold=0.4)

# Process frames
for frame_idx, frame in enumerate(frames[:10]):  # First 10 frames
    print(f"\nProcessing frame {frame_idx}")
    
    # Detect objects
    detections = detector.detect(frame, ["person", "chair", "table"], threshold=0.3)
    
    # Save detection visualization
    detector.visualize_detections(
        frame, 
        detections, 
        output_dir=os.path.join(output_dir, "detections"),
        frame_idx=frame_idx,
        show=False
    )
    
    # Update tracking
    current_objects = tracker.update_tracks(
        frame, 
        frame_idx, 
        detections, 
        embedding_extractor,
        output_dir=os.path.join(output_dir, "tracking")
    )
    
    print(f"Frame {frame_idx}: {len(current_objects)} objects tracked")