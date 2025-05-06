# test_owlv2_with_saving.py
from src.frame_loader import load_frames_from_directory
from src.owlv2_detector import OWLv2Detector
import os

# Create output directory
output_dir = "./owl_detection_results"
os.makedirs(output_dir, exist_ok=True)

# Load frames
frames, _ = load_frames_from_directory("/home/ubuntu/code/drew/test_data/frames/Scenes 001-020__220D-2-_20230815190723523/subset")
detector = OWLv2Detector()

# Process and save visualizations for first few frames
for i in range(min(5, len(frames))):
    # Detect objects
    detections = detector.detect(frames[i], ["goat"], threshold=0.5)
    
    # Visualize and save
    detector.visualize_detections(
        frames[i], 
        detections, 
        output_dir=output_dir, 
        frame_idx=i,
        show=False  # Don't display, just save
    )

print(f"Saved detection visualizations to {output_dir}")
