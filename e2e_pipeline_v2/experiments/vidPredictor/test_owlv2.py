# test_owlv2.py
from src.frame_loader import load_frames_from_directory
from src.owlv2_detector import OWLv2Detector

# Load a frame
frames, _ = load_frames_from_directory("./video_frames")
detector = OWLv2Detector()

# Test on a single frame
detections = detector.detect(frames[0], ["person", "chair", "table"])
detector.visualize_detections(frames[0], detections)