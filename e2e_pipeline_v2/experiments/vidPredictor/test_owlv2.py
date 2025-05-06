# test_owlv2.py
from src.frame_loader import load_frames_from_directory
from src.owlv2_detector import OWLv2Detector

# Load a frame
frames, _ = load_frames_from_directory("/home/ubuntu/code/drew/test_data/frames/Scenes 001-020__220D-2-_20230815190723523/subset")
detector = OWLv2Detector()

# Test on a single frame
detections = detector.detect(frames[0], ["goat"])
detector.visualize_detections(frames[0], detections)