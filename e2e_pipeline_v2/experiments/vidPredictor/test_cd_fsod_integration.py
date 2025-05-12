#!/usr/bin/env python3
"""
Test script for CD-FSOD detector integration with the object tracking pipeline.
"""

import argparse
import os
import sys
import logging
import time
import json
import re
from pathlib import Path
from datetime import datetime
from functools import wraps

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from object_tracking_pipeline import ObjectTrackingPipeline
import cd_fsod_detector

# Add natural sorting function
def natural_sort_key(s):
    """Key function for natural sorting"""
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', str(s))]

# Store the original methods
original_load_detections = cd_fsod_detector.CDFSODDetector._load_detections
original_process_detections = cd_fsod_detector.CDFSODDetector._process_detections
original_detect = cd_fsod_detector.CDFSODDetector.detect

# Configure logging
def setup_logger(log_level=logging.INFO, output_dir=None):
    """Set up a logger with console and file handlers."""
    # Create logger
    logger = logging.getLogger("cd_fsod_test")
    logger.setLevel(log_level)
    logger.handlers = []  # Clear existing handlers
    
    # Create console handler with formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    # Add file handler if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, f"cd_fsod_test_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    return logger

# Enhanced version of _load_detections with logging
def enhanced_load_detections(self):
    logger = logging.getLogger("cd_fsod_test")
    logger.info(f"Loading CD-FSOD detections from {self.json_dir}...")
    
    start_time = time.time()
    detections_by_frame = original_load_detections(self)
    
    total_detections = sum(len(dets) for dets in detections_by_frame.values())
    unique_classes = set()
    confidence_stats = {"min": float('inf'), "max": float('-inf'), "sum": 0, "count": 0}
    confidence_by_class = {}
    
    for frame_idx, frame_dets in detections_by_frame.items():
        for det in frame_dets:
            label = det.get('label', '')
            unique_classes.add(label)
            
            # Track confidence statistics
            confidence = det.get('confidence', 0)
            confidence_stats["min"] = min(confidence_stats["min"], confidence)
            confidence_stats["max"] = max(confidence_stats["max"], confidence)
            confidence_stats["sum"] += confidence
            confidence_stats["count"] += 1
            
            # Track confidence by class
            if label not in confidence_by_class:
                confidence_by_class[label] = {"min": float('inf'), "max": float('-inf'), "sum": 0, "count": 0}
            
            confidence_by_class[label]["min"] = min(confidence_by_class[label]["min"], confidence)
            confidence_by_class[label]["max"] = max(confidence_by_class[label]["max"], confidence)
            confidence_by_class[label]["sum"] += confidence
            confidence_by_class[label]["count"] += 1
    
    # Log the confidence statistics
    if confidence_stats["count"] > 0:
        avg_confidence = confidence_stats["sum"] / confidence_stats["count"]
        logger.info(f"Detection confidence: min={confidence_stats['min']:.6f}, max={confidence_stats['max']:.6f}, avg={avg_confidence:.6f}")
        
        # Log confidence by class
        logger.info("Confidence by class:")
        for label, stats in confidence_by_class.items():
            if stats["count"] > 0:
                class_avg = stats["sum"] / stats["count"]
                logger.info(f"  {label}: min={stats['min']:.6f}, max={stats['max']:.6f}, avg={class_avg:.6f}, count={stats['count']}")
    
    logger.info(f"Loaded {len(detections_by_frame)} frames with {total_detections} total detections")
    logger.info(f"Found {len(unique_classes)} unique object classes: {sorted(list(unique_classes))}")
    logger.info(f"Detection loading took {time.time() - start_time:.2f} seconds")
    
    # Log detailed stats in debug mode
    if logger.level <= logging.DEBUG:
        logger.debug("Detections per frame:")
        for frame_idx in sorted(detections_by_frame.keys())[:10]:  # Show first 10 frames only
            frame_dets = detections_by_frame[frame_idx]
            if frame_dets:
                confidences = [f"{det.get('confidence', 0):.6f}" for det in frame_dets[:3]]
                labels = [det.get('label', '') for det in frame_dets[:3]]
                logger.debug(f"  Frame {frame_idx}: {len(frame_dets)} detections - Labels: {labels[:3]}, Confidences: {confidences[:3]}")
                if len(frame_dets) > 3:
                    logger.debug(f"    ... and {len(frame_dets) - 3} more")
            else:
                logger.debug(f"  Frame {frame_idx}: {len(frame_dets)} detections")
    
    return detections_by_frame

# Enhanced version of _process_detections with logging
def enhanced_process_detections(self):
    logger = logging.getLogger("cd_fsod_test")
    logger.info("Processing detections to identify first appearances and reappearances...")
    
    start_time = time.time()
    original_process_detections(self)
    
    # Log stats after processing
    total_first_appearances = sum(len(apps) for apps in self.first_appearances.values())
    total_reappearances = sum(len(reapps) for reapps in self.reappearances.values())
    
    logger.info(f"Processed detections in {time.time() - start_time:.2f} seconds")
    logger.info(f"Found {total_first_appearances} first appearances and {total_reappearances} reappearances")
    logger.info(f"Using minimum gap of {self.min_gap_frames} frames for reappearance detection")
    
    # Log more detailed stats in debug mode
    if logger.level <= logging.DEBUG:
        logger.debug("First appearances per frame:")
        for frame_idx in sorted(self.first_appearances.keys())[:10]:  # Show first 10 frames
            if len(self.first_appearances[frame_idx]) > 0:
                logger.debug(f"  Frame {frame_idx}: {len(self.first_appearances[frame_idx])} " + 
                             f"objects: {[d.get('label') for d in self.first_appearances[frame_idx]]}")
        
        logger.debug("Reappearances per frame:")
        for frame_idx in sorted(self.reappearances.keys())[:10]:  # Show first 10 frames
            if len(self.reappearances[frame_idx]) > 0:
                logger.debug(f"  Frame {frame_idx}: {len(self.reappearances[frame_idx])} " + 
                             f"objects: {[d.get('label') for d in self.reappearances[frame_idx]]}")
    
    return

# Enhanced version of detect with logging
def enhanced_detect(self, image, text_queries, threshold=None):
    logger = logging.getLogger("cd_fsod_test")
    
    # Extract frame info
    frame_idx = None
    if isinstance(image, dict) and 'frame_idx' in image:
        frame_idx = image['frame_idx']
    else:
        frame_idx = self._extract_frame_idx(image)
    
    if frame_idx is not None:
        logger.debug(f"Detecting objects in frame {frame_idx}")
    
    # Call original method
    start_time = time.time()
    result = original_detect(self, image, text_queries, threshold)
    
    # Log results
    num_detections = len(result["boxes"])
    if num_detections > 0:
        logger.debug(f"Frame {frame_idx}: Detected {num_detections} objects " +
                    f"{result['labels']} with scores {[f'{s:.2f}' for s in result['scores']]}")
    else:
        logger.debug(f"Frame {frame_idx}: No objects detected")
    
    logger.debug(f"Detection took {(time.time() - start_time)*1000:.1f}ms")
    
    return result

def apply_detector_patches():
    """Apply monkey patches to the CD-FSOD detector class to enhance logging."""
    logger = logging.getLogger("cd_fsod_test")
    logger.info("Applying enhanced logging patches to CD-FSOD detector...")
    
    # Patch the detector methods
    cd_fsod_detector.CDFSODDetector._load_detections = enhanced_load_detections
    cd_fsod_detector.CDFSODDetector._process_detections = enhanced_process_detections
    cd_fsod_detector.CDFSODDetector.detect = enhanced_detect
    
    logger.info("CD-FSOD detector patched with enhanced logging")

def log_pipeline_progress(pipeline, total_frames, frame_idx, interval=5):
    """Log progress of the pipeline processing with object statistics."""
    if frame_idx % interval != 0 and frame_idx != total_frames - 1:
        return
        
    logger = logging.getLogger("cd_fsod_test")
    progress_pct = (frame_idx + 1) / total_frames * 100
    
    # Get object statistics
    total_objects = len(pipeline.tracked_objects) if hasattr(pipeline, 'tracked_objects') else 0
    
    logger.info(f"Progress: {progress_pct:.1f}% ({frame_idx+1}/{total_frames} frames) - Tracking {total_objects} objects")
    
    # Log more detailed object info in debug mode
    if logger.level <= logging.DEBUG and total_objects > 0:
        # Get object classes
        object_classes = {}
        for obj_id, obj_data in pipeline.tracked_objects.items():
            obj_class = obj_data.get('class', 'unknown')
            if obj_class not in object_classes:
                object_classes[obj_class] = 0
            object_classes[obj_class] += 1
            
        # Log object class distribution
        class_info = ", ".join([f"{cls}: {count}" for cls, count in object_classes.items()])
        logger.debug(f"Object distribution: {class_info}")
        
        # Log some example object details
        if frame_idx > 0:
            # Find objects visible in current frame
            visible_objects = [
                obj_id for obj_id, data in pipeline.tracked_objects.items()
                if data.get('last_seen') == frame_idx
            ]
            
            if visible_objects:
                sample_obj_id = visible_objects[0]
                obj_data = pipeline.tracked_objects[sample_obj_id]
                logger.debug(f"Sample object {sample_obj_id} ({obj_data.get('class')}): "
                           f"first seen at frame {obj_data.get('first_detected')}, "
                           f"last seen at frame {obj_data.get('last_seen')}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test CD-FSOD detector integration with the object tracking pipeline")
    
    # Data paths
    parser.add_argument("--frames-dir", required=True, help="Directory containing video frames")
    parser.add_argument("--cd-fsod-path", required=True, help="Path to CD-FSOD JSON detections directory")
    parser.add_argument("--output-dir", default="./cd_fsod_results", help="Output directory for results")
    
    # Model paths
    parser.add_argument("--sam2-checkpoint", required=True, help="Path to SAM2 checkpoint")
    parser.add_argument("--sam2-config", required=True, help="Path to SAM2 config file")
    
    # Detection parameters
    parser.add_argument("--text-queries", default=["all"], nargs="+", help="Text queries for object detection (default: 'all')")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for detections")
    parser.add_argument("--min-gap-frames", type=int, default=10, help="Minimum gap frames for CD-FSOD reappearances")
    
    # Processing options
    parser.add_argument("--separate-objects", action="store_true", help="Process each object separately to avoid dtype issues")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-enhanced-logging", action="store_true", help="Disable enhanced detector logging")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Set up logger
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(log_level=log_level, output_dir=args.output_dir)
    
    # Log script start and configuration
    logger.info("=" * 80)
    logger.info("CD-FSOD Integration Test Started")
    logger.info("=" * 80)
    logger.info("Test configuration:")
    logger.info(f"  - Frames directory: {args.frames_dir}")
    logger.info(f"  - CD-FSOD detections directory: {args.cd_fsod_path}")
    logger.info(f"  - SAM2 checkpoint: {args.sam2_checkpoint}")
    logger.info(f"  - SAM2 config: {args.sam2_config}")
    logger.info(f"  - Output directory: {args.output_dir}")
    logger.info(f"  - Confidence threshold: {args.confidence}")
    logger.info(f"  - Minimum gap frames: {args.min_gap_frames}")
    logger.info(f"  - Text queries: {args.text_queries}")
    logger.info(f"  - Using separate objects: {args.separate_objects}")
    logger.info(f"  - Debug mode: {args.debug}")
    logger.info(f"  - Enhanced logging: {not args.no_enhanced_logging}")
    
    # Check for files in the frames directory
    try:
        frames_path = Path(args.frames_dir)
        frame_files = sorted([f for f in frames_path.glob("*.jpg") or frames_path.glob("*.png")], key=natural_sort_key)
        logger.info(f"Found {len(frame_files)} frames in {args.frames_dir}")
        if len(frame_files) == 0:
            logger.error(f"No frames found in {args.frames_dir}. Exiting.")
            return
        logger.debug(f"First 5 frames: {[f.name for f in frame_files[:5]]}")
    except Exception as e:
        logger.error(f"Error accessing frames directory: {e}")
        return
    
    # Check for CD-FSOD JSON files
    try:
        cd_fsod_path = Path(args.cd_fsod_path)
        json_files = sorted([f for f in cd_fsod_path.glob("*.json")], key=natural_sort_key)
        logger.info(f"Found {len(json_files)} JSON detection files in {args.cd_fsod_path}")
        if len(json_files) == 0:
            logger.error(f"No JSON files found in {args.cd_fsod_path}. Exiting.")
            return
        logger.debug(f"First 5 JSON files: {[f.name for f in json_files[:5]]}")
        
        # Sample the first JSON file to show format
        if logger.level <= logging.DEBUG and len(json_files) > 0:
            try:
                with open(json_files[0], 'r') as f:
                    sample_data = json.load(f)
                logger.debug(f"Sample JSON format (first file, up to 3 detections):")
                for i, det in enumerate(sample_data[:3]):
                    logger.debug(f"  Detection {i+1}: {det}")
                if len(sample_data) > 3:
                    logger.debug(f"  ... and {len(sample_data)-3} more detections")
            except Exception as e:
                logger.error(f"Error reading sample JSON file: {e}")
    except Exception as e:
        logger.error(f"Error accessing CD-FSOD directory: {e}")
        return
    
    # Apply detector patches for enhanced logging if requested
    if not args.no_enhanced_logging:
        try:
            apply_detector_patches()
        except Exception as e:
            logger.error(f"Error applying detector patches: {e}")
            logger.warning("Continuing without enhanced detector logging")
    
    # Initialize timer
    start_time = time.time()
    logger.info("Initializing pipeline with CD-FSOD detector...")
    
    try:
        # Initialize pipeline with CD-FSOD detector
        pipeline = ObjectTrackingPipeline(
            owlv2_checkpoint=None,  # Not used with CD-FSOD detector
            sam2_checkpoint=args.sam2_checkpoint,
            sam2_config=args.sam2_config,
            output_dir=args.output_dir,
            confidence_threshold=args.confidence,
            detector_type="cd_fsod",  # Use CD-FSOD detector
            cd_fsod_path=args.cd_fsod_path,
            min_gap_frames=args.min_gap_frames
        )
        logger.info("Pipeline initialized successfully")
        logger.info(f"Pipeline initialization took {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error initializing pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Process video
    processing_start = time.time()
    logger.info("Starting video processing...")
    
    try:
        # Process video using the appropriate method
        if args.separate_objects:
            logger.info("Using separate object initialization method...")
            
            # Define a progress monitoring callback
            def progress_callback(frame_idx, total_frames):
                log_pipeline_progress(pipeline, total_frames, frame_idx)
            
            # Store the original method
            original_process_video = pipeline.process_video_separate_objects
            
            # Create a wrapped version with progress monitoring
            @wraps(original_process_video)
            def wrapped_process_video(frames_dir, text_queries):
                # Get total frame count
                frames_path = Path(frames_dir)
                frame_files = sorted([f for f in frames_path.glob("*.jpg") or frames_path.glob("*.png")], key=natural_sort_key)
                total_frames = len(frame_files)
                
                # Add hooks for progress logging
                original_process_frame = pipeline._process_frame if hasattr(pipeline, '_process_frame') else None
                
                if original_process_frame:
                    @wraps(original_process_frame)
                    def wrapped_process_frame(frame_idx, *args, **kwargs):
                        result = original_process_frame(frame_idx, *args, **kwargs)
                        # Log progress
                        log_pipeline_progress(pipeline, total_frames, frame_idx)
                        return result
                    
                    pipeline._process_frame = wrapped_process_frame
                
                logger.info(f"Starting processing of {total_frames} frames...")
                return original_process_video(frames_dir, text_queries)
            
            # Replace the method with our wrapped version
            pipeline.process_video_separate_objects = wrapped_process_video
            
            # Call the wrapped method
            pipeline.process_video_separate_objects(
                frames_dir=args.frames_dir,
                text_queries=args.text_queries
            )
        else:
            logger.info("Using standard processing method...")
            
            # Store the original method
            original_process_video = pipeline.process_video
            
            # Create a wrapped version with progress monitoring
            @wraps(original_process_video)
            def wrapped_process_video(frames_dir, text_queries):
                # Get total frame count
                frames_path = Path(frames_dir)
                frame_files = sorted([f for f in frames_path.glob("*.jpg") or frames_path.glob("*.png")], key=natural_sort_key)
                total_frames = len(frame_files)
                
                # Add hooks for progress logging
                original_process_frame = pipeline._process_frame if hasattr(pipeline, '_process_frame') else None
                
                if original_process_frame:
                    @wraps(original_process_frame)
                    def wrapped_process_frame(frame_idx, *args, **kwargs):
                        result = original_process_frame(frame_idx, *args, **kwargs)
                        # Log progress
                        log_pipeline_progress(pipeline, total_frames, frame_idx)
                        return result
                    
                    pipeline._process_frame = wrapped_process_frame
                
                logger.info(f"Starting processing of {total_frames} frames...")
                return original_process_video(frames_dir, text_queries)
            
            # Replace the method with our wrapped version
            pipeline.process_video = wrapped_process_video
            
            # Call the wrapped method
            pipeline.process_video(
                frames_dir=args.frames_dir,
                text_queries=args.text_queries
            )
        
        processing_time = time.time() - processing_start
        logger.info(f"Video processing completed in {processing_time:.2f} seconds")
        
        # Log first detection information if in debug mode
        if args.debug:
            logger.debug("First detection details:")
            for obj_id, obj_data in pipeline.tracked_objects.items():
                first_frame_idx = obj_data.get("first_detected")
                obj_class = obj_data.get("class", "unknown")
                logger.debug(f"  Object #{obj_id} ({obj_class}) first detected at frame {first_frame_idx}")
                
                # Add detection info if using CD-FSOD detector
                if hasattr(pipeline.detector, 'json_dir'):
                    json_dir = Path(pipeline.detector.json_dir)
                    json_file = json_dir / f"{first_frame_idx}.json"
                    if json_file.exists():
                        try:
                            with open(json_file, 'r') as f:
                                json_data = json.load(f)
                                relevant_detections = [d for d in json_data if d.get('label') == obj_class]
                                if relevant_detections:
                                    sample = relevant_detections[0]
                                    logger.debug(f"    From {json_file.name}: {sample}")
                        except Exception as e:
                            logger.error(f"    Error reading JSON file {json_file}: {e}")
        
        # Check if results were generated
        results_files = list(Path(args.output_dir).glob("*"))
        logger.info(f"Generated {len(results_files)} output files")
        if args.debug:
            logger.debug(f"Output files: {[f.name for f in results_files[:10]]}")
        
        # Log completion
        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"CD-FSOD Integration Test Completed Successfully")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during video processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=" * 80)
        logger.info("CD-FSOD Integration Test Failed")
        logger.info("=" * 80)

if __name__ == "__main__":
    main() 


# python e2e_pipeline_v2/experiments/vidPredictor/test_cd_fsod_integration.py  --frames-dir "/home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/data/frames/Scenes 061-080__265H-2-_20230815215828529" --cd-fsod-path "/home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/data/detections/Scenes 061-080__265H-2-_20230815215828529" --sam2-checkpoint checkpoints/sam2.1_hiera_large.pt --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml --confidence 0.9 --min-gap-frames 10 --separate-objects --text-queries "all" --debug
# 2025-05-12 20:47:07,749 - INFO - Logging to file: ./cd_fsod_results/cd_fsod_test_20250512_204707.log
# 2025-05-12 20:47:07,749 - INFO - ================================================================================
# 2025-05-12 20:47:07,749 - INFO - CD-FSOD Integration Test Started
# 2025-05-12 20:47:07,749 - INFO - ================================================================================
# 2025-05-12 20:47:07,749 - INFO - Test configuration:
# 2025-05-12 20:47:07,749 - INFO -   - Frames directory: /home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/data/frames/Scenes 061-080__265H-2-_20230815215828529
# 2025-05-12 20:47:07,749 - INFO -   - CD-FSOD detections directory: /home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/data/detections/Scenes 061-080__265H-2-_20230815215828529
# 2025-05-12 20:47:07,749 - INFO -   - SAM2 checkpoint: checkpoints/sam2.1_hiera_large.pt
# 2025-05-12 20:47:07,749 - INFO -   - SAM2 config: configs/sam2.1/sam2.1_hiera_l.yaml
# 2025-05-12 20:47:07,749 - INFO -   - Output directory: ./cd_fsod_results
# 2025-05-12 20:47:07,749 - INFO -   - Confidence threshold: 0.9
# 2025-05-12 20:47:07,750 - INFO -   - Minimum gap frames: 10
# 2025-05-12 20:47:07,750 - INFO -   - Text queries: ['all']
# 2025-05-12 20:47:07,750 - INFO -   - Using separate objects: True
# 2025-05-12 20:47:07,750 - INFO -   - Debug mode: True
# 2025-05-12 20:47:07,750 - INFO -   - Enhanced logging: True
# 2025-05-12 20:47:07,751 - INFO - Found 54 frames in /home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/data/frames/Scenes 061-080__265H-2-_20230815215828529
# 2025-05-12 20:47:07,751 - DEBUG - First 5 frames: ['0.jpg', '1.jpg', '2.jpg', '3.jpg', '4.jpg']
# 2025-05-12 20:47:07,752 - INFO - Found 54 JSON detection files in /home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/data/detections/Scenes 061-080__265H-2-_20230815215828529
# 2025-05-12 20:47:07,752 - DEBUG - First 5 JSON files: ['0.json', '1.json', '2.json', '3.json', '4.json']
# 2025-05-12 20:47:07,752 - DEBUG - Sample JSON format (first file, up to 3 detections):
# 2025-05-12 20:47:07,752 - DEBUG -   Detection 1: {'coordinates': [321, 145, 456, 239], 'label': 'TVA Monitor', 'confidence': 0.0064104781486094}
# 2025-05-12 20:47:07,752 - DEBUG -   Detection 2: {'coordinates': [232, 143, 301, 223], 'label': 'TVA Monitor', 'confidence': 0.0029197395779192448}
# 2025-05-12 20:47:07,752 - DEBUG -   Detection 3: {'coordinates': [234, 151, 246, 219], 'label': 'TVA Monitor', 'confidence': 0.0007871381822042167}
# 2025-05-12 20:47:07,752 - DEBUG -   ... and 39 more detections
# 2025-05-12 20:47:07,752 - INFO - Applying enhanced logging patches to CD-FSOD detector...
# 2025-05-12 20:47:07,752 - INFO - CD-FSOD detector patched with enhanced logging
# 2025-05-12 20:47:07,752 - INFO - Initializing pipeline with CD-FSOD detector...
# Using device: cuda
# 2025-05-12 20:47:07,882 - INFO - Loading CD-FSOD detections from /home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/data/detections/Scenes 061-080__265H-2-_20230815215828529...
# 2025-05-12 20:47:07,888 - INFO - Detection confidence: min=0.906447, max=0.996994, avg=0.965909
# 2025-05-12 20:47:07,888 - INFO - Confidence by class:
# 2025-05-12 20:47:07,889 - INFO -   Time Stick: min=0.919637, max=0.996994, avg=0.963398, count=13
# 2025-05-12 20:47:07,889 - INFO -   TVA Uniform: min=0.966902, max=0.993916, avg=0.982981, count=4
# 2025-05-12 20:47:07,889 - INFO -   Sylvie's horned headpiece: min=0.906447, max=0.985599, avg=0.956999, count=4
# 2025-05-12 20:47:07,889 - INFO - Loaded 54 frames with 21 total detections
# 2025-05-12 20:47:07,889 - INFO - Found 3 unique object classes: ["Sylvie's horned headpiece", 'TVA Uniform', 'Time Stick']
# 2025-05-12 20:47:07,889 - INFO - Detection loading took 0.01 seconds
# 2025-05-12 20:47:07,889 - DEBUG - Detections per frame:
# 2025-05-12 20:47:07,889 - DEBUG -   Frame 0: 0 detections
# 2025-05-12 20:47:07,889 - DEBUG -   Frame 1: 0 detections
# 2025-05-12 20:47:07,889 - DEBUG -   Frame 2: 0 detections
# 2025-05-12 20:47:07,889 - DEBUG -   Frame 3: 1 detections - Labels: ['Time Stick'], Confidences: ['0.993159']
# 2025-05-12 20:47:07,889 - DEBUG -   Frame 4: 1 detections - Labels: ['Time Stick'], Confidences: ['0.996994']
# 2025-05-12 20:47:07,889 - DEBUG -   Frame 5: 1 detections - Labels: ['Time Stick'], Confidences: ['0.972391']
# 2025-05-12 20:47:07,889 - DEBUG -   Frame 6: 1 detections - Labels: ['Time Stick'], Confidences: ['0.983048']
# 2025-05-12 20:47:07,889 - DEBUG -   Frame 7: 1 detections - Labels: ['Time Stick'], Confidences: ['0.985463']
# 2025-05-12 20:47:07,889 - DEBUG -   Frame 8: 1 detections - Labels: ['Time Stick'], Confidences: ['0.960590']
# 2025-05-12 20:47:07,889 - DEBUG -   Frame 9: 1 detections - Labels: ['Time Stick'], Confidences: ['0.968005']
# 2025-05-12 20:47:07,889 - INFO - Processing detections to identify first appearances and reappearances...
# 2025-05-12 20:47:07,889 - INFO - Processed detections in 0.00 seconds
# 2025-05-12 20:47:07,889 - INFO - Found 3 first appearances and 2 reappearances
# 2025-05-12 20:47:07,889 - INFO - Using minimum gap of 10 frames for reappearance detection
# 2025-05-12 20:47:07,889 - DEBUG - First appearances per frame:
# 2025-05-12 20:47:07,889 - DEBUG -   Frame 3: 1 objects: ['Time Stick']
# 2025-05-12 20:47:07,890 - DEBUG - Reappearances per frame:
# SAM2 using device: cuda
# /home/ubuntu/code/drew/e2e_sam2/venv/lib/python3.12/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
#   warnings.warn(
# /home/ubuntu/code/drew/e2e_sam2/venv/lib/python3.12/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet50_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet50_Weights.DEFAULT` to get the most up-to-date weights.
#   warnings.warn(msg)
# 2025-05-12 20:47:11,521 - INFO - Pipeline initialized successfully
# INFO:cd_fsod_test:Pipeline initialized successfully
# 2025-05-12 20:47:11,521 - INFO - Pipeline initialization took 3.77 seconds
# INFO:cd_fsod_test:Pipeline initialization took 3.77 seconds
# 2025-05-12 20:47:11,521 - INFO - Starting video processing...
# INFO:cd_fsod_test:Starting video processing...
# 2025-05-12 20:47:11,521 - INFO - Using separate object initialization method...
# INFO:cd_fsod_test:Using separate object initialization method...
# 2025-05-12 20:47:11,522 - INFO - Starting processing of 54 frames...
# INFO:cd_fsod_test:Starting processing of 54 frames...
# Processing 54 frames with queries: ['all']
# Using detector: cd_fsod
# Frame 0 in sequence maps to frame index 0
# Frame 1 in sequence maps to frame index 1
# Frame 2 in sequence maps to frame index 2
# Frame 3 in sequence maps to frame index 3
# Frame 4 in sequence maps to frame index 4
# Frame 5 in sequence maps to frame index 5
# Frame 6 in sequence maps to frame index 6
# Frame 7 in sequence maps to frame index 7
# Frame 8 in sequence maps to frame index 8
# Frame 9 in sequence maps to frame index 9
# Frame 10 in sequence maps to frame index 10
# Frame 11 in sequence maps to frame index 11
# Frame 12 in sequence maps to frame index 12
# Frame 13 in sequence maps to frame index 13
# Frame 14 in sequence maps to frame index 14
# Frame 15 in sequence maps to frame index 15
# Frame 16 in sequence maps to frame index 16
# Frame 17 in sequence maps to frame index 17
# Frame 18 in sequence maps to frame index 18
# Frame 19 in sequence maps to frame index 19
# Frame 20 in sequence maps to frame index 20
# Frame 21 in sequence maps to frame index 21
# Frame 22 in sequence maps to frame index 22
# Frame 23 in sequence maps to frame index 23
# Frame 24 in sequence maps to frame index 24
# Frame 25 in sequence maps to frame index 25
# Frame 26 in sequence maps to frame index 26
# Frame 27 in sequence maps to frame index 27
# Frame 28 in sequence maps to frame index 28
# Frame 29 in sequence maps to frame index 29
# Frame 30 in sequence maps to frame index 30
# Frame 31 in sequence maps to frame index 31
# Frame 32 in sequence maps to frame index 32
# Frame 33 in sequence maps to frame index 33
# Frame 34 in sequence maps to frame index 34
# Frame 35 in sequence maps to frame index 35
# Frame 36 in sequence maps to frame index 36
# Frame 37 in sequence maps to frame index 37
# Frame 38 in sequence maps to frame index 38
# Frame 39 in sequence maps to frame index 39
# Frame 40 in sequence maps to frame index 40
# Frame 41 in sequence maps to frame index 41
# Frame 42 in sequence maps to frame index 42
# Frame 43 in sequence maps to frame index 43
# Frame 44 in sequence maps to frame index 44
# Frame 45 in sequence maps to frame index 45
# Frame 46 in sequence maps to frame index 46
# Frame 47 in sequence maps to frame index 47
# Frame 48 in sequence maps to frame index 48
# Frame 49 in sequence maps to frame index 49
# Frame 50 in sequence maps to frame index 50
# Frame 51 in sequence maps to frame index 51
# Frame 52 in sequence maps to frame index 52
# Frame 53 in sequence maps to frame index 53
# Phase 1: Detecting and tracking objects...
# 2025-05-12 20:47:11,531 - DEBUG - Detecting objects in frame 0
# DEBUG:cd_fsod_test:Detecting objects in frame 0
# 2025-05-12 20:47:11,531 - DEBUG - Frame 0: No objects detected
# DEBUG:cd_fsod_test:Frame 0: No objects detected
# 2025-05-12 20:47:11,531 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# -----Initializing new objects-----
# detections: []
# 2025-05-12 20:47:11,534 - DEBUG - Detecting objects in frame 1
# DEBUG:cd_fsod_test:Detecting objects in frame 1
# 2025-05-12 20:47:11,534 - DEBUG - Frame 1: No objects detected
# DEBUG:cd_fsod_test:Frame 1: No objects detected
# 2025-05-12 20:47:11,534 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# -----Initializing new objects-----
# detections: []
# 2025-05-12 20:47:11,538 - DEBUG - Detecting objects in frame 2
# DEBUG:cd_fsod_test:Detecting objects in frame 2
# 2025-05-12 20:47:11,538 - DEBUG - Frame 2: No objects detected
# DEBUG:cd_fsod_test:Frame 2: No objects detected
# 2025-05-12 20:47:11,538 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# -----Initializing new objects-----
# detections: []
# 2025-05-12 20:47:11,541 - DEBUG - Detecting objects in frame 3
# DEBUG:cd_fsod_test:Detecting objects in frame 3
# 2025-05-12 20:47:11,541 - DEBUG - Frame 3: Detected 1 objects ['Time Stick'] with scores ['0.99']
# DEBUG:cd_fsod_test:Frame 3: Detected 1 objects ['Time Stick'] with scores ['0.99']
# 2025-05-12 20:47:11,541 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# -----Initializing new objects-----
# detections: [{'box': array([453, 224, 643, 473]), 'score': np.float64(0.9931594133377075), 'text': 'Time Stick'}]
# -----Initializing [{'box': array([453, 224, 643, 473]), 'score': np.float64(0.9931594133377075), 'text': 'Time Stick'}]-----
# Initialized object 1 (Time Stick)
# Created new object 1 (Time Stick) at frame 3
# 2025-05-12 20:47:11,733 - DEBUG - Detecting objects in frame 4
# DEBUG:cd_fsod_test:Detecting objects in frame 4
# 2025-05-12 20:47:11,733 - DEBUG - Frame 4: No objects detected
# DEBUG:cd_fsod_test:Frame 4: No objects detected
# 2025-05-12 20:47:11,733 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,737 - DEBUG - Detecting objects in frame 5
# DEBUG:cd_fsod_test:Detecting objects in frame 5
# 2025-05-12 20:47:11,737 - DEBUG - Frame 5: No objects detected
# DEBUG:cd_fsod_test:Frame 5: No objects detected
# 2025-05-12 20:47:11,737 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,740 - DEBUG - Detecting objects in frame 6
# DEBUG:cd_fsod_test:Detecting objects in frame 6
# 2025-05-12 20:47:11,740 - DEBUG - Frame 6: No objects detected
# DEBUG:cd_fsod_test:Frame 6: No objects detected
# 2025-05-12 20:47:11,740 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,743 - DEBUG - Detecting objects in frame 7
# DEBUG:cd_fsod_test:Detecting objects in frame 7
# 2025-05-12 20:47:11,743 - DEBUG - Frame 7: No objects detected
# DEBUG:cd_fsod_test:Frame 7: No objects detected
# 2025-05-12 20:47:11,743 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,746 - DEBUG - Detecting objects in frame 8
# DEBUG:cd_fsod_test:Detecting objects in frame 8
# 2025-05-12 20:47:11,746 - DEBUG - Frame 8: No objects detected
# DEBUG:cd_fsod_test:Frame 8: No objects detected
# 2025-05-12 20:47:11,746 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,749 - DEBUG - Detecting objects in frame 9
# DEBUG:cd_fsod_test:Detecting objects in frame 9
# 2025-05-12 20:47:11,749 - DEBUG - Frame 9: No objects detected
# DEBUG:cd_fsod_test:Frame 9: No objects detected
# 2025-05-12 20:47:11,749 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,752 - DEBUG - Detecting objects in frame 10
# DEBUG:cd_fsod_test:Detecting objects in frame 10
# 2025-05-12 20:47:11,752 - DEBUG - Frame 10: No objects detected
# DEBUG:cd_fsod_test:Frame 10: No objects detected
# 2025-05-12 20:47:11,752 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,755 - DEBUG - Detecting objects in frame 11
# DEBUG:cd_fsod_test:Detecting objects in frame 11
# 2025-05-12 20:47:11,755 - DEBUG - Frame 11: No objects detected
# DEBUG:cd_fsod_test:Frame 11: No objects detected
# 2025-05-12 20:47:11,755 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,758 - DEBUG - Detecting objects in frame 12
# DEBUG:cd_fsod_test:Detecting objects in frame 12
# 2025-05-12 20:47:11,759 - DEBUG - Frame 12: No objects detected
# DEBUG:cd_fsod_test:Frame 12: No objects detected
# 2025-05-12 20:47:11,759 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,761 - DEBUG - Detecting objects in frame 13
# DEBUG:cd_fsod_test:Detecting objects in frame 13
# 2025-05-12 20:47:11,762 - DEBUG - Frame 13: Detected 1 objects ['TVA Uniform'] with scores ['0.97']
# DEBUG:cd_fsod_test:Frame 13: Detected 1 objects ['TVA Uniform'] with scores ['0.97']
# 2025-05-12 20:47:11,762 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# Updated object 1 (TVA Uniform) with score 0.56 (IoU: 0.39, Emb: 0.74)
# 2025-05-12 20:47:11,803 - DEBUG - Detecting objects in frame 14
# DEBUG:cd_fsod_test:Detecting objects in frame 14
# 2025-05-12 20:47:11,803 - DEBUG - Frame 14: No objects detected
# DEBUG:cd_fsod_test:Frame 14: No objects detected
# 2025-05-12 20:47:11,803 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,806 - DEBUG - Detecting objects in frame 15
# DEBUG:cd_fsod_test:Detecting objects in frame 15
# 2025-05-12 20:47:11,806 - DEBUG - Frame 15: No objects detected
# DEBUG:cd_fsod_test:Frame 15: No objects detected
# 2025-05-12 20:47:11,807 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,809 - DEBUG - Detecting objects in frame 16
# DEBUG:cd_fsod_test:Detecting objects in frame 16
# 2025-05-12 20:47:11,809 - DEBUG - Frame 16: No objects detected
# DEBUG:cd_fsod_test:Frame 16: No objects detected
# 2025-05-12 20:47:11,810 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,813 - DEBUG - Detecting objects in frame 17
# DEBUG:cd_fsod_test:Detecting objects in frame 17
# 2025-05-12 20:47:11,813 - DEBUG - Frame 17: No objects detected
# DEBUG:cd_fsod_test:Frame 17: No objects detected
# 2025-05-12 20:47:11,813 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,816 - DEBUG - Detecting objects in frame 18
# DEBUG:cd_fsod_test:Detecting objects in frame 18
# 2025-05-12 20:47:11,816 - DEBUG - Frame 18: Detected 1 objects ["Sylvie's horned headpiece"] with scores ['0.99']
# DEBUG:cd_fsod_test:Frame 18: Detected 1 objects ["Sylvie's horned headpiece"] with scores ['0.99']
# 2025-05-12 20:47:11,816 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# Created new object 2 (Sylvie's horned headpiece)
# Created new object 2 (Sylvie's horned headpiece) at frame 18
# 2025-05-12 20:47:11,850 - DEBUG - Detecting objects in frame 19
# DEBUG:cd_fsod_test:Detecting objects in frame 19
# 2025-05-12 20:47:11,850 - DEBUG - Frame 19: No objects detected
# DEBUG:cd_fsod_test:Frame 19: No objects detected
# 2025-05-12 20:47:11,850 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,853 - DEBUG - Detecting objects in frame 20
# DEBUG:cd_fsod_test:Detecting objects in frame 20
# 2025-05-12 20:47:11,853 - DEBUG - Frame 20: No objects detected
# DEBUG:cd_fsod_test:Frame 20: No objects detected
# 2025-05-12 20:47:11,853 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,856 - DEBUG - Detecting objects in frame 21
# DEBUG:cd_fsod_test:Detecting objects in frame 21
# 2025-05-12 20:47:11,856 - DEBUG - Frame 21: No objects detected
# DEBUG:cd_fsod_test:Frame 21: No objects detected
# 2025-05-12 20:47:11,856 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,859 - DEBUG - Detecting objects in frame 22
# DEBUG:cd_fsod_test:Detecting objects in frame 22
# 2025-05-12 20:47:11,859 - DEBUG - Frame 22: No objects detected
# DEBUG:cd_fsod_test:Frame 22: No objects detected
# 2025-05-12 20:47:11,859 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,862 - DEBUG - Detecting objects in frame 23
# DEBUG:cd_fsod_test:Detecting objects in frame 23
# 2025-05-12 20:47:11,862 - DEBUG - Frame 23: No objects detected
# DEBUG:cd_fsod_test:Frame 23: No objects detected
# 2025-05-12 20:47:11,862 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,865 - DEBUG - Detecting objects in frame 24
# DEBUG:cd_fsod_test:Detecting objects in frame 24
# 2025-05-12 20:47:11,865 - DEBUG - Frame 24: No objects detected
# DEBUG:cd_fsod_test:Frame 24: No objects detected
# 2025-05-12 20:47:11,865 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,868 - DEBUG - Detecting objects in frame 25
# DEBUG:cd_fsod_test:Detecting objects in frame 25
# 2025-05-12 20:47:11,868 - DEBUG - Frame 25: No objects detected
# DEBUG:cd_fsod_test:Frame 25: No objects detected
# 2025-05-12 20:47:11,868 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,871 - DEBUG - Detecting objects in frame 26
# DEBUG:cd_fsod_test:Detecting objects in frame 26
# 2025-05-12 20:47:11,871 - DEBUG - Frame 26: No objects detected
# DEBUG:cd_fsod_test:Frame 26: No objects detected
# 2025-05-12 20:47:11,871 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,874 - DEBUG - Detecting objects in frame 27
# DEBUG:cd_fsod_test:Detecting objects in frame 27
# 2025-05-12 20:47:11,874 - DEBUG - Frame 27: No objects detected
# DEBUG:cd_fsod_test:Frame 27: No objects detected
# 2025-05-12 20:47:11,874 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,877 - DEBUG - Detecting objects in frame 28
# DEBUG:cd_fsod_test:Detecting objects in frame 28
# 2025-05-12 20:47:11,878 - DEBUG - Frame 28: No objects detected
# DEBUG:cd_fsod_test:Frame 28: No objects detected
# 2025-05-12 20:47:11,878 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,880 - DEBUG - Detecting objects in frame 29
# DEBUG:cd_fsod_test:Detecting objects in frame 29
# 2025-05-12 20:47:11,880 - DEBUG - Frame 29: No objects detected
# DEBUG:cd_fsod_test:Frame 29: No objects detected
# 2025-05-12 20:47:11,880 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,883 - DEBUG - Detecting objects in frame 30
# DEBUG:cd_fsod_test:Detecting objects in frame 30
# 2025-05-12 20:47:11,883 - DEBUG - Frame 30: No objects detected
# DEBUG:cd_fsod_test:Frame 30: No objects detected
# 2025-05-12 20:47:11,883 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,886 - DEBUG - Detecting objects in frame 31
# DEBUG:cd_fsod_test:Detecting objects in frame 31
# 2025-05-12 20:47:11,886 - DEBUG - Frame 31: No objects detected
# DEBUG:cd_fsod_test:Frame 31: No objects detected
# 2025-05-12 20:47:11,886 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,889 - DEBUG - Detecting objects in frame 32
# DEBUG:cd_fsod_test:Detecting objects in frame 32
# 2025-05-12 20:47:11,890 - DEBUG - Frame 32: No objects detected
# DEBUG:cd_fsod_test:Frame 32: No objects detected
# 2025-05-12 20:47:11,890 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,893 - DEBUG - Detecting objects in frame 33
# DEBUG:cd_fsod_test:Detecting objects in frame 33
# 2025-05-12 20:47:11,893 - DEBUG - Frame 33: Detected 1 objects ["Sylvie's horned headpiece"] with scores ['0.96']
# DEBUG:cd_fsod_test:Frame 33: Detected 1 objects ["Sylvie's horned headpiece"] with scores ['0.96']
# 2025-05-12 20:47:11,893 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# Created new object 3 (Sylvie's horned headpiece)
# Created new object 3 (Sylvie's horned headpiece) at frame 33
# 2025-05-12 20:47:11,927 - DEBUG - Detecting objects in frame 34
# DEBUG:cd_fsod_test:Detecting objects in frame 34
# 2025-05-12 20:47:11,928 - DEBUG - Frame 34: No objects detected
# DEBUG:cd_fsod_test:Frame 34: No objects detected
# 2025-05-12 20:47:11,928 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,931 - DEBUG - Detecting objects in frame 35
# DEBUG:cd_fsod_test:Detecting objects in frame 35
# 2025-05-12 20:47:11,931 - DEBUG - Frame 35: No objects detected
# DEBUG:cd_fsod_test:Frame 35: No objects detected
# 2025-05-12 20:47:11,931 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,934 - DEBUG - Detecting objects in frame 36
# DEBUG:cd_fsod_test:Detecting objects in frame 36
# 2025-05-12 20:47:11,935 - DEBUG - Frame 36: No objects detected
# DEBUG:cd_fsod_test:Frame 36: No objects detected
# 2025-05-12 20:47:11,935 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,938 - DEBUG - Detecting objects in frame 37
# DEBUG:cd_fsod_test:Detecting objects in frame 37
# 2025-05-12 20:47:11,938 - DEBUG - Frame 37: No objects detected
# DEBUG:cd_fsod_test:Frame 37: No objects detected
# 2025-05-12 20:47:11,938 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,941 - DEBUG - Detecting objects in frame 38
# DEBUG:cd_fsod_test:Detecting objects in frame 38
# 2025-05-12 20:47:11,941 - DEBUG - Frame 38: No objects detected
# DEBUG:cd_fsod_test:Frame 38: No objects detected
# 2025-05-12 20:47:11,941 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,944 - DEBUG - Detecting objects in frame 39
# DEBUG:cd_fsod_test:Detecting objects in frame 39
# 2025-05-12 20:47:11,944 - DEBUG - Frame 39: No objects detected
# DEBUG:cd_fsod_test:Frame 39: No objects detected
# 2025-05-12 20:47:11,944 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,947 - DEBUG - Detecting objects in frame 40
# DEBUG:cd_fsod_test:Detecting objects in frame 40
# 2025-05-12 20:47:11,947 - DEBUG - Frame 40: No objects detected
# DEBUG:cd_fsod_test:Frame 40: No objects detected
# 2025-05-12 20:47:11,947 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,950 - DEBUG - Detecting objects in frame 41
# DEBUG:cd_fsod_test:Detecting objects in frame 41
# 2025-05-12 20:47:11,950 - DEBUG - Frame 41: No objects detected
# DEBUG:cd_fsod_test:Frame 41: No objects detected
# 2025-05-12 20:47:11,951 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,954 - DEBUG - Detecting objects in frame 42
# DEBUG:cd_fsod_test:Detecting objects in frame 42
# 2025-05-12 20:47:11,954 - DEBUG - Frame 42: No objects detected
# DEBUG:cd_fsod_test:Frame 42: No objects detected
# 2025-05-12 20:47:11,954 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,957 - DEBUG - Detecting objects in frame 43
# DEBUG:cd_fsod_test:Detecting objects in frame 43
# 2025-05-12 20:47:11,957 - DEBUG - Frame 43: Detected 1 objects ["Sylvie's horned headpiece"] with scores ['0.91']
# DEBUG:cd_fsod_test:Frame 43: Detected 1 objects ["Sylvie's horned headpiece"] with scores ['0.91']
# 2025-05-12 20:47:11,957 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# Updated object 2 (Sylvie's horned headpiece) with score 0.41 (IoU: 0.10, Emb: 0.71)
# 2025-05-12 20:47:11,992 - DEBUG - Detecting objects in frame 44
# DEBUG:cd_fsod_test:Detecting objects in frame 44
# 2025-05-12 20:47:11,992 - DEBUG - Frame 44: No objects detected
# DEBUG:cd_fsod_test:Frame 44: No objects detected
# 2025-05-12 20:47:11,992 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,996 - DEBUG - Detecting objects in frame 45
# DEBUG:cd_fsod_test:Detecting objects in frame 45
# 2025-05-12 20:47:11,996 - DEBUG - Frame 45: No objects detected
# DEBUG:cd_fsod_test:Frame 45: No objects detected
# 2025-05-12 20:47:11,996 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:11,999 - DEBUG - Detecting objects in frame 46
# DEBUG:cd_fsod_test:Detecting objects in frame 46
# 2025-05-12 20:47:11,999 - DEBUG - Frame 46: No objects detected
# DEBUG:cd_fsod_test:Frame 46: No objects detected
# 2025-05-12 20:47:11,999 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:12,002 - DEBUG - Detecting objects in frame 47
# DEBUG:cd_fsod_test:Detecting objects in frame 47
# 2025-05-12 20:47:12,002 - DEBUG - Frame 47: No objects detected
# DEBUG:cd_fsod_test:Frame 47: No objects detected
# 2025-05-12 20:47:12,002 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:12,006 - DEBUG - Detecting objects in frame 48
# DEBUG:cd_fsod_test:Detecting objects in frame 48
# 2025-05-12 20:47:12,006 - DEBUG - Frame 48: No objects detected
# DEBUG:cd_fsod_test:Frame 48: No objects detected
# 2025-05-12 20:47:12,006 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:12,009 - DEBUG - Detecting objects in frame 49
# DEBUG:cd_fsod_test:Detecting objects in frame 49
# 2025-05-12 20:47:12,009 - DEBUG - Frame 49: No objects detected
# DEBUG:cd_fsod_test:Frame 49: No objects detected
# 2025-05-12 20:47:12,009 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:12,012 - DEBUG - Detecting objects in frame 50
# DEBUG:cd_fsod_test:Detecting objects in frame 50
# 2025-05-12 20:47:12,012 - DEBUG - Frame 50: No objects detected
# DEBUG:cd_fsod_test:Frame 50: No objects detected
# 2025-05-12 20:47:12,013 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:12,015 - DEBUG - Detecting objects in frame 51
# DEBUG:cd_fsod_test:Detecting objects in frame 51
# 2025-05-12 20:47:12,015 - DEBUG - Frame 51: No objects detected
# DEBUG:cd_fsod_test:Frame 51: No objects detected
# 2025-05-12 20:47:12,016 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:12,018 - DEBUG - Detecting objects in frame 52
# DEBUG:cd_fsod_test:Detecting objects in frame 52
# 2025-05-12 20:47:12,019 - DEBUG - Frame 52: No objects detected
# DEBUG:cd_fsod_test:Frame 52: No objects detected
# 2025-05-12 20:47:12,019 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-12 20:47:12,021 - DEBUG - Detecting objects in frame 53
# DEBUG:cd_fsod_test:Detecting objects in frame 53
# 2025-05-12 20:47:12,022 - DEBUG - Frame 53: No objects detected
# DEBUG:cd_fsod_test:Frame 53: No objects detected
# 2025-05-12 20:47:12,022 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# Found 3 unique objects to process with SAM2

# ==== Processing object 1 (Time Stick) separately ====
# Resetting SAM2 state for object 1...
# frame loading (JPEG): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 54/54 [00:01<00:00, 37.49it/s]
# Set video from directory: /home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/data/frames/Scenes 061-080__265H-2-_20230815215828529
# Adding box for object 1 at frame 3
# /home/ubuntu/code/drew/sam2/sam2/sam2_video_predictor.py:786: UserWarning: cannot import name '_C' from 'sam2' (/home/ubuntu/code/drew/sam2/sam2/__init__.py)

# Skipping the post-processing step due to the error above. You can still use SAM 2 and it's OK to ignore the error above, although some post-processing functionality may be limited (which doesn't affect the results in most cases; see https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).
#   pred_masks_gpu = fill_holes_in_mask_scores(
# Running propagation for object 1...
# propagate in video:   0%|                                                                                                                                         | 0/51 [00:00<?, ?it/s]Processed frame 3, found 1 objects
# Processed frame 4, found 1 objects
# propagate in video:   4%|█████                                                                                                                            | 2/51 [00:00<00:15,  3.12it/s]Processed frame 5, found 1 objects
# propagate in video:   6%|███████▌                                                                                                                         | 3/51 [00:01<00:22,  2.13it/s]Processed frame 6, found 1 objects
# propagate in video:   8%|██████████                                                                                                                       | 4/51 [00:02<00:26,  1.80it/s]Processed frame 7, found 1 objects
# propagate in video:  10%|████████████▋                                                                                                                    | 5/51 [00:02<00:28,  1.62it/s]Processed frame 8, found 1 objects
# propagate in video:  12%|███████████████▏                                                                                                                 | 6/51 [00:03<00:29,  1.51it/s]Processed frame 9, found 1 objects
# propagate in video:  14%|█████████████████▋                                                                                                               | 7/51 [00:04<00:30,  1.43it/s]Processed frame 10, found 1 objects
# propagate in video:  16%|████████████████████▏                                                                                                            | 8/51 [00:05<00:31,  1.36it/s]Processed frame 11, found 1 objects
# propagate in video:  18%|██████████████████████▊                                                                                                          | 9/51 [00:05<00:31,  1.32it/s]Processed frame 12, found 1 objects
# propagate in video:  20%|█████████████████████████                                                                                                       | 10/51 [00:06<00:31,  1.29it/s]Processed frame 13, found 1 objects
# propagate in video:  22%|███████████████████████████▌                                                                                                    | 11/51 [00:07<00:31,  1.27it/s]Processed frame 14, found 1 objects
# propagate in video:  24%|██████████████████████████████                                                                                                  | 12/51 [00:08<00:30,  1.26it/s]Processed frame 15, found 1 objects
# propagate in video:  25%|████████████████████████████████▋                                                                                               | 13/51 [00:09<00:30,  1.25it/s]Processed frame 16, found 1 objects
# propagate in video:  27%|███████████████████████████████████▏                                                                                            | 14/51 [00:09<00:29,  1.25it/s]Processed frame 17, found 1 objects
# propagate in video:  29%|█████████████████████████████████████▋                                                                                          | 15/51 [00:10<00:28,  1.24it/s]Processed frame 18, found 1 objects
# propagate in video:  31%|████████████████████████████████████████▏                                                                                       | 16/51 [00:11<00:28,  1.24it/s]Processed frame 19, found 1 objects
# propagate in video:  33%|██████████████████████████████████████████▋                                                                                     | 17/51 [00:12<00:27,  1.23it/s]Processed frame 20, found 1 objects
# propagate in video:  35%|█████████████████████████████████████████████▏                                                                                  | 18/51 [00:13<00:26,  1.23it/s]Processed frame 21, found 1 objects
# propagate in video:  37%|███████████████████████████████████████████████▋                                                                                | 19/51 [00:14<00:25,  1.23it/s]Processed frame 22, found 1 objects
# propagate in video:  39%|██████████████████████████████████████████████████▏                                                                             | 20/51 [00:14<00:25,  1.23it/s]Processed frame 23, found 1 objects
# propagate in video:  41%|████████████████████████████████████████████████████▋                                                                           | 21/51 [00:15<00:24,  1.23it/s]Processed frame 24, found 1 objects
# propagate in video:  43%|███████████████████████████████████████████████████████▏                                                                        | 22/51 [00:16<00:23,  1.23it/s]Processed frame 25, found 1 objects
# propagate in video:  45%|█████████████████████████████████████████████████████████▋                                                                      | 23/51 [00:17<00:22,  1.23it/s]Processed frame 26, found 1 objects
# propagate in video:  47%|████████████████████████████████████████████████████████████▏                                                                   | 24/51 [00:18<00:22,  1.23it/s]Processed frame 27, found 1 objects
# propagate in video:  49%|██████████████████████████████████████████████████████████████▋                                                                 | 25/51 [00:18<00:21,  1.23it/s]Processed frame 28, found 1 objects
# propagate in video:  51%|█████████████████████████████████████████████████████████████████▎                                                              | 26/51 [00:19<00:20,  1.23it/s]Processed frame 29, found 1 objects
# propagate in video:  53%|███████████████████████████████████████████████████████████████████▊                                                            | 27/51 [00:20<00:19,  1.23it/s]Processed frame 30, found 1 objects
# propagate in video:  55%|██████████████████████████████████████████████████████████████████████▎                                                         | 28/51 [00:21<00:18,  1.23it/s]Processed frame 31, found 1 objects
# propagate in video:  57%|████████████████████████████████████████████████████████████████████████▊                                                       | 29/51 [00:22<00:17,  1.23it/s]Processed frame 32, found 1 objects
# propagate in video:  59%|███████████████████████████████████████████████████████████████████████████▎                                                    | 30/51 [00:23<00:17,  1.22it/s]Processed frame 33, found 1 objects
# propagate in video:  61%|█████████████████████████████████████████████████████████████████████████████▊                                                  | 31/51 [00:23<00:16,  1.22it/s]Processed frame 34, found 1 objects
# propagate in video:  63%|████████████████████████████████████████████████████████████████████████████████▎                                               | 32/51 [00:24<00:15,  1.22it/s]Processed frame 35, found 1 objects
# propagate in video:  65%|██████████████████████████████████████████████████████████████████████████████████▊                                             | 33/51 [00:25<00:14,  1.22it/s]Processed frame 36, found 1 objects
# propagate in video:  67%|█████████████████████████████████████████████████████████████████████████████████████▎                                          | 34/51 [00:26<00:13,  1.22it/s]Processed frame 37, found 1 objects
# propagate in video:  69%|███████████████████████████████████████████████████████████████████████████████████████▊                                        | 35/51 [00:27<00:13,  1.22it/s]Processed frame 38, found 1 objects
# propagate in video:  71%|██████████████████████████████████████████████████████████████████████████████████████████▎                                     | 36/51 [00:27<00:12,  1.23it/s]Processed frame 39, found 1 objects
# propagate in video:  73%|████████████████████████████████████████████████████████████████████████████████████████████▊                                   | 37/51 [00:28<00:11,  1.22it/s]Processed frame 40, found 1 objects
# propagate in video:  75%|███████████████████████████████████████████████████████████████████████████████████████████████▎                                | 38/51 [00:29<00:10,  1.22it/s]Processed frame 41, found 1 objects
# propagate in video:  76%|█████████████████████████████████████████████████████████████████████████████████████████████████▉                              | 39/51 [00:30<00:09,  1.22it/s]Processed frame 42, found 1 objects
# propagate in video:  78%|████████████████████████████████████████████████████████████████████████████████████████████████████▍                           | 40/51 [00:31<00:08,  1.22it/s]Processed frame 43, found 1 objects
# propagate in video:  80%|██████████████████████████████████████████████████████████████████████████████████████████████████████▉                         | 41/51 [00:31<00:08,  1.22it/s]Processed frame 44, found 1 objects
# propagate in video:  82%|█████████████████████████████████████████████████████████████████████████████████████████████████████████▍                      | 42/51 [00:32<00:07,  1.22it/s]Processed frame 45, found 1 objects
# propagate in video:  84%|███████████████████████████████████████████████████████████████████████████████████████████████████████████▉                    | 43/51 [00:33<00:06,  1.22it/s]Processed frame 46, found 1 objects
# propagate in video:  86%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                 | 44/51 [00:34<00:05,  1.22it/s]Processed frame 47, found 1 objects
# propagate in video:  88%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉               | 45/51 [00:35<00:04,  1.22it/s]Processed frame 48, found 1 objects
# propagate in video:  90%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍            | 46/51 [00:36<00:04,  1.22it/s]Processed frame 49, found 1 objects
# propagate in video:  92%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉          | 47/51 [00:36<00:03,  1.22it/s]Processed frame 50, found 1 objects
# propagate in video:  94%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍       | 48/51 [00:37<00:02,  1.22it/s]Processed frame 51, found 1 objects
# propagate in video:  96%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉     | 49/51 [00:38<00:01,  1.22it/s]Processed frame 52, found 1 objects
# propagate in video:  98%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍  | 50/51 [00:39<00:00,  1.22it/s]Processed frame 53, found 1 objects
# propagate in video: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 51/51 [00:40<00:00,  1.27it/s]
# Successfully propagated masks for object 1

# ==== Processing object 2 (Sylvie's horned headpiece) separately ====
# Resetting SAM2 state for object 2...
# frame loading (JPEG): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 54/54 [00:01<00:00, 38.64it/s]
# Set video from directory: /home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/data/frames/Scenes 061-080__265H-2-_20230815215828529
# Adding box for object 2 at frame 18
# Running propagation for object 2...
# propagate in video:   0%|                                                                                                                                         | 0/36 [00:00<?, ?it/s]Processed frame 18, found 1 objects
# Processed frame 19, found 1 objects
# propagate in video:   6%|███████▏                                                                                                                         | 2/36 [00:00<00:11,  3.04it/s]Processed frame 20, found 1 objects
# propagate in video:   8%|██████████▊                                                                                                                      | 3/36 [00:01<00:15,  2.10it/s]Processed frame 21, found 1 objects
# propagate in video:  11%|██████████████▎                                                                                                                  | 4/36 [00:02<00:18,  1.78it/s]Processed frame 22, found 1 objects
# propagate in video:  14%|█████████████████▉                                                                                                               | 5/36 [00:02<00:19,  1.60it/s]Processed frame 23, found 1 objects
# propagate in video:  17%|█████████████████████▌                                                                                                           | 6/36 [00:03<00:20,  1.49it/s]Processed frame 24, found 1 objects
# propagate in video:  19%|█████████████████████████                                                                                                        | 7/36 [00:04<00:20,  1.41it/s]Processed frame 25, found 1 objects
# propagate in video:  22%|████████████████████████████▋                                                                                                    | 8/36 [00:05<00:20,  1.34it/s]Processed frame 26, found 1 objects
# propagate in video:  25%|████████████████████████████████▎                                                                                                | 9/36 [00:05<00:20,  1.30it/s]Processed frame 27, found 1 objects
# propagate in video:  28%|███████████████████████████████████▌                                                                                            | 10/36 [00:06<00:20,  1.28it/s]Processed frame 28, found 1 objects
# propagate in video:  31%|███████████████████████████████████████                                                                                         | 11/36 [00:07<00:19,  1.26it/s]Processed frame 29, found 1 objects
# propagate in video:  33%|██████████████████████████████████████████▋                                                                                     | 12/36 [00:08<00:19,  1.24it/s]Processed frame 30, found 1 objects
# propagate in video:  36%|██████████████████████████████████████████████▏                                                                                 | 13/36 [00:09<00:18,  1.24it/s]Processed frame 31, found 1 objects
# propagate in video:  39%|█████████████████████████████████████████████████▊                                                                              | 14/36 [00:10<00:17,  1.23it/s]Processed frame 32, found 1 objects
# propagate in video:  42%|█████████████████████████████████████████████████████▎                                                                          | 15/36 [00:10<00:17,  1.22it/s]Processed frame 33, found 1 objects
# propagate in video:  44%|████████████████████████████████████████████████████████▉                                                                       | 16/36 [00:11<00:16,  1.22it/s]Processed frame 34, found 1 objects
# propagate in video:  47%|████████████████████████████████████████████████████████████▍                                                                   | 17/36 [00:12<00:15,  1.22it/s]Processed frame 35, found 1 objects
# propagate in video:  50%|████████████████████████████████████████████████████████████████                                                                | 18/36 [00:13<00:14,  1.22it/s]Processed frame 36, found 1 objects
# propagate in video:  53%|███████████████████████████████████████████████████████████████████▌                                                            | 19/36 [00:14<00:13,  1.22it/s]Processed frame 37, found 1 objects
# propagate in video:  56%|███████████████████████████████████████████████████████████████████████                                                         | 20/36 [00:15<00:13,  1.21it/s]Processed frame 38, found 1 objects
# propagate in video:  58%|██████████████████████████████████████████████████████████████████████████▋                                                     | 21/36 [00:15<00:12,  1.21it/s]Processed frame 39, found 1 objects
# propagate in video:  61%|██████████████████████████████████████████████████████████████████████████████▏                                                 | 22/36 [00:16<00:11,  1.21it/s]Processed frame 40, found 1 objects
# propagate in video:  64%|█████████████████████████████████████████████████████████████████████████████████▊                                              | 23/36 [00:17<00:10,  1.21it/s]Processed frame 41, found 1 objects
# propagate in video:  67%|█████████████████████████████████████████████████████████████████████████████████████▎                                          | 24/36 [00:18<00:09,  1.21it/s]Processed frame 42, found 1 objects
# propagate in video:  69%|████████████████████████████████████████████████████████████████████████████████████████▉                                       | 25/36 [00:19<00:09,  1.21it/s]Processed frame 43, found 1 objects
# propagate in video:  72%|████████████████████████████████████████████████████████████████████████████████████████████▍                                   | 26/36 [00:19<00:08,  1.21it/s]Processed frame 44, found 1 objects
# propagate in video:  75%|████████████████████████████████████████████████████████████████████████████████████████████████                                | 27/36 [00:20<00:07,  1.21it/s]Processed frame 45, found 1 objects
# propagate in video:  78%|███████████████████████████████████████████████████████████████████████████████████████████████████▌                            | 28/36 [00:21<00:06,  1.21it/s]Processed frame 46, found 1 objects
# propagate in video:  81%|███████████████████████████████████████████████████████████████████████████████████████████████████████                         | 29/36 [00:22<00:05,  1.21it/s]Processed frame 47, found 1 objects
# propagate in video:  83%|██████████████████████████████████████████████████████████████████████████████████████████████████████████▋                     | 30/36 [00:23<00:04,  1.21it/s]Processed frame 48, found 1 objects
# propagate in video:  86%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                 | 31/36 [00:24<00:04,  1.21it/s]Processed frame 49, found 1 objects
# propagate in video:  89%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊              | 32/36 [00:24<00:03,  1.21it/s]Processed frame 50, found 1 objects
# propagate in video:  92%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎          | 33/36 [00:25<00:02,  1.21it/s]Processed frame 51, found 1 objects
# propagate in video:  94%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉       | 34/36 [00:26<00:01,  1.21it/s]Processed frame 52, found 1 objects
# propagate in video:  97%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍   | 35/36 [00:27<00:00,  1.21it/s]Processed frame 53, found 1 objects
# propagate in video: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 36/36 [00:28<00:00,  1.27it/s]
# Successfully propagated masks for object 2

# ==== Processing object 3 (Sylvie's horned headpiece) separately ====
# Resetting SAM2 state for object 3...
# frame loading (JPEG): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 54/54 [00:01<00:00, 38.47it/s]
# Set video from directory: /home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/data/frames/Scenes 061-080__265H-2-_20230815215828529
# Adding box for object 3 at frame 33
# Running propagation for object 3...
# propagate in video:   0%|                                                                                                                                         | 0/21 [00:00<?, ?it/s]Processed frame 33, found 1 objects
# Processed frame 34, found 1 objects
# propagate in video:  10%|████████████▎                                                                                                                    | 2/21 [00:00<00:06,  3.02it/s]Processed frame 35, found 1 objects
# propagate in video:  14%|██████████████████▍                                                                                                              | 3/21 [00:01<00:08,  2.09it/s]Processed frame 36, found 1 objects
# propagate in video:  19%|████████████████████████▌                                                                                                        | 4/21 [00:02<00:09,  1.76it/s]Processed frame 37, found 1 objects
# propagate in video:  24%|██████████████████████████████▋                                                                                                  | 5/21 [00:02<00:10,  1.59it/s]Processed frame 38, found 1 objects
# propagate in video:  29%|████████████████████████████████████▊                                                                                            | 6/21 [00:03<00:10,  1.48it/s]Processed frame 39, found 1 objects
# propagate in video:  33%|███████████████████████████████████████████                                                                                      | 7/21 [00:04<00:10,  1.40it/s]Processed frame 40, found 1 objects
# propagate in video:  38%|█████████████████████████████████████████████████▏                                                                               | 8/21 [00:05<00:09,  1.33it/s]Processed frame 41, found 1 objects
# propagate in video:  43%|███████████████████████████████████████████████████████▎                                                                         | 9/21 [00:06<00:09,  1.29it/s]Processed frame 42, found 1 objects
# propagate in video:  48%|████████████████████████████████████████████████████████████▉                                                                   | 10/21 [00:06<00:08,  1.27it/s]Processed frame 43, found 1 objects
# propagate in video:  52%|███████████████████████████████████████████████████████████████████                                                             | 11/21 [00:07<00:08,  1.25it/s]Processed frame 44, found 1 objects
# propagate in video:  57%|█████████████████████████████████████████████████████████████████████████▏                                                      | 12/21 [00:08<00:07,  1.24it/s]Processed frame 45, found 1 objects
# propagate in video:  62%|███████████████████████████████████████████████████████████████████████████████▏                                                | 13/21 [00:09<00:06,  1.23it/s]Processed frame 46, found 1 objects
# propagate in video:  67%|█████████████████████████████████████████████████████████████████████████████████████▎                                          | 14/21 [00:10<00:05,  1.22it/s]Processed frame 47, found 1 objects
# propagate in video:  71%|███████████████████████████████████████████████████████████████████████████████████████████▍                                    | 15/21 [00:10<00:04,  1.22it/s]Processed frame 48, found 1 objects
# propagate in video:  76%|█████████████████████████████████████████████████████████████████████████████████████████████████▌                              | 16/21 [00:11<00:04,  1.21it/s]Processed frame 49, found 1 objects
# propagate in video:  81%|███████████████████████████████████████████████████████████████████████████████████████████████████████▌                        | 17/21 [00:12<00:03,  1.21it/s]Processed frame 50, found 1 objects
# propagate in video:  86%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                  | 18/21 [00:13<00:02,  1.21it/s]Processed frame 51, found 1 objects
# propagate in video:  90%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊            | 19/21 [00:14<00:01,  1.21it/s]Processed frame 52, found 1 objects
# propagate in video:  95%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉      | 20/21 [00:15<00:00,  1.21it/s]Processed frame 53, found 1 objects
# propagate in video: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21/21 [00:15<00:00,  1.32it/s]
# Successfully propagated masks for object 3
# Saving per-object segmentation visualizations...
# Processing object 1 visible from frame 3 to 13
#   Saved 11 visualizations for object #1 (Time Stick)
# Processing object 2 visible from frame 18 to 43
#   Saved 26 visualizations for object #2 (Sylvie's horned headpiece)
# Processing object 3 visible from frame 33 to 33
#   Saved 1 visualizations for object #3 (Sylvie's horned headpiece)
# All per-object mask visualizations saved to cd_fsod_results/object_masks
# Saving first detection frames for each object...
# Using detection files from: /home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/data/detections/Scenes 061-080__265H-2-_20230815215828529
#   Object #1 (Time Stick) first detected in frame 3, using 3.json
#     JSON file contains 51 detections
#     Sample detection for Time Stick: confidence=0.9932, coordinates=[453, 224, 643, 473]
#   Saved first detection frame for object #1 (Time Stick)
#   Object #2 (Sylvie's horned headpiece) first detected in frame 18, using 18.json
#     JSON file contains 51 detections
#     Sample detection for Sylvie's horned headpiece: confidence=0.9856, coordinates=[322, 128, 481, 202]
#   Saved first detection frame for object #2 (Sylvie's horned headpiece)
#   Object #3 (Sylvie's horned headpiece) first detected in frame 33, using 33.json
#     JSON file contains 42 detections
#     Sample detection for Sylvie's horned headpiece: confidence=0.9580, coordinates=[582, 104, 671, 201]
#   Saved first detection frame for object #3 (Sylvie's horned headpiece)
# All first detection frames saved to cd_fsod_results/first_detections
# Creating object-to-frame mapping...
# Verifying consistency with saved mask images...
# Verification successful: All objects' masks are consistent with the mapping
# Saved object-to-frame mapping to cd_fsod_results/object_frame_mapping.json
# All results saved to: cd_fsod_results
# 2025-05-12 20:48:46,270 - INFO - Video processing completed in 94.75 seconds
# INFO:cd_fsod_test:Video processing completed in 94.75 seconds
# 2025-05-12 20:48:46,270 - DEBUG - First detection details:
# DEBUG:cd_fsod_test:First detection details:
# 2025-05-12 20:48:46,270 - DEBUG -   Object #1 (Time Stick) first detected at frame 3
# DEBUG:cd_fsod_test:  Object #1 (Time Stick) first detected at frame 3
# 2025-05-12 20:48:46,271 - DEBUG -     From 3.json: {'coordinates': [453, 224, 643, 473], 'label': 'Time Stick', 'confidence': 0.9931594133377075}
# DEBUG:cd_fsod_test:    From 3.json: {'coordinates': [453, 224, 643, 473], 'label': 'Time Stick', 'confidence': 0.9931594133377075}
# 2025-05-12 20:48:46,271 - DEBUG -   Object #2 (Sylvie's horned headpiece) first detected at frame 18
# DEBUG:cd_fsod_test:  Object #2 (Sylvie's horned headpiece) first detected at frame 18
# 2025-05-12 20:48:46,271 - DEBUG -     From 18.json: {'coordinates': [322, 128, 481, 202], 'label': "Sylvie's horned headpiece", 'confidence': 0.9855985045433044}
# DEBUG:cd_fsod_test:    From 18.json: {'coordinates': [322, 128, 481, 202], 'label': "Sylvie's horned headpiece", 'confidence': 0.9855985045433044}
# 2025-05-12 20:48:46,271 - DEBUG -   Object #3 (Sylvie's horned headpiece) first detected at frame 33
# DEBUG:cd_fsod_test:  Object #3 (Sylvie's horned headpiece) first detected at frame 33
# 2025-05-12 20:48:46,271 - DEBUG -     From 33.json: {'coordinates': [582, 104, 671, 201], 'label': "Sylvie's horned headpiece", 'confidence': 0.9580428600311279}
# DEBUG:cd_fsod_test:    From 33.json: {'coordinates': [582, 104, 671, 201], 'label': "Sylvie's horned headpiece", 'confidence': 0.9580428600311279}
# 2025-05-12 20:48:46,272 - INFO - Generated 59 output files
# INFO:cd_fsod_test:Generated 59 output files
# 2025-05-12 20:48:46,272 - DEBUG - Output files: ['frame_0049.jpg', 'frame_0045.jpg', 'frame_0014.jpg', 'frame_0025.jpg', 'frame_0051.jpg', 'frame_0039.jpg', 'frame_0030.jpg', 'frame_0008.jpg', 'frame_0002.jpg', 'frame_0042.jpg']
# DEBUG:cd_fsod_test:Output files: ['frame_0049.jpg', 'frame_0045.jpg', 'frame_0014.jpg', 'frame_0025.jpg', 'frame_0051.jpg', 'frame_0039.jpg', 'frame_0030.jpg', 'frame_0008.jpg', 'frame_0002.jpg', 'frame_0042.jpg']
# 2025-05-12 20:48:46,272 - INFO - ================================================================================
# INFO:cd_fsod_test:================================================================================
# 2025-05-12 20:48:46,272 - INFO - CD-FSOD Integration Test Completed Successfully
# INFO:cd_fsod_test:CD-FSOD Integration Test Completed Successfully
# 2025-05-12 20:48:46,272 - INFO - Total processing time: 98.52 seconds
# INFO:cd_fsod_test:Total processing time: 98.52 seconds
# 2025-05-12 20:48:46,272 - INFO - Results saved to: ./cd_fsod_results
# INFO:cd_fsod_test:Results saved to: ./cd_fsod_results
# 2025-05-12 20:48:46,272 - INFO - ================================================================================
# INFO:cd_fsod_test:================================================================================
