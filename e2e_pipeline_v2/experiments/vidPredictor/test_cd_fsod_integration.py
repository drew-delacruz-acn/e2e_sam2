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
    
    # Mask quality options
    parser.add_argument("--mask-quality-threshold", type=int, default=0, help="Minimum pixel count for high-quality masks (default: 0)")
    
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
    logger.info(f"  - Mask quality threshold: {args.mask_quality_threshold}")
    
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
            min_gap_frames=args.min_gap_frames,
            mask_quality_threshold=args.mask_quality_threshold
        )
        logger.info("Pipeline initialized successfully")
        logger.info(f"Pipeline initialization took {time.time() - start_time:.2f} seconds")
        
        # Log mask quality threshold approach
        logger.info("=" * 50)
        logger.info(f"Using mask quality-based approach with threshold: {args.mask_quality_threshold} pixels")
        logger.info("This will save all frames where mask pixel count > 0, with quality indicators")
        logger.info("=" * 50)
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
# 2025-05-13 20:17:21,768 - DEBUG - Detecting objects in frame 0
# DEBUG:cd_fsod_test:Detecting objects in frame 0
# 2025-05-13 20:17:21,768 - DEBUG - Frame 0: No objects detected
# DEBUG:cd_fsod_test:Frame 0: No objects detected
# 2025-05-13 20:17:21,768 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# -----Initializing new objects-----
# detections: []
# 2025-05-13 20:17:21,772 - DEBUG - Detecting objects in frame 1
# DEBUG:cd_fsod_test:Detecting objects in frame 1
# 2025-05-13 20:17:21,772 - DEBUG - Frame 1: No objects detected
# DEBUG:cd_fsod_test:Frame 1: No objects detected
# 2025-05-13 20:17:21,772 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# -----Initializing new objects-----
# detections: []
# 2025-05-13 20:17:21,775 - DEBUG - Detecting objects in frame 2
# DEBUG:cd_fsod_test:Detecting objects in frame 2
# 2025-05-13 20:17:21,775 - DEBUG - Frame 2: No objects detected
# DEBUG:cd_fsod_test:Frame 2: No objects detected
# 2025-05-13 20:17:21,775 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# -----Initializing new objects-----
# detections: []
# 2025-05-13 20:17:21,778 - DEBUG - Detecting objects in frame 3
# DEBUG:cd_fsod_test:Detecting objects in frame 3
# 2025-05-13 20:17:21,778 - DEBUG - Frame 3: Detected 1 objects ['Time Stick'] with scores ['0.99']
# DEBUG:cd_fsod_test:Frame 3: Detected 1 objects ['Time Stick'] with scores ['0.99']
# 2025-05-13 20:17:21,778 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# -----Initializing new objects-----
# detections: [{'box': array([453, 224, 643, 473]), 'score': np.float64(0.9931594133377075), 'text': 'Time Stick'}]
# -----Initializing [{'box': array([453, 224, 643, 473]), 'score': np.float64(0.9931594133377075), 'text': 'Time Stick'}]-----
# Initialized object 1 (Time Stick)
# Created new object 1 (Time Stick) at frame 3
# 2025-05-13 20:17:21,965 - DEBUG - Detecting objects in frame 4
# DEBUG:cd_fsod_test:Detecting objects in frame 4
# 2025-05-13 20:17:21,965 - DEBUG - Frame 4: No objects detected
# DEBUG:cd_fsod_test:Frame 4: No objects detected
# 2025-05-13 20:17:21,965 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:21,969 - DEBUG - Detecting objects in frame 5
# DEBUG:cd_fsod_test:Detecting objects in frame 5
# 2025-05-13 20:17:21,969 - DEBUG - Frame 5: No objects detected
# DEBUG:cd_fsod_test:Frame 5: No objects detected
# 2025-05-13 20:17:21,969 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:21,972 - DEBUG - Detecting objects in frame 6
# DEBUG:cd_fsod_test:Detecting objects in frame 6
# 2025-05-13 20:17:21,972 - DEBUG - Frame 6: No objects detected
# DEBUG:cd_fsod_test:Frame 6: No objects detected
# 2025-05-13 20:17:21,972 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:21,975 - DEBUG - Detecting objects in frame 7
# DEBUG:cd_fsod_test:Detecting objects in frame 7
# 2025-05-13 20:17:21,975 - DEBUG - Frame 7: No objects detected
# DEBUG:cd_fsod_test:Frame 7: No objects detected
# 2025-05-13 20:17:21,975 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:21,978 - DEBUG - Detecting objects in frame 8
# DEBUG:cd_fsod_test:Detecting objects in frame 8
# 2025-05-13 20:17:21,978 - DEBUG - Frame 8: No objects detected
# DEBUG:cd_fsod_test:Frame 8: No objects detected
# 2025-05-13 20:17:21,978 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:21,981 - DEBUG - Detecting objects in frame 9
# DEBUG:cd_fsod_test:Detecting objects in frame 9
# 2025-05-13 20:17:21,981 - DEBUG - Frame 9: No objects detected
# DEBUG:cd_fsod_test:Frame 9: No objects detected
# 2025-05-13 20:17:21,981 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:21,984 - DEBUG - Detecting objects in frame 10
# DEBUG:cd_fsod_test:Detecting objects in frame 10
# 2025-05-13 20:17:21,984 - DEBUG - Frame 10: No objects detected
# DEBUG:cd_fsod_test:Frame 10: No objects detected
# 2025-05-13 20:17:21,984 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:21,987 - DEBUG - Detecting objects in frame 11
# DEBUG:cd_fsod_test:Detecting objects in frame 11
# 2025-05-13 20:17:21,987 - DEBUG - Frame 11: No objects detected
# DEBUG:cd_fsod_test:Frame 11: No objects detected
# 2025-05-13 20:17:21,987 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:21,990 - DEBUG - Detecting objects in frame 12
# DEBUG:cd_fsod_test:Detecting objects in frame 12
# 2025-05-13 20:17:21,990 - DEBUG - Frame 12: No objects detected
# DEBUG:cd_fsod_test:Frame 12: No objects detected
# 2025-05-13 20:17:21,990 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:21,993 - DEBUG - Detecting objects in frame 13
# DEBUG:cd_fsod_test:Detecting objects in frame 13
# 2025-05-13 20:17:21,993 - DEBUG - Frame 13: No objects detected
# DEBUG:cd_fsod_test:Frame 13: No objects detected
# 2025-05-13 20:17:21,993 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:21,996 - DEBUG - Detecting objects in frame 14
# DEBUG:cd_fsod_test:Detecting objects in frame 14
# 2025-05-13 20:17:21,996 - DEBUG - Frame 14: No objects detected
# DEBUG:cd_fsod_test:Frame 14: No objects detected
# 2025-05-13 20:17:21,996 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:21,999 - DEBUG - Detecting objects in frame 15
# DEBUG:cd_fsod_test:Detecting objects in frame 15
# 2025-05-13 20:17:21,999 - DEBUG - Frame 15: No objects detected
# DEBUG:cd_fsod_test:Frame 15: No objects detected
# 2025-05-13 20:17:21,999 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,001 - DEBUG - Detecting objects in frame 16
# DEBUG:cd_fsod_test:Detecting objects in frame 16
# 2025-05-13 20:17:22,002 - DEBUG - Frame 16: No objects detected
# DEBUG:cd_fsod_test:Frame 16: No objects detected
# 2025-05-13 20:17:22,002 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,005 - DEBUG - Detecting objects in frame 17
# DEBUG:cd_fsod_test:Detecting objects in frame 17
# 2025-05-13 20:17:22,005 - DEBUG - Frame 17: No objects detected
# DEBUG:cd_fsod_test:Frame 17: No objects detected
# 2025-05-13 20:17:22,005 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,007 - DEBUG - Detecting objects in frame 18
# DEBUG:cd_fsod_test:Detecting objects in frame 18
# 2025-05-13 20:17:22,008 - DEBUG - Frame 18: No objects detected
# DEBUG:cd_fsod_test:Frame 18: No objects detected
# 2025-05-13 20:17:22,008 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,010 - DEBUG - Detecting objects in frame 19
# DEBUG:cd_fsod_test:Detecting objects in frame 19
# 2025-05-13 20:17:22,011 - DEBUG - Frame 19: No objects detected
# DEBUG:cd_fsod_test:Frame 19: No objects detected
# 2025-05-13 20:17:22,011 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,013 - DEBUG - Detecting objects in frame 20
# DEBUG:cd_fsod_test:Detecting objects in frame 20
# 2025-05-13 20:17:22,014 - DEBUG - Frame 20: No objects detected
# DEBUG:cd_fsod_test:Frame 20: No objects detected
# 2025-05-13 20:17:22,014 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,016 - DEBUG - Detecting objects in frame 21
# DEBUG:cd_fsod_test:Detecting objects in frame 21
# 2025-05-13 20:17:22,016 - DEBUG - Frame 21: No objects detected
# DEBUG:cd_fsod_test:Frame 21: No objects detected
# 2025-05-13 20:17:22,016 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,019 - DEBUG - Detecting objects in frame 22
# DEBUG:cd_fsod_test:Detecting objects in frame 22
# 2025-05-13 20:17:22,019 - DEBUG - Frame 22: No objects detected
# DEBUG:cd_fsod_test:Frame 22: No objects detected
# 2025-05-13 20:17:22,020 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,022 - DEBUG - Detecting objects in frame 23
# DEBUG:cd_fsod_test:Detecting objects in frame 23
# 2025-05-13 20:17:22,023 - DEBUG - Frame 23: No objects detected
# DEBUG:cd_fsod_test:Frame 23: No objects detected
# 2025-05-13 20:17:22,023 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,025 - DEBUG - Detecting objects in frame 24
# DEBUG:cd_fsod_test:Detecting objects in frame 24
# 2025-05-13 20:17:22,025 - DEBUG - Frame 24: No objects detected
# DEBUG:cd_fsod_test:Frame 24: No objects detected
# 2025-05-13 20:17:22,026 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,028 - DEBUG - Detecting objects in frame 25
# DEBUG:cd_fsod_test:Detecting objects in frame 25
# 2025-05-13 20:17:22,028 - DEBUG - Frame 25: No objects detected
# DEBUG:cd_fsod_test:Frame 25: No objects detected
# 2025-05-13 20:17:22,029 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,031 - DEBUG - Detecting objects in frame 26
# DEBUG:cd_fsod_test:Detecting objects in frame 26
# 2025-05-13 20:17:22,031 - DEBUG - Frame 26: No objects detected
# DEBUG:cd_fsod_test:Frame 26: No objects detected
# 2025-05-13 20:17:22,032 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,034 - DEBUG - Detecting objects in frame 27
# DEBUG:cd_fsod_test:Detecting objects in frame 27
# 2025-05-13 20:17:22,034 - DEBUG - Frame 27: No objects detected
# DEBUG:cd_fsod_test:Frame 27: No objects detected
# 2025-05-13 20:17:22,035 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,038 - DEBUG - Detecting objects in frame 28
# DEBUG:cd_fsod_test:Detecting objects in frame 28
# 2025-05-13 20:17:22,038 - DEBUG - Frame 28: No objects detected
# DEBUG:cd_fsod_test:Frame 28: No objects detected
# 2025-05-13 20:17:22,038 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,041 - DEBUG - Detecting objects in frame 29
# DEBUG:cd_fsod_test:Detecting objects in frame 29
# 2025-05-13 20:17:22,041 - DEBUG - Frame 29: No objects detected
# DEBUG:cd_fsod_test:Frame 29: No objects detected
# 2025-05-13 20:17:22,041 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,044 - DEBUG - Detecting objects in frame 30
# DEBUG:cd_fsod_test:Detecting objects in frame 30
# 2025-05-13 20:17:22,044 - DEBUG - Frame 30: No objects detected
# DEBUG:cd_fsod_test:Frame 30: No objects detected
# 2025-05-13 20:17:22,044 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,046 - DEBUG - Detecting objects in frame 31
# DEBUG:cd_fsod_test:Detecting objects in frame 31
# 2025-05-13 20:17:22,046 - DEBUG - Frame 31: No objects detected
# DEBUG:cd_fsod_test:Frame 31: No objects detected
# 2025-05-13 20:17:22,047 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,050 - DEBUG - Detecting objects in frame 32
# DEBUG:cd_fsod_test:Detecting objects in frame 32
# 2025-05-13 20:17:22,050 - DEBUG - Frame 32: No objects detected
# DEBUG:cd_fsod_test:Frame 32: No objects detected
# 2025-05-13 20:17:22,050 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,053 - DEBUG - Detecting objects in frame 33
# DEBUG:cd_fsod_test:Detecting objects in frame 33
# 2025-05-13 20:17:22,053 - DEBUG - Frame 33: No objects detected
# DEBUG:cd_fsod_test:Frame 33: No objects detected
# 2025-05-13 20:17:22,053 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,056 - DEBUG - Detecting objects in frame 34
# DEBUG:cd_fsod_test:Detecting objects in frame 34
# 2025-05-13 20:17:22,056 - DEBUG - Frame 34: No objects detected
# DEBUG:cd_fsod_test:Frame 34: No objects detected
# 2025-05-13 20:17:22,056 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,059 - DEBUG - Detecting objects in frame 35
# DEBUG:cd_fsod_test:Detecting objects in frame 35
# 2025-05-13 20:17:22,059 - DEBUG - Frame 35: No objects detected
# DEBUG:cd_fsod_test:Frame 35: No objects detected
# 2025-05-13 20:17:22,059 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,062 - DEBUG - Detecting objects in frame 36
# DEBUG:cd_fsod_test:Detecting objects in frame 36
# 2025-05-13 20:17:22,062 - DEBUG - Frame 36: No objects detected
# DEBUG:cd_fsod_test:Frame 36: No objects detected
# 2025-05-13 20:17:22,062 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,065 - DEBUG - Detecting objects in frame 37
# DEBUG:cd_fsod_test:Detecting objects in frame 37
# 2025-05-13 20:17:22,066 - DEBUG - Frame 37: No objects detected
# DEBUG:cd_fsod_test:Frame 37: No objects detected
# 2025-05-13 20:17:22,066 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,069 - DEBUG - Detecting objects in frame 38
# DEBUG:cd_fsod_test:Detecting objects in frame 38
# 2025-05-13 20:17:22,069 - DEBUG - Frame 38: No objects detected
# DEBUG:cd_fsod_test:Frame 38: No objects detected
# 2025-05-13 20:17:22,069 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,072 - DEBUG - Detecting objects in frame 39
# DEBUG:cd_fsod_test:Detecting objects in frame 39
# 2025-05-13 20:17:22,073 - DEBUG - Frame 39: No objects detected
# DEBUG:cd_fsod_test:Frame 39: No objects detected
# 2025-05-13 20:17:22,073 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,076 - DEBUG - Detecting objects in frame 40
# DEBUG:cd_fsod_test:Detecting objects in frame 40
# 2025-05-13 20:17:22,076 - DEBUG - Frame 40: No objects detected
# DEBUG:cd_fsod_test:Frame 40: No objects detected
# 2025-05-13 20:17:22,076 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,079 - DEBUG - Detecting objects in frame 41
# DEBUG:cd_fsod_test:Detecting objects in frame 41
# 2025-05-13 20:17:22,079 - DEBUG - Frame 41: No objects detected
# DEBUG:cd_fsod_test:Frame 41: No objects detected
# 2025-05-13 20:17:22,079 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,082 - DEBUG - Detecting objects in frame 42
# DEBUG:cd_fsod_test:Detecting objects in frame 42
# 2025-05-13 20:17:22,082 - DEBUG - Frame 42: No objects detected
# DEBUG:cd_fsod_test:Frame 42: No objects detected
# 2025-05-13 20:17:22,082 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,085 - DEBUG - Detecting objects in frame 43
# DEBUG:cd_fsod_test:Detecting objects in frame 43
# 2025-05-13 20:17:22,086 - DEBUG - Frame 43: No objects detected
# DEBUG:cd_fsod_test:Frame 43: No objects detected
# 2025-05-13 20:17:22,086 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,089 - DEBUG - Detecting objects in frame 44
# DEBUG:cd_fsod_test:Detecting objects in frame 44
# 2025-05-13 20:17:22,089 - DEBUG - Frame 44: No objects detected
# DEBUG:cd_fsod_test:Frame 44: No objects detected
# 2025-05-13 20:17:22,089 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,092 - DEBUG - Detecting objects in frame 45
# DEBUG:cd_fsod_test:Detecting objects in frame 45
# 2025-05-13 20:17:22,092 - DEBUG - Frame 45: No objects detected
# DEBUG:cd_fsod_test:Frame 45: No objects detected
# 2025-05-13 20:17:22,092 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,095 - DEBUG - Detecting objects in frame 46
# DEBUG:cd_fsod_test:Detecting objects in frame 46
# 2025-05-13 20:17:22,095 - DEBUG - Frame 46: No objects detected
# DEBUG:cd_fsod_test:Frame 46: No objects detected
# 2025-05-13 20:17:22,095 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,098 - DEBUG - Detecting objects in frame 47
# DEBUG:cd_fsod_test:Detecting objects in frame 47
# 2025-05-13 20:17:22,098 - DEBUG - Frame 47: No objects detected
# DEBUG:cd_fsod_test:Frame 47: No objects detected
# 2025-05-13 20:17:22,098 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,101 - DEBUG - Detecting objects in frame 48
# DEBUG:cd_fsod_test:Detecting objects in frame 48
# 2025-05-13 20:17:22,101 - DEBUG - Frame 48: No objects detected
# DEBUG:cd_fsod_test:Frame 48: No objects detected
# 2025-05-13 20:17:22,101 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,104 - DEBUG - Detecting objects in frame 49
# DEBUG:cd_fsod_test:Detecting objects in frame 49
# 2025-05-13 20:17:22,104 - DEBUG - Frame 49: No objects detected
# DEBUG:cd_fsod_test:Frame 49: No objects detected
# 2025-05-13 20:17:22,104 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,107 - DEBUG - Detecting objects in frame 50
# DEBUG:cd_fsod_test:Detecting objects in frame 50
# 2025-05-13 20:17:22,107 - DEBUG - Frame 50: No objects detected
# DEBUG:cd_fsod_test:Frame 50: No objects detected
# 2025-05-13 20:17:22,107 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,110 - DEBUG - Detecting objects in frame 51
# DEBUG:cd_fsod_test:Detecting objects in frame 51
# 2025-05-13 20:17:22,110 - DEBUG - Frame 51: No objects detected
# DEBUG:cd_fsod_test:Frame 51: No objects detected
# 2025-05-13 20:17:22,110 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,113 - DEBUG - Detecting objects in frame 52
# DEBUG:cd_fsod_test:Detecting objects in frame 52
# 2025-05-13 20:17:22,113 - DEBUG - Frame 52: No objects detected
# DEBUG:cd_fsod_test:Frame 52: No objects detected
# 2025-05-13 20:17:22,113 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# 2025-05-13 20:17:22,116 - DEBUG - Detecting objects in frame 53
# DEBUG:cd_fsod_test:Detecting objects in frame 53
# 2025-05-13 20:17:22,116 - DEBUG - Frame 53: No objects detected
# DEBUG:cd_fsod_test:Frame 53: No objects detected
# 2025-05-13 20:17:22,116 - DEBUG - Detection took 0.1ms
# DEBUG:cd_fsod_test:Detection took 0.1ms
# Found 1 unique objects to process with SAM2

# ==== Processing object 1 (Time Stick) separately ====
# Resetting SAM2 state for object 1...
# frame loading (JPEG): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 54/54 [00:01<00:00, 38.51it/s]
# Set video from directory: /home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/data/frames/Scenes 061-080__265H-2-_20230815215828529
# Adding box for object 1 at frame 3
# /home/ubuntu/code/drew/sam2/sam2/sam2_video_predictor.py:786: UserWarning: cannot import name '_C' from 'sam2' (/home/ubuntu/code/drew/sam2/sam2/__init__.py)

# Skipping the post-processing step due to the error above. You can still use SAM 2 and it's OK to ignore the error above, although some post-processing functionality may be limited (which doesn't affect the results in most cases; see https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).
#   pred_masks_gpu = fill_holes_in_mask_scores(
# Running propagation for object 1...
# Starting propagate_masks with objects_to_track=[1]
# Using predictor type: SAM2VideoPredictor
# Propagate method: propagate_in_video from sam2.sam2_video_predictor
# Iterator type: generator
# propagate in video:   0%|                                                                                                                                         | 0/51 [00:00<?, ?it/s]First item type: tuple
# First item value: (3, [1], tensor([[[[-12.1376, -12.1376, -12.0080,  ..., -11.5534, -11.4969, -11.4969],
#           [-12.1437, -12.1437, -11.9905,  ..., -11.3386, -11.1093, -11.1093],
#           [-12.1573, -12.1573, -11.9513,  ..., -10.8564, -10.2387, -10.2387],
#           ...,
#           [-11.8437, -11.8437, -11.5942,  ..., -10.7574, -10.2548, -10.2548],
#           [-10.7387, -10.7387, -10.7928,  ..., -11.0382, -10.5994, -10.5994],
#           [-10.2466, -10.2466, -10.4359,  ..., -11.1633, -10.7529, -10.7529]]]],
#        device='cuda:0'))
# First item tuple length: 3
#   Element 0: type=int, value=3
#   Element 1: type=list, value=[1]
#   Element 2: type=Tensor, value=tensor([[[[-12.1376, -12.1376, -12.0080,  ..., -11.5534, -11.4969, -11.4969],
#           [-12.1437, -12.1437, -11.9905,  ..., -11.3386, -11.1093, -11.1093],
#           [-12.1573, -12.1573, -11.9513,  ..., -10.8564, -10.2387, -10.2387],
#           ...,
#           [-11.8437, -11.8437, -11.5942,  ..., -10.7574, -10.2548, -10.2548],
#           [-10.7387, -10.7387, -10.7928,  ..., -11.0382, -10.5994, -10.5994],
#           [-10.2466, -10.2466, -10.4359,  ..., -11.1633, -10.7529, -10.7529]]]],
#        device='cuda:0')
# Processing item 1: frame=3, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 3
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 29970, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 29970 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [459.0, 232.0, 633.0, 480.0]
# Processed frame 3, found 1 objects
# Processing item 2: frame=4, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 4
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 27617, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 27617 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [386.0, 265.0, 576.0, 480.0]
# Processed frame 4, found 1 objects
# propagate in video:   4%|█████                                                                                                                            | 2/51 [00:00<00:17,  2.86it/s]Processing item 3: frame=5, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 5
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 27160, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 27160 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [296.0, 270.0, 513.0, 481.0]
# Processed frame 5, found 1 objects
# propagate in video:   6%|███████▌                                                                                                                         | 3/51 [00:01<00:23,  2.05it/s]Processing item 4: frame=6, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 6
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 28683, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 28683 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [234.0, 231.0, 495.0, 481.0]
# Processed frame 6, found 1 objects
# propagate in video:   8%|██████████                                                                                                                       | 4/51 [00:02<00:26,  1.77it/s]Processing item 5: frame=7, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 7
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 23273, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 23273 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [231.0, 273.0, 452.0, 481.0]
# Processed frame 7, found 1 objects
# propagate in video:  10%|████████████▋                                                                                                                    | 5/51 [00:02<00:28,  1.60it/s]Processing item 6: frame=8, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 8
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 21370, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 21370 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [309.0, 289.0, 492.0, 481.0]
# Processed frame 8, found 1 objects
# propagate in video:  12%|███████████████▏                                                                                                                 | 6/51 [00:03<00:30,  1.50it/s]Processing item 7: frame=9, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 9
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 20164, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 20164 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [320.0, 304.0, 496.0, 481.0]
# Processed frame 9, found 1 objects
# propagate in video:  14%|█████████████████▋                                                                                                               | 7/51 [00:04<00:31,  1.42it/s]Processing item 8: frame=10, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 10
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 19455, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 19455 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [314.0, 312.0, 487.0, 481.0]
# Processed frame 10, found 1 objects
# propagate in video:  16%|████████████████████▏                                                                                                            | 8/51 [00:05<00:31,  1.35it/s]Processing item 9: frame=11, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 11
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 20962, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 20962 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [309.0, 298.0, 493.0, 481.0]
# Processed frame 11, found 1 objects
# propagate in video:  18%|██████████████████████▊                                                                                                          | 9/51 [00:05<00:31,  1.31it/s]Processing item 10: frame=12, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 12
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 22393, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 22393 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [306.0, 287.0, 495.0, 481.0]
# Processed frame 12, found 1 objects
# propagate in video:  20%|█████████████████████████                                                                                                       | 10/51 [00:06<00:31,  1.29it/s]Processing item 11: frame=13, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 13
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 23139, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 23139 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [312.0, 281.0, 527.0, 481.0]
# Processed frame 13, found 1 objects
# propagate in video:  22%|███████████████████████████▌                                                                                                    | 11/51 [00:07<00:31,  1.27it/s]Processing item 12: frame=14, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 14
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 23049, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 23049 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [302.0, 294.0, 505.0, 481.0]
# Processed frame 14, found 1 objects
# propagate in video:  24%|██████████████████████████████                                                                                                  | 12/51 [00:08<00:31,  1.26it/s]Processing item 13: frame=15, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 15
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 22082, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 22082 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [302.0, 293.0, 491.0, 481.0]
# Processed frame 15, found 1 objects
# propagate in video:  25%|████████████████████████████████▋                                                                                               | 13/51 [00:09<00:30,  1.25it/s]Processing item 14: frame=16, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 16
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 6671, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 6671 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [286.0, 419.0, 594.0, 480.0]
# Processed frame 16, found 1 objects
# propagate in video:  27%|███████████████████████████████████▏                                                                                            | 14/51 [00:10<00:29,  1.24it/s]Processing item 15: frame=17, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 17
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 17 (no true pixels)
# Using fallback box from previous frame: [286.0, 419.0, 594.0, 480.0]
# Processed frame 17, found 1 objects
# propagate in video:  29%|█████████████████████████████████████▋                                                                                          | 15/51 [00:10<00:29,  1.24it/s]Processing item 16: frame=18, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 18
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 18 (no true pixels)
# Using fallback box from previous frame: [286.0, 419.0, 594.0, 480.0]
# Processed frame 18, found 1 objects
# propagate in video:  31%|████████████████████████████████████████▏                                                                                       | 16/51 [00:11<00:28,  1.23it/s]Processing item 17: frame=19, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 19
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 17444, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 17444 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [187.0, 370.0, 460.0, 481.0]
# Processed frame 19, found 1 objects
# propagate in video:  33%|██████████████████████████████████████████▋                                                                                     | 17/51 [00:12<00:27,  1.23it/s]Processing item 18: frame=20, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 20
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 20 (no true pixels)
# Using fallback box from previous frame: [187.0, 370.0, 460.0, 481.0]
# Processed frame 20, found 1 objects
# propagate in video:  35%|█████████████████████████████████████████████▏                                                                                  | 18/51 [00:13<00:26,  1.23it/s]Processing item 19: frame=21, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 21
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 21810, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 21810 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [662.0, 235.0, 838.0, 480.0]
# Processed frame 21, found 1 objects
# propagate in video:  37%|███████████████████████████████████████████████▋                                                                                | 19/51 [00:14<00:26,  1.23it/s]Processing item 20: frame=22, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 22
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 22 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 22, found 1 objects
# propagate in video:  39%|██████████████████████████████████████████████████▏                                                                             | 20/51 [00:14<00:25,  1.23it/s]Processing item 21: frame=23, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 23
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 23 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 23, found 1 objects
# propagate in video:  41%|████████████████████████████████████████████████████▋                                                                           | 21/51 [00:15<00:24,  1.23it/s]Processing item 22: frame=24, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 24
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 24 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 24, found 1 objects
# propagate in video:  43%|███████████████████████████████████████████████████████▏                                                                        | 22/51 [00:16<00:23,  1.22it/s]Processing item 23: frame=25, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 25
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 25 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 25, found 1 objects
# propagate in video:  45%|█████████████████████████████████████████████████████████▋                                                                      | 23/51 [00:17<00:22,  1.22it/s]Processing item 24: frame=26, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 26
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 26 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 26, found 1 objects
# propagate in video:  47%|████████████████████████████████████████████████████████████▏                                                                   | 24/51 [00:18<00:22,  1.22it/s]Processing item 25: frame=27, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 27
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 27 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 27, found 1 objects
# propagate in video:  49%|██████████████████████████████████████████████████████████████▋                                                                 | 25/51 [00:19<00:21,  1.22it/s]Processing item 26: frame=28, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 28
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 28 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 28, found 1 objects
# propagate in video:  51%|█████████████████████████████████████████████████████████████████▎                                                              | 26/51 [00:19<00:20,  1.22it/s]Processing item 27: frame=29, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 29
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 29 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 29, found 1 objects
# propagate in video:  53%|███████████████████████████████████████████████████████████████████▊                                                            | 27/51 [00:20<00:19,  1.22it/s]Processing item 28: frame=30, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 30
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 30 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 30, found 1 objects
# propagate in video:  55%|██████████████████████████████████████████████████████████████████████▎                                                         | 28/51 [00:21<00:18,  1.22it/s]Processing item 29: frame=31, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 31
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 31 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 31, found 1 objects
# propagate in video:  57%|████████████████████████████████████████████████████████████████████████▊                                                       | 29/51 [00:22<00:18,  1.22it/s]Processing item 30: frame=32, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 32
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 32 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 32, found 1 objects
# propagate in video:  59%|███████████████████████████████████████████████████████████████████████████▎                                                    | 30/51 [00:23<00:17,  1.22it/s]Processing item 31: frame=33, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 33
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 33 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 33, found 1 objects
# propagate in video:  61%|█████████████████████████████████████████████████████████████████████████████▊                                                  | 31/51 [00:23<00:16,  1.22it/s]Processing item 32: frame=34, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 34
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 34 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 34, found 1 objects
# propagate in video:  63%|████████████████████████████████████████████████████████████████████████████████▎                                               | 32/51 [00:24<00:15,  1.22it/s]Processing item 33: frame=35, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 35
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 35 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 35, found 1 objects
# propagate in video:  65%|██████████████████████████████████████████████████████████████████████████████████▊                                             | 33/51 [00:25<00:14,  1.22it/s]Processing item 34: frame=36, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 36
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 36 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 36, found 1 objects
# propagate in video:  67%|█████████████████████████████████████████████████████████████████████████████████████▎                                          | 34/51 [00:26<00:13,  1.22it/s]Processing item 35: frame=37, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 37
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 37 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 37, found 1 objects
# propagate in video:  69%|███████████████████████████████████████████████████████████████████████████████████████▊                                        | 35/51 [00:27<00:13,  1.22it/s]Processing item 36: frame=38, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 38
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 38 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 38, found 1 objects
# propagate in video:  71%|██████████████████████████████████████████████████████████████████████████████████████████▎                                     | 36/51 [00:28<00:12,  1.22it/s]Processing item 37: frame=39, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 39
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 39 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 39, found 1 objects
# propagate in video:  73%|████████████████████████████████████████████████████████████████████████████████████████████▊                                   | 37/51 [00:28<00:11,  1.22it/s]Processing item 38: frame=40, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 40
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 40 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 40, found 1 objects
# propagate in video:  75%|███████████████████████████████████████████████████████████████████████████████████████████████▎                                | 38/51 [00:29<00:10,  1.22it/s]Processing item 39: frame=41, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 41
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 41 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 41, found 1 objects
# propagate in video:  76%|█████████████████████████████████████████████████████████████████████████████████████████████████▉                              | 39/51 [00:30<00:09,  1.22it/s]Processing item 40: frame=42, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 42
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 42 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 42, found 1 objects
# propagate in video:  78%|████████████████████████████████████████████████████████████████████████████████████████████████████▍                           | 40/51 [00:31<00:09,  1.22it/s]Processing item 41: frame=43, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 43
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 43 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 43, found 1 objects
# propagate in video:  80%|██████████████████████████████████████████████████████████████████████████████████████████████████████▉                         | 41/51 [00:32<00:08,  1.22it/s]Processing item 42: frame=44, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 44
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 44 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 44, found 1 objects
# propagate in video:  82%|█████████████████████████████████████████████████████████████████████████████████████████████████████████▍                      | 42/51 [00:32<00:07,  1.22it/s]Processing item 43: frame=45, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 45
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 45 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 45, found 1 objects
# propagate in video:  84%|███████████████████████████████████████████████████████████████████████████████████████████████████████████▉                    | 43/51 [00:33<00:06,  1.22it/s]Processing item 44: frame=46, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 46
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 46 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 46, found 1 objects
# propagate in video:  86%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                 | 44/51 [00:34<00:05,  1.22it/s]Processing item 45: frame=47, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 47
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 47 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 47, found 1 objects
# propagate in video:  88%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉               | 45/51 [00:35<00:04,  1.22it/s]Processing item 46: frame=48, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 48
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 48 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 48, found 1 objects
# propagate in video:  90%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍            | 46/51 [00:36<00:04,  1.22it/s]Processing item 47: frame=49, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 49
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 49 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 49, found 1 objects
# propagate in video:  92%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉          | 47/51 [00:37<00:03,  1.22it/s]Processing item 48: frame=50, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 50
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 50 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 50, found 1 objects
# propagate in video:  94%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍       | 48/51 [00:37<00:02,  1.22it/s]Processing item 49: frame=51, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 51
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 51 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 51, found 1 objects
# propagate in video:  96%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉     | 49/51 [00:38<00:01,  1.22it/s]Processing item 50: frame=52, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 52
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 52 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 52, found 1 objects
# propagate in video:  98%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍  | 50/51 [00:39<00:00,  1.22it/s]Processing item 51: frame=53, objects=[1]
# Filtering objects to track: [1]
# After filtering: 1 objects remain
# Processing object 1 in frame 53
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 1 in frame 53 (no true pixels)
# Using fallback box from previous frame: [662.0, 235.0, 838.0, 480.0]
# Processed frame 53, found 1 objects
# propagate in video: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 51/51 [00:40<00:00,  1.26it/s]
# Finished propagation, processed 51 frames, found 51 frames with objects
# Mask statistics: 16 valid masks, 35 empty masks
# DEBUG: propagate_masks returned type: <class 'tuple'>
# DEBUG: propagate_masks tuple length: 2
# DEBUG: Unpacked 2-element tuple - segments (51 frames) and boxes (51 frames)
# DEBUG: Box statistics - 16 valid boxes, 35 fallback boxes
# DEBUG: Segments contains 51 frames
# DEBUG: Frame 3 has 1 objects
# DEBUG: Frame 4 has 1 objects
# DEBUG: Frame 5 has 1 objects
# Successfully propagated masks for object 1, available in 51 frames
# Saving per-object segmentation visualizations...
# Processing object 1 visible from frame 3 to 3
#   Saved 1 visualizations for object #1 (Time Stick)
# All per-object mask visualizations saved to cd_fsod_results/object_masks
# Saving first detection frames for each object...
# Using detection files from: /home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/data/detections/Scenes 061-080__265H-2-_20230815215828529
#   Object #1 (Time Stick) first detected in frame 3, using 3.json
#     JSON file contains 51 detections
#     Sample detection for Time Stick: confidence=0.9932, coordinates=[453, 224, 643, 473]
#   Saved first detection frame for object #1 (Time Stick)
# All first detection frames saved to cd_fsod_results/first_detections
# Creating object-to-frame mapping...
# Object tracking statistics:
#   Total valid frames across all objects: 1
#   Total empty mask frames: 0
#   Total fallback box frames: 0
#   Valid frame percentage: 100.0%
# Verifying consistency with saved mask images...
# Verification successful: All objects' masks are consistent with the mapping
# Saved object-to-frame mapping to cd_fsod_results/object_frame_mapping.json
# All results saved to: cd_fsod_results
# 2025-05-13 20:18:05,988 - INFO - Video processing completed in 44.23 seconds
# INFO:cd_fsod_test:Video processing completed in 44.23 seconds
# 2025-05-13 20:18:05,988 - DEBUG - First detection details:
# DEBUG:cd_fsod_test:First detection details:
# 2025-05-13 20:18:05,988 - DEBUG -   Object #1 (Time Stick) first detected at frame 3
# DEBUG:cd_fsod_test:  Object #1 (Time Stick) first detected at frame 3
# 2025-05-13 20:18:05,988 - DEBUG -     From 3.json: {'coordinates': [453, 224, 643, 473], 'label': 'Time Stick', 'confidence': 0.9931594133377075}
# DEBUG:cd_fsod_test:    From 3.json: {'coordinates': [453, 224, 643, 473], 'label': 'Time Stick', 'confidence': 0.9931594133377075}
# 2025-05-13 20:18:05,989 - INFO - Generated 62 output files
# INFO:cd_fsod_test:Generated 62 output files
# 2025-05-13 20:18:05,989 - DEBUG - Output files: ['frame_0049.jpg', 'frame_0045.jpg', 'frame_0014.jpg', 'frame_0025.jpg', 'frame_0051.jpg', 'frame_0039.jpg', 'frame_0030.jpg', 'frame_0008.jpg', 'frame_0002.jpg', 'frame_0042.jpg']
# DEBUG:cd_fsod_test:Output files: ['frame_0049.jpg', 'frame_0045.jpg', 'frame_0014.jpg', 'frame_0025.jpg', 'frame_0051.jpg', 'frame_0039.jpg', 'frame_0030.jpg', 'frame_0008.jpg', 'frame_0002.jpg', 'frame_0042.jpg']
# 2025-05-13 20:18:05,989 - INFO - ================================================================================
# INFO:cd_fsod_test:================================================================================
# 2025-05-13 20:18:05,989 - INFO - CD-FSOD Integration Test Completed Successfully
# INFO:cd_fsod_test:CD-FSOD Integration Test Completed Successfully
# 2025-05-13 20:18:05,989 - INFO - Total processing time: 48.00 seconds
# INFO:cd_fsod_test:Total processing time: 48.00 seconds
# 2025-05-13 20:18:05,989 - INFO - Results saved to: ./cd_fsod_results
# INFO:cd_fsod_test:Results saved to: ./cd_fsod_results
# 2025-05-13 20:18:05,989 - INFO - ================================================================================
# INFO:cd_fsod_test:================================================================================
# ^C