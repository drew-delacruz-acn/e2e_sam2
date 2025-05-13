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
# g objects to track: [2]
# After filtering: 1 objects remain
# Processing object 2 in frame 50
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 2 in frame 50 (no true pixels)
# Using fallback box from previous frame: [0.0, 367.0, 467.0, 480.0]
# Processed frame 50, found 1 objects
# propagate in video:  93%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋         | 38/41 [00:30<00:02,  1.21it/s]Processing item 39: frame=51, objects=[2]
# Filtering objects to track: [2]
# After filtering: 1 objects remain
# Processing object 2 in frame 51
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 2 in frame 51 (no true pixels)
# Using fallback box from previous frame: [0.0, 367.0, 467.0, 480.0]
# Processed frame 51, found 1 objects
# propagate in video:  95%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊      | 39/41 [00:30<00:01,  1.21it/s]Processing item 40: frame=52, objects=[2]
# Filtering objects to track: [2]
# After filtering: 1 objects remain
# Processing object 2 in frame 52
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 2 in frame 52 (no true pixels)
# Using fallback box from previous frame: [0.0, 367.0, 467.0, 480.0]
# Processed frame 52, found 1 objects
# propagate in video:  98%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉   | 40/41 [00:31<00:00,  1.21it/s]Processing item 41: frame=53, objects=[2]
# Filtering objects to track: [2]
# After filtering: 1 objects remain
# Processing object 2 in frame 53
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 2 in frame 53 (no true pixels)
# Using fallback box from previous frame: [0.0, 367.0, 467.0, 480.0]
# Processed frame 53, found 1 objects
# propagate in video: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41/41 [00:32<00:00,  1.26it/s]
# Finished propagation, processed 41 frames, found 41 frames with objects
# Mask statistics: 10 valid masks, 31 empty masks
# DEBUG: propagate_masks(objects_to_track=2) returned type: <class 'tuple'>
# DEBUG: propagate_masks(objects_to_track=2) tuple length: 2
#   Frame 13: Object #2 mask statistics - sum: 45434 pixels - HIGH QUALITY
#   Frame 14: Object #2 mask statistics - sum: 46901 pixels - HIGH QUALITY
#   Frame 15: Object #2 mask statistics - sum: 48741 pixels - HIGH QUALITY
#   Frame 16: Object #2 mask statistics - sum: 58284 pixels - HIGH QUALITY
#   Frame 17: Object #2 mask statistics - sum: 80564 pixels - HIGH QUALITY
#   Frame 18: Object #2 mask statistics - sum: 98649 pixels - HIGH QUALITY
#   Frame 19: Object #2 mask statistics - sum: 80948 pixels - HIGH QUALITY
#   Frame 20: Object #2 mask statistics - sum: 63661 pixels - HIGH QUALITY
#   Frame 21: Object #2 mask statistics - sum: 151589 pixels - HIGH QUALITY
#   Frame 22: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 23: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 24: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 25: Object #2 mask statistics - sum: 30608 pixels - HIGH QUALITY
#   Frame 26: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 27: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 28: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 29: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 30: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 31: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 32: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 33: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 34: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 35: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 36: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 37: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 38: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 39: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 40: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 41: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 42: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 43: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 44: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 45: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 46: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 47: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 48: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 49: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 50: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 51: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 52: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 53: Object #2 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 22: Object #2 using FALLBACK BOX
#   Frame 23: Object #2 using FALLBACK BOX
#   Frame 24: Object #2 using FALLBACK BOX
#   Frame 26: Object #2 using FALLBACK BOX
#   Frame 27: Object #2 using FALLBACK BOX
#   Frame 28: Object #2 using FALLBACK BOX
#   Frame 29: Object #2 using FALLBACK BOX
#   Frame 30: Object #2 using FALLBACK BOX
#   Frame 31: Object #2 using FALLBACK BOX
#   Frame 32: Object #2 using FALLBACK BOX
#   Frame 33: Object #2 using FALLBACK BOX
#   Frame 34: Object #2 using FALLBACK BOX
#   Frame 35: Object #2 using FALLBACK BOX
#   Frame 36: Object #2 using FALLBACK BOX
#   Frame 37: Object #2 using FALLBACK BOX
#   Frame 38: Object #2 using FALLBACK BOX
#   Frame 39: Object #2 using FALLBACK BOX
#   Frame 40: Object #2 using FALLBACK BOX
#   Frame 41: Object #2 using FALLBACK BOX
#   Frame 42: Object #2 using FALLBACK BOX
#   Frame 43: Object #2 using FALLBACK BOX
#   Frame 44: Object #2 using FALLBACK BOX
#   Frame 45: Object #2 using FALLBACK BOX
#   Frame 46: Object #2 using FALLBACK BOX
#   Frame 47: Object #2 using FALLBACK BOX
#   Frame 48: Object #2 using FALLBACK BOX
#   Frame 49: Object #2 using FALLBACK BOX
#   Frame 50: Object #2 using FALLBACK BOX
#   Frame 51: Object #2 using FALLBACK BOX
#   Frame 52: Object #2 using FALLBACK BOX
#   Frame 53: Object #2 using FALLBACK BOX
# Adding box for frame 13 to object 2: [379.0, 110.0, 602.0, 479.0]
# Adding box for frame 14 to object 2: [369.0, 101.0, 590.0, 478.0]
# Adding box for frame 15 to object 2: [365.0, 115.0, 622.0, 478.0]
# Adding box for frame 16 to object 2: [293.0, 101.0, 579.0, 480.0]
# Adding box for frame 17 to object 2: [480.0, 103.0, 827.0, 474.0]
# Adding box for frame 18 to object 2: [159.0, 123.0, 611.0, 477.0]
# Adding box for frame 19 to object 2: [0.0, 59.0, 213.0, 477.0]
# Adding box for frame 20 to object 2: [0.0, 227.0, 331.0, 479.0]
# Adding box for frame 21 to object 2: [199.0, 60.0, 804.0, 479.0]
# Adding box for frame 22 to object 2: [199.0, 60.0, 804.0, 479.0]
# Adding box for frame 23 to object 2: [199.0, 60.0, 804.0, 479.0]
# Adding box for frame 24 to object 2: [199.0, 60.0, 804.0, 479.0]
# Adding box for frame 25 to object 2: [0.0, 367.0, 467.0, 480.0]
# Adding box for frame 26 to object 2: [0.0, 367.0, 467.0, 480.0]
# Adding box for frame 27 to object 2: [0.0, 367.0, 467.0, 480.0]
# Adding box for frame 28 to object 2: [0.0, 367.0, 467.0, 480.0]
# Adding box for frame 29 to object 2: [0.0, 367.0, 467.0, 480.0]
# Adding box for frame 30 to object 2: [0.0, 367.0, 467.0, 480.0]
# Adding box for frame 31 to object 2: [0.0, 367.0, 467.0, 480.0]
# Adding box for frame 32 to object 2: [0.0, 367.0, 467.0, 480.0]
# Adding box for frame 33 to object 2: [0.0, 367.0, 467.0, 480.0]
# Adding box for frame 34 to object 2: [0.0, 367.0, 467.0, 480.0]
# Adding box for frame 35 to object 2: [0.0, 367.0, 467.0, 480.0]
# Adding box for frame 36 to object 2: [0.0, 367.0, 467.0, 480.0]
# Adding box for frame 37 to object 2: [0.0, 367.0, 467.0, 480.0]
# Adding box for frame 38 to object 2: [0.0, 367.0, 467.0, 480.0]
# Adding box for frame 39 to object 2: [0.0, 367.0, 467.0, 480.0]
# Adding box for frame 40 to object 2: [0.0, 367.0, 467.0, 480.0]
# Adding box for frame 41 to object 2: [0.0, 367.0, 467.0, 480.0]
# Adding box for frame 42 to object 2: [0.0, 367.0, 467.0, 480.0]
# Adding box for frame 43 to object 2: [0.0, 367.0, 467.0, 480.0]
# Adding box for frame 44 to object 2: [0.0, 367.0, 467.0, 480.0]
# Adding box for frame 45 to object 2: [0.0, 367.0, 467.0, 480.0]
# Adding box for frame 46 to object 2: [0.0, 367.0, 467.0, 480.0]
# Adding box for frame 47 to object 2: [0.0, 367.0, 467.0, 480.0]
# Adding box for frame 48 to object 2: [0.0, 367.0, 467.0, 480.0]
# Adding box for frame 49 to object 2: [0.0, 367.0, 467.0, 480.0]
# Adding box for frame 50 to object 2: [0.0, 367.0, 467.0, 480.0]
# Adding box for frame 51 to object 2: [0.0, 367.0, 467.0, 480.0]
# Adding box for frame 52 to object 2: [0.0, 367.0, 467.0, 480.0]
# Adding box for frame 53 to object 2: [0.0, 367.0, 467.0, 480.0]
# Updating object 2 with 41 boxes from mask propagation
# Error during propagation for object 2: 2
# Propagation error traceback: Traceback (most recent call last):
#   File "/home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/src/object_tracking_pipeline.py", line 1240, in process_video_separate_objects
#     self.tracked_objects[obj_id]["boxes"] = all_boxes
#     ~~~~~~~~~~~~~~~~~~~~^^^^^^^^
# KeyError: 2


# ==== Processing object 3 (Sylvie's horned headpiece) separately ====
# Resetting SAM2 state for object 3...
# frame loading (JPEG): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 54/54 [00:01<00:00, 38.76it/s]
# Set video from directory: /home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/data/frames/Scenes 061-080__265H-2-_20230815215828529
# Adding box for object 3 at frame 18
# Running propagation for object 3...
# Starting propagate_masks with objects_to_track=[3]
# Using predictor type: SAM2VideoPredictor
# Propagate method: propagate_in_video from sam2.sam2_video_predictor
# Iterator type: generator
# propagate in video:   0%|                                                                                                                                         | 0/36 [00:00<?, ?it/s]First item type: tuple
# First item value: (18, [3], tensor([[[[-17.3911, -17.3911, -18.4886,  ..., -25.1047, -26.5941, -26.5941],
#           [-18.9784, -18.9784, -19.5686,  ..., -23.6122, -24.2996, -24.2996],
#           [-22.5428, -22.5428, -21.9941,  ..., -20.2604, -19.1469, -19.1469],
#           ...,
#           [-20.9912, -20.9912, -21.0111,  ..., -23.9965, -23.5670, -23.5670],
#           [-21.3911, -21.3911, -21.2431,  ..., -22.0893, -20.7484, -20.7484],
#           [-21.5692, -21.5692, -21.3463,  ..., -21.2400, -19.4932, -19.4932]]]],
#        device='cuda:0'))
# First item tuple length: 3
#   Element 0: type=int, value=18
#   Element 1: type=list, value=[3]
#   Element 2: type=Tensor, value=tensor([[[[-17.3911, -17.3911, -18.4886,  ..., -25.1047, -26.5941, -26.5941],
#           [-18.9784, -18.9784, -19.5686,  ..., -23.6122, -24.2996, -24.2996],
#           [-22.5428, -22.5428, -21.9941,  ..., -20.2604, -19.1469, -19.1469],
#           ...,
#           [-20.9912, -20.9912, -21.0111,  ..., -23.9965, -23.5670, -23.5670],
#           [-21.3911, -21.3911, -21.2431,  ..., -22.0893, -20.7484, -20.7484],
#           [-21.5692, -21.5692, -21.3463,  ..., -21.2400, -19.4932, -19.4932]]]],
#        device='cuda:0')
# Processing item 1: frame=18, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 18
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 4329, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 4329 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [324.0, 124.0, 468.0, 197.0]
# Processed frame 18, found 1 objects
# Processing item 2: frame=19, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 19
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 3911, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 3911 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [541.0, 161.0, 656.0, 251.0]
# Processed frame 19, found 1 objects
# propagate in video:   6%|███████▏                                                                                                                         | 2/36 [00:00<00:11,  2.99it/s]Processing item 3: frame=20, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 20
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 6453, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 6453 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [545.0, 68.0, 712.0, 159.0]
# Processed frame 20, found 1 objects
# propagate in video:   8%|██████████▊                                                                                                                      | 3/36 [00:01<00:15,  2.07it/s]Processing item 4: frame=21, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 21
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 3 in frame 21 (no true pixels)
# Using fallback box from previous frame: [545.0, 68.0, 712.0, 159.0]
# Processed frame 21, found 1 objects
# propagate in video:  11%|██████████████▎                                                                                                                  | 4/36 [00:02<00:18,  1.75it/s]Processing item 5: frame=22, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 22
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 3614, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 3614 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [394.0, 188.0, 495.0, 276.0]
# Processed frame 22, found 1 objects
# propagate in video:  14%|█████████████████▉                                                                                                               | 5/36 [00:02<00:19,  1.58it/s]Processing item 6: frame=23, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 23
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 4035, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 4035 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [364.0, 77.0, 506.0, 150.0]
# Processed frame 23, found 1 objects
# propagate in video:  17%|█████████████████████▌                                                                                                           | 6/36 [00:03<00:20,  1.47it/s]Processing item 7: frame=24, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 24
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 3 in frame 24 (no true pixels)
# Using fallback box from previous frame: [364.0, 77.0, 506.0, 150.0]
# Processed frame 24, found 1 objects
# propagate in video:  19%|█████████████████████████                                                                                                        | 7/36 [00:04<00:20,  1.39it/s]Processing item 8: frame=25, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 25
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 3 in frame 25 (no true pixels)
# Using fallback box from previous frame: [364.0, 77.0, 506.0, 150.0]
# Processed frame 25, found 1 objects
# propagate in video:  22%|████████████████████████████▋                                                                                                    | 8/36 [00:05<00:21,  1.32it/s]Processing item 9: frame=26, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 26
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 3 in frame 26 (no true pixels)
# Using fallback box from previous frame: [364.0, 77.0, 506.0, 150.0]
# Processed frame 26, found 1 objects
# propagate in video:  25%|████████████████████████████████▎                                                                                                | 9/36 [00:06<00:21,  1.28it/s]Processing item 10: frame=27, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 27
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 3 in frame 27 (no true pixels)
# Using fallback box from previous frame: [364.0, 77.0, 506.0, 150.0]
# Processed frame 27, found 1 objects
# propagate in video:  28%|███████████████████████████████████▌                                                                                            | 10/36 [00:06<00:20,  1.26it/s]Processing item 11: frame=28, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 28
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 3 in frame 28 (no true pixels)
# Using fallback box from previous frame: [364.0, 77.0, 506.0, 150.0]
# Processed frame 28, found 1 objects
# propagate in video:  31%|███████████████████████████████████████                                                                                         | 11/36 [00:07<00:20,  1.24it/s]Processing item 12: frame=29, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 29
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 3 in frame 29 (no true pixels)
# Using fallback box from previous frame: [364.0, 77.0, 506.0, 150.0]
# Processed frame 29, found 1 objects
# propagate in video:  33%|██████████████████████████████████████████▋                                                                                     | 12/36 [00:08<00:19,  1.23it/s]Processing item 13: frame=30, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 30
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 3 in frame 30 (no true pixels)
# Using fallback box from previous frame: [364.0, 77.0, 506.0, 150.0]
# Processed frame 30, found 1 objects
# propagate in video:  36%|██████████████████████████████████████████████▏                                                                                 | 13/36 [00:09<00:18,  1.22it/s]Processing item 14: frame=31, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 31
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 3498, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 3498 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [515.0, 281.0, 725.0, 336.0]
# Processed frame 31, found 1 objects
# propagate in video:  39%|█████████████████████████████████████████████████▊                                                                              | 14/36 [00:10<00:18,  1.22it/s]Processing item 15: frame=32, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 32
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 3 in frame 32 (no true pixels)
# Using fallback box from previous frame: [515.0, 281.0, 725.0, 336.0]
# Processed frame 32, found 1 objects
# propagate in video:  42%|█████████████████████████████████████████████████████▎                                                                          | 15/36 [00:11<00:17,  1.21it/s]Processing item 16: frame=33, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 33
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 1530, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 1530 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [600.0, 102.0, 650.0, 186.0]
# Processed frame 33, found 1 objects
# propagate in video:  44%|████████████████████████████████████████████████████████▉                                                                       | 16/36 [00:11<00:16,  1.21it/s]Processing item 17: frame=34, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 34
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 2419, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 2419 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [323.0, 181.0, 408.0, 271.0]
# Processed frame 34, found 1 objects
# propagate in video:  47%|████████████████████████████████████████████████████████████▍                                                                   | 17/36 [00:12<00:15,  1.21it/s]Processing item 18: frame=35, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 35
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 1422, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 1422 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [579.0, 145.0, 641.0, 201.0]
# Processed frame 35, found 1 objects
# propagate in video:  50%|████████████████████████████████████████████████████████████████                                                                | 18/36 [00:13<00:14,  1.21it/s]Processing item 19: frame=36, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 36
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 1157, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 1157 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [488.0, 181.0, 527.0, 239.0]
# Processed frame 36, found 1 objects
# propagate in video:  53%|███████████████████████████████████████████████████████████████████▌                                                            | 19/36 [00:14<00:14,  1.20it/s]Processing item 20: frame=37, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 37
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 1118, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 1118 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [494.0, 202.0, 531.0, 253.0]
# Processed frame 37, found 1 objects
# propagate in video:  56%|███████████████████████████████████████████████████████████████████████                                                         | 20/36 [00:15<00:13,  1.20it/s]Processing item 21: frame=38, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 38
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 1208, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 1208 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [533.0, 176.0, 576.0, 225.0]
# Processed frame 38, found 1 objects
# propagate in video:  58%|██████████████████████████████████████████████████████████████████████████▋                                                     | 21/36 [00:16<00:12,  1.20it/s]Processing item 22: frame=39, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 39
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 1013, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 1013 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [469.0, 163.0, 504.0, 215.0]
# Processed frame 39, found 1 objects
# propagate in video:  61%|██████████████████████████████████████████████████████████████████████████████▏                                                 | 22/36 [00:16<00:11,  1.20it/s]Processing item 23: frame=40, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 40
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 3 in frame 40 (no true pixels)
# Using fallback box from previous frame: [469.0, 163.0, 504.0, 215.0]
# Processed frame 40, found 1 objects
# propagate in video:  64%|█████████████████████████████████████████████████████████████████████████████████▊                                              | 23/36 [00:17<00:10,  1.20it/s]Processing item 24: frame=41, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 41
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 466, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 466 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [496.0, 182.0, 512.0, 219.0]
# Processed frame 41, found 1 objects
# propagate in video:  67%|█████████████████████████████████████████████████████████████████████████████████████▎                                          | 24/36 [00:18<00:09,  1.20it/s]Processing item 25: frame=42, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 42
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 1116, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 1116 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [395.0, 119.0, 422.0, 176.0]
# Processed frame 42, found 1 objects
# propagate in video:  69%|████████████████████████████████████████████████████████████████████████████████████████▉                                       | 25/36 [00:19<00:09,  1.20it/s]Processing item 26: frame=43, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 43
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 953, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 953 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [495.0, 159.0, 525.0, 207.0]
# Processed frame 43, found 1 objects
# propagate in video:  72%|████████████████████████████████████████████████████████████████████████████████████████████▍                                   | 26/36 [00:20<00:08,  1.20it/s]Processing item 27: frame=44, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 44
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 1017, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 1017 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [485.0, 143.0, 519.0, 188.0]
# Processed frame 44, found 1 objects
# propagate in video:  75%|████████████████████████████████████████████████████████████████████████████████████████████████                                | 27/36 [00:21<00:07,  1.20it/s]Processing item 28: frame=45, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 45
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 849, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask with 849 true pixels
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [608.0, 242.0, 640.0, 283.0]
# Processed frame 45, found 1 objects
# propagate in video:  78%|███████████████████████████████████████████████████████████████████████████████████████████████████▌                            | 28/36 [00:21<00:06,  1.20it/s]Processing item 29: frame=46, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 46
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 3 in frame 46 (no true pixels)
# Using fallback box from previous frame: [608.0, 242.0, 640.0, 283.0]
# Processed frame 46, found 1 objects
# propagate in video:  81%|███████████████████████████████████████████████████████████████████████████████████████████████████████                         | 29/36 [00:22<00:05,  1.20it/s]Processing item 30: frame=47, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 47
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 3 in frame 47 (no true pixels)
# Using fallback box from previous frame: [608.0, 242.0, 640.0, 283.0]
# Processed frame 47, found 1 objects
# propagate in video:  83%|██████████████████████████████████████████████████████████████████████████████████████████████████████████▋                     | 30/36 [00:23<00:04,  1.20it/s]Processing item 31: frame=48, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 48
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 3 in frame 48 (no true pixels)
# Using fallback box from previous frame: [608.0, 242.0, 640.0, 283.0]
# Processed frame 48, found 1 objects
# propagate in video:  86%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                 | 31/36 [00:24<00:04,  1.20it/s]Processing item 32: frame=49, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 49
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 3 in frame 49 (no true pixels)
# Using fallback box from previous frame: [608.0, 242.0, 640.0, 283.0]
# Processed frame 49, found 1 objects
# propagate in video:  89%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊              | 32/36 [00:25<00:03,  1.20it/s]Processing item 33: frame=50, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 50
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 3 in frame 50 (no true pixels)
# Using fallback box from previous frame: [608.0, 242.0, 640.0, 283.0]
# Processed frame 50, found 1 objects
# propagate in video:  92%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎          | 33/36 [00:26<00:02,  1.20it/s]Processing item 34: frame=51, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 51
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 3 in frame 51 (no true pixels)
# Using fallback box from previous frame: [608.0, 242.0, 640.0, 283.0]
# Processed frame 51, found 1 objects
# propagate in video:  94%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉       | 34/36 [00:26<00:01,  1.20it/s]Processing item 35: frame=52, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 52
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 3 in frame 52 (no true pixels)
# Using fallback box from previous frame: [608.0, 242.0, 640.0, 283.0]
# Processed frame 52, found 1 objects
# propagate in video:  97%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍   | 35/36 [00:27<00:00,  1.20it/s]Processing item 36: frame=53, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 53
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Mask statistics - sum: 0, shape: torch.Size([1, 540, 960])
# Empty mask detected for object 3 in frame 53 (no true pixels)
# Using fallback box from previous frame: [608.0, 242.0, 640.0, 283.0]
# Processed frame 53, found 1 objects
# propagate in video: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 36/36 [00:28<00:00,  1.26it/s]
# Finished propagation, processed 36 frames, found 36 frames with objects
# Mask statistics: 18 valid masks, 18 empty masks
# DEBUG: propagate_masks(objects_to_track=3) returned type: <class 'tuple'>
# DEBUG: propagate_masks(objects_to_track=3) tuple length: 2
#   Frame 18: Object #3 mask statistics - sum: 4329 pixels - HIGH QUALITY
#   Frame 19: Object #3 mask statistics - sum: 3911 pixels - HIGH QUALITY
#   Frame 20: Object #3 mask statistics - sum: 6453 pixels - HIGH QUALITY
#   Frame 21: Object #3 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 22: Object #3 mask statistics - sum: 3614 pixels - HIGH QUALITY
#   Frame 23: Object #3 mask statistics - sum: 4035 pixels - HIGH QUALITY
#   Frame 24: Object #3 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 25: Object #3 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 26: Object #3 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 27: Object #3 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 28: Object #3 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 29: Object #3 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 30: Object #3 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 31: Object #3 mask statistics - sum: 3498 pixels - HIGH QUALITY
#   Frame 32: Object #3 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 33: Object #3 mask statistics - sum: 1530 pixels - HIGH QUALITY
#   Frame 34: Object #3 mask statistics - sum: 2419 pixels - HIGH QUALITY
#   Frame 35: Object #3 mask statistics - sum: 1422 pixels - HIGH QUALITY
#   Frame 36: Object #3 mask statistics - sum: 1157 pixels - HIGH QUALITY
#   Frame 37: Object #3 mask statistics - sum: 1118 pixels - HIGH QUALITY
#   Frame 38: Object #3 mask statistics - sum: 1208 pixels - HIGH QUALITY
#   Frame 39: Object #3 mask statistics - sum: 1013 pixels - HIGH QUALITY
#   Frame 40: Object #3 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 41: Object #3 mask statistics - sum: 466 pixels - HIGH QUALITY
#   Frame 42: Object #3 mask statistics - sum: 1116 pixels - HIGH QUALITY
#   Frame 43: Object #3 mask statistics - sum: 953 pixels - HIGH QUALITY
#   Frame 44: Object #3 mask statistics - sum: 1017 pixels - HIGH QUALITY
#   Frame 45: Object #3 mask statistics - sum: 849 pixels - HIGH QUALITY
#   Frame 46: Object #3 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 47: Object #3 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 48: Object #3 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 49: Object #3 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 50: Object #3 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 51: Object #3 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 52: Object #3 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 53: Object #3 mask statistics - sum: 0 pixels - HIGH QUALITY
#   Frame 21: Object #3 using FALLBACK BOX
#   Frame 24: Object #3 using FALLBACK BOX
#   Frame 25: Object #3 using FALLBACK BOX
#   Frame 26: Object #3 using FALLBACK BOX
#   Frame 27: Object #3 using FALLBACK BOX
#   Frame 28: Object #3 using FALLBACK BOX
#   Frame 29: Object #3 using FALLBACK BOX
#   Frame 30: Object #3 using FALLBACK BOX
#   Frame 32: Object #3 using FALLBACK BOX
#   Frame 40: Object #3 using FALLBACK BOX
#   Frame 46: Object #3 using FALLBACK BOX
#   Frame 47: Object #3 using FALLBACK BOX
#   Frame 48: Object #3 using FALLBACK BOX
#   Frame 49: Object #3 using FALLBACK BOX
#   Frame 50: Object #3 using FALLBACK BOX
#   Frame 51: Object #3 using FALLBACK BOX
#   Frame 52: Object #3 using FALLBACK BOX
#   Frame 53: Object #3 using FALLBACK BOX
# Adding box for frame 18 to object 3: [324.0, 124.0, 468.0, 197.0]
# Adding box for frame 19 to object 3: [541.0, 161.0, 656.0, 251.0]
# Adding box for frame 20 to object 3: [545.0, 68.0, 712.0, 159.0]
# Adding box for frame 21 to object 3: [545.0, 68.0, 712.0, 159.0]
# Adding box for frame 22 to object 3: [394.0, 188.0, 495.0, 276.0]
# Adding box for frame 23 to object 3: [364.0, 77.0, 506.0, 150.0]
# Adding box for frame 24 to object 3: [364.0, 77.0, 506.0, 150.0]
# Adding box for frame 25 to object 3: [364.0, 77.0, 506.0, 150.0]
# Adding box for frame 26 to object 3: [364.0, 77.0, 506.0, 150.0]
# Adding box for frame 27 to object 3: [364.0, 77.0, 506.0, 150.0]
# Adding box for frame 28 to object 3: [364.0, 77.0, 506.0, 150.0]
# Adding box for frame 29 to object 3: [364.0, 77.0, 506.0, 150.0]
# Adding box for frame 30 to object 3: [364.0, 77.0, 506.0, 150.0]
# Adding box for frame 31 to object 3: [515.0, 281.0, 725.0, 336.0]
# Adding box for frame 32 to object 3: [515.0, 281.0, 725.0, 336.0]
# Adding box for frame 33 to object 3: [600.0, 102.0, 650.0, 186.0]
# Adding box for frame 34 to object 3: [323.0, 181.0, 408.0, 271.0]
# Adding box for frame 35 to object 3: [579.0, 145.0, 641.0, 201.0]
# Adding box for frame 36 to object 3: [488.0, 181.0, 527.0, 239.0]
# Adding box for frame 37 to object 3: [494.0, 202.0, 531.0, 253.0]
# Adding box for frame 38 to object 3: [533.0, 176.0, 576.0, 225.0]
# Adding box for frame 39 to object 3: [469.0, 163.0, 504.0, 215.0]
# Adding box for frame 40 to object 3: [469.0, 163.0, 504.0, 215.0]
# Adding box for frame 41 to object 3: [496.0, 182.0, 512.0, 219.0]
# Adding box for frame 42 to object 3: [395.0, 119.0, 422.0, 176.0]
# Adding box for frame 43 to object 3: [495.0, 159.0, 525.0, 207.0]
# Adding box for frame 44 to object 3: [485.0, 143.0, 519.0, 188.0]
# Adding box for frame 45 to object 3: [608.0, 242.0, 640.0, 283.0]
# Adding box for frame 46 to object 3: [608.0, 242.0, 640.0, 283.0]
# Adding box for frame 47 to object 3: [608.0, 242.0, 640.0, 283.0]
# Adding box for frame 48 to object 3: [608.0, 242.0, 640.0, 283.0]
# Adding box for frame 49 to object 3: [608.0, 242.0, 640.0, 283.0]
# Adding box for frame 50 to object 3: [608.0, 242.0, 640.0, 283.0]
# Adding box for frame 51 to object 3: [608.0, 242.0, 640.0, 283.0]
# Adding box for frame 52 to object 3: [608.0, 242.0, 640.0, 283.0]
# Adding box for frame 53 to object 3: [608.0, 242.0, 640.0, 283.0]
# Updating object 3 with 36 boxes from mask propagation
# Error during propagation for object 3: 3
# Propagation error traceback: Traceback (most recent call last):
#   File "/home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/src/object_tracking_pipeline.py", line 1240, in process_video_separate_objects
#     self.tracked_objects[obj_id]["boxes"] = all_boxes
#     ~~~~~~~~~~~~~~~~~~~~^^^^^^^^
# KeyError: 3

# Saving per-object segmentation visualizations based on mask quality...
#   Object #1 (Time Stick) mask quality statistics:
#     Frames processed: 51
#     Frames saved: 16
#     High quality masks (>0 pixels): 16
#     Low quality masks (1-0 pixels): 0
#     Empty masks (0 pixels): 35 (not saved)
#   Object #2 (TVA Uniform) mask quality statistics:
#     Frames processed: 41
#     Frames saved: 10
#     High quality masks (>0 pixels): 10
#     Low quality masks (1-0 pixels): 0
#     Empty masks (0 pixels): 31 (not saved)
#   Object #3 (Sylvie's horned headpiece) mask quality statistics:
#     Frames processed: 36
#     Frames saved: 18
#     High quality masks (>0 pixels): 18
#     Low quality masks (1-0 pixels): 0
#     Empty masks (0 pixels): 18 (not saved)
# All per-object mask visualizations saved to cd_fsod_results/object_masks
# Saving first detection frames for each object...
# Using detection files from: /home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/data/detections/Scenes 061-080__265H-2-_20230815215828529
#   Object #1 (Time Stick) first detected in frame 3, using 3.json
#     JSON file contains 51 detections
#     Sample detection for Time Stick: confidence=0.9932, coordinates=[453, 224, 643, 473]
#   Saved first detection frame for object #1 (Time Stick)
#   Object #2 (TVA Uniform) first detected in frame 13, using 13.json
#     JSON file contains 36 detections
#     Sample detection for TVA Uniform: confidence=0.9669, coordinates=[374, 115, 604, 490]
#   Saved first detection frame for object #2 (TVA Uniform)
#   Object #3 (Sylvie's horned headpiece) first detected in frame 18, using 18.json
#     JSON file contains 51 detections
#     Sample detection for Sylvie's horned headpiece: confidence=0.9856, coordinates=[322, 128, 481, 202]
#   Saved first detection frame for object #3 (Sylvie's horned headpiece)
# All first detection frames saved to cd_fsod_results/first_detections
# Creating object-to-frame mapping with quality metrics...
# Object tracking quality statistics:
#   Total objects tracked: 3
#   Total high quality frames (>0 pixels): 44
#   Total low quality frames (1-0 pixels): 0
#   Total empty mask frames (0 pixels): 84 (not saved)
#   Total saved frames (non-empty masks): 44
# Saved tracking summary to cd_fsod_results/tracking_summary.json
# Verifying consistency with saved mask images...
# Verification successful: All objects' masks are consistent with the mapping
# Saved object-to-frame mapping to cd_fsod_results/object_frame_mapping.json
# All results saved to: cd_fsod_results
# 2025-05-13 22:33:56,950 - INFO - Video processing completed in 111.39 seconds
# INFO:cd_fsod_test:Video processing completed in 111.39 seconds
# 2025-05-13 22:33:56,950 - DEBUG - First detection details:
# DEBUG:cd_fsod_test:First detection details:
# 2025-05-13 22:33:56,950 - DEBUG -   Object #1 (Time Stick) first detected at frame 3
# DEBUG:cd_fsod_test:  Object #1 (Time Stick) first detected at frame 3
# 2025-05-13 22:33:56,950 - DEBUG -     From 3.json: {'coordinates': [453, 224, 643, 473], 'label': 'Time Stick', 'confidence': 0.9931594133377075}
# DEBUG:cd_fsod_test:    From 3.json: {'coordinates': [453, 224, 643, 473], 'label': 'Time Stick', 'confidence': 0.9931594133377075}
# 2025-05-13 22:33:56,950 - DEBUG -   Object #2 (TVA Uniform) first detected at frame 13
# DEBUG:cd_fsod_test:  Object #2 (TVA Uniform) first detected at frame 13
# 2025-05-13 22:33:56,951 - DEBUG -     From 13.json: {'coordinates': [374, 115, 604, 490], 'label': 'TVA Uniform', 'confidence': 0.966902494430542}
# DEBUG:cd_fsod_test:    From 13.json: {'coordinates': [374, 115, 604, 490], 'label': 'TVA Uniform', 'confidence': 0.966902494430542}
# 2025-05-13 22:33:56,951 - DEBUG -   Object #3 (Sylvie's horned headpiece) first detected at frame 18
# DEBUG:cd_fsod_test:  Object #3 (Sylvie's horned headpiece) first detected at frame 18
# 2025-05-13 22:33:56,951 - DEBUG -     From 18.json: {'coordinates': [322, 128, 481, 202], 'label': "Sylvie's horned headpiece", 'confidence': 0.9855985045433044}
# DEBUG:cd_fsod_test:    From 18.json: {'coordinates': [322, 128, 481, 202], 'label': "Sylvie's horned headpiece", 'confidence': 0.9855985045433044}
# 2025-05-13 22:33:56,951 - INFO - Generated 6 output files
# INFO:cd_fsod_test:Generated 6 output files
# 2025-05-13 22:33:56,951 - DEBUG - Output files: ['object_frame_mapping.json', 'tracking_summary.json', 'first_detections', 'cd_fsod_test_20250513_223201.log', 'tracking_results.json', 'object_masks']
# DEBUG:cd_fsod_test:Output files: ['object_frame_mapping.json', 'tracking_summary.json', 'first_detections', 'cd_fsod_test_20250513_223201.log', 'tracking_results.json', 'object_masks']
# 2025-05-13 22:33:56,951 - INFO - ================================================================================
# INFO:cd_fsod_test:================================================================================
# 2025-05-13 22:33:56,951 - INFO - CD-FSOD Integration Test Completed Successfully
# INFO:cd_fsod_test:CD-FSOD Integration Test Completed Successfully
# 2025-05-13 22:33:56,951 - INFO - Total processing time: 115.18 seconds
# INFO:cd_fsod_test:Total processing time: 115.18 seconds
# 2025-05-13 22:33:56,952 - INFO - Results saved to: ./cd_fsod_results
# INFO:cd_fsod_test:Results saved to: ./cd_fsod_results
# 2025-05-13 22:33:56,952 - INFO - ================================================================================
# INFO:cd_fsod_test:================================================================================