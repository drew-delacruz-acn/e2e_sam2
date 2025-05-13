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

# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 2: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 32, found 1 objects
# propagate in video:  49%|██████████████████████████████████████████████████████████████▍                                                                 | 20/41 [00:15<00:17,  1.21it/s]Processing item 21: frame=33, objects=[2]
# Filtering objects to track: [2]
# After filtering: 1 objects remain
# Processing object 2 in frame 33
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 2: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 33, found 1 objects
# propagate in video:  51%|█████████████████████████████████████████████████████████████████▌                                                              | 21/41 [00:15<00:16,  1.21it/s]Processing item 22: frame=34, objects=[2]
# Filtering objects to track: [2]
# After filtering: 1 objects remain
# Processing object 2 in frame 34
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 2: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 34, found 1 objects
# propagate in video:  54%|████████████████████████████████████████████████████████████████████▋                                                           | 22/41 [00:16<00:15,  1.21it/s]Processing item 23: frame=35, objects=[2]
# Filtering objects to track: [2]
# After filtering: 1 objects remain
# Processing object 2 in frame 35
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 2: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 35, found 1 objects
# propagate in video:  56%|███████████████████████████████████████████████████████████████████████▊                                                        | 23/41 [00:17<00:14,  1.21it/s]Processing item 24: frame=36, objects=[2]
# Filtering objects to track: [2]
# After filtering: 1 objects remain
# Processing object 2 in frame 36
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 2: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 36, found 1 objects
# propagate in video:  59%|██████████████████████████████████████████████████████████████████████████▉                                                     | 24/41 [00:18<00:14,  1.21it/s]Processing item 25: frame=37, objects=[2]
# Filtering objects to track: [2]
# After filtering: 1 objects remain
# Processing object 2 in frame 37
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 2: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 37, found 1 objects
# propagate in video:  61%|██████████████████████████████████████████████████████████████████████████████                                                  | 25/41 [00:19<00:13,  1.21it/s]Processing item 26: frame=38, objects=[2]
# Filtering objects to track: [2]
# After filtering: 1 objects remain
# Processing object 2 in frame 38
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 2: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 38, found 1 objects
# propagate in video:  63%|█████████████████████████████████████████████████████████████████████████████████▏                                              | 26/41 [00:20<00:12,  1.21it/s]Processing item 27: frame=39, objects=[2]
# Filtering objects to track: [2]
# After filtering: 1 objects remain
# Processing object 2 in frame 39
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 2: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 39, found 1 objects
# propagate in video:  66%|████████████████████████████████████████████████████████████████████████████████████▎                                           | 27/41 [00:20<00:11,  1.21it/s]Processing item 28: frame=40, objects=[2]
# Filtering objects to track: [2]
# After filtering: 1 objects remain
# Processing object 2 in frame 40
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 2: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 40, found 1 objects
# propagate in video:  68%|███████████████████████████████████████████████████████████████████████████████████████▍                                        | 28/41 [00:21<00:10,  1.21it/s]Processing item 29: frame=41, objects=[2]
# Filtering objects to track: [2]
# After filtering: 1 objects remain
# Processing object 2 in frame 41
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 2: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 41, found 1 objects
# propagate in video:  71%|██████████████████████████████████████████████████████████████████████████████████████████▌                                     | 29/41 [00:22<00:09,  1.21it/s]Processing item 30: frame=42, objects=[2]
# Filtering objects to track: [2]
# After filtering: 1 objects remain
# Processing object 2 in frame 42
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 2: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 42, found 1 objects
# propagate in video:  73%|█████████████████████████████████████████████████████████████████████████████████████████████▋                                  | 30/41 [00:23<00:09,  1.21it/s]Processing item 31: frame=43, objects=[2]
# Filtering objects to track: [2]
# After filtering: 1 objects remain
# Processing object 2 in frame 43
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 2: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 43, found 1 objects
# propagate in video:  76%|████████████████████████████████████████████████████████████████████████████████████████████████▊                               | 31/41 [00:24<00:08,  1.21it/s]Processing item 32: frame=44, objects=[2]
# Filtering objects to track: [2]
# After filtering: 1 objects remain
# Processing object 2 in frame 44
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 2: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 44, found 1 objects
# propagate in video:  78%|███████████████████████████████████████████████████████████████████████████████████████████████████▉                            | 32/41 [00:25<00:07,  1.21it/s]Processing item 33: frame=45, objects=[2]
# Filtering objects to track: [2]
# After filtering: 1 objects remain
# Processing object 2 in frame 45
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 2: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 45, found 1 objects
# propagate in video:  80%|███████████████████████████████████████████████████████████████████████████████████████████████████████                         | 33/41 [00:25<00:06,  1.21it/s]Processing item 34: frame=46, objects=[2]
# Filtering objects to track: [2]
# After filtering: 1 objects remain
# Processing object 2 in frame 46
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 2: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 46, found 1 objects
# propagate in video:  83%|██████████████████████████████████████████████████████████████████████████████████████████████████████████▏                     | 34/41 [00:26<00:05,  1.21it/s]Processing item 35: frame=47, objects=[2]
# Filtering objects to track: [2]
# After filtering: 1 objects remain
# Processing object 2 in frame 47
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 2: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 47, found 1 objects
# propagate in video:  85%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                  | 35/41 [00:27<00:04,  1.21it/s]Processing item 36: frame=48, objects=[2]
# Filtering objects to track: [2]
# After filtering: 1 objects remain
# Processing object 2 in frame 48
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 2: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 48, found 1 objects
# propagate in video:  88%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍               | 36/41 [00:28<00:04,  1.21it/s]Processing item 37: frame=49, objects=[2]
# Filtering objects to track: [2]
# After filtering: 1 objects remain
# Processing object 2 in frame 49
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 2: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 49, found 1 objects
# propagate in video:  90%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌            | 37/41 [00:29<00:03,  1.21it/s]Processing item 38: frame=50, objects=[2]
# Filtering objects to track: [2]
# After filtering: 1 objects remain
# Processing object 2 in frame 50
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 2: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 50, found 1 objects
# propagate in video:  93%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋         | 38/41 [00:29<00:02,  1.21it/s]Processing item 39: frame=51, objects=[2]
# Filtering objects to track: [2]
# After filtering: 1 objects remain
# Processing object 2 in frame 51
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 2: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 51, found 1 objects
# propagate in video:  95%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊      | 39/41 [00:30<00:01,  1.21it/s]Processing item 40: frame=52, objects=[2]
# Filtering objects to track: [2]
# After filtering: 1 objects remain
# Processing object 2 in frame 52
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 2: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 52, found 1 objects
# propagate in video:  98%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉   | 40/41 [00:31<00:00,  1.21it/s]Processing item 41: frame=53, objects=[2]
# Filtering objects to track: [2]
# After filtering: 1 objects remain
# Processing object 2 in frame 53
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 2: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 53, found 1 objects
# propagate in video: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41/41 [00:32<00:00,  1.26it/s]
# Finished propagation, processed 41 frames, found 41 frames with objects
# DEBUG: propagate_masks returned type: <class 'tuple'>
# DEBUG: propagate_masks tuple length: 2
# DEBUG: Unpacked 2-element tuple - segments (41 frames) and boxes
# DEBUG: Segments contains 41 frames
# DEBUG: Frame 13 has 1 objects
# DEBUG: Frame 14 has 1 objects
# DEBUG: Frame 15 has 1 objects
# Successfully propagated masks for object 2, available in 41 frames

# ==== Processing object 3 (Sylvie's horned headpiece) separately ====
# Resetting SAM2 state for object 3...
# frame loading (JPEG): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 54/54 [00:01<00:00, 39.83it/s]
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
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [324.0, 124.0, 468.0, 197.0]
# Processed frame 18, found 1 objects
# Processing item 2: frame=19, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 19
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [541.0, 161.0, 656.0, 251.0]
# Processed frame 19, found 1 objects
# propagate in video:   6%|███████▏                                                                                                                         | 2/36 [00:00<00:11,  3.00it/s]Processing item 3: frame=20, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 20
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [545.0, 68.0, 712.0, 159.0]
# Processed frame 20, found 1 objects
# propagate in video:   8%|██████████▊                                                                                                                      | 3/36 [00:01<00:15,  2.08it/s]Processing item 4: frame=21, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 21
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 3: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 21, found 1 objects
# propagate in video:  11%|██████████████▎                                                                                                                  | 4/36 [00:02<00:18,  1.76it/s]Processing item 5: frame=22, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 22
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [394.0, 188.0, 495.0, 276.0]
# Processed frame 22, found 1 objects
# propagate in video:  14%|█████████████████▉                                                                                                               | 5/36 [00:02<00:19,  1.59it/s]Processing item 6: frame=23, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 23
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [364.0, 77.0, 506.0, 150.0]
# Processed frame 23, found 1 objects
# propagate in video:  17%|█████████████████████▌                                                                                                           | 6/36 [00:03<00:20,  1.48it/s]Processing item 7: frame=24, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 24
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 3: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 24, found 1 objects
# propagate in video:  19%|█████████████████████████                                                                                                        | 7/36 [00:04<00:20,  1.40it/s]Processing item 8: frame=25, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 25
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 3: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 25, found 1 objects
# propagate in video:  22%|████████████████████████████▋                                                                                                    | 8/36 [00:05<00:21,  1.33it/s]Processing item 9: frame=26, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 26
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 3: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 26, found 1 objects
# propagate in video:  25%|████████████████████████████████▎                                                                                                | 9/36 [00:06<00:20,  1.29it/s]Processing item 10: frame=27, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 27
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 3: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 27, found 1 objects
# propagate in video:  28%|███████████████████████████████████▌                                                                                            | 10/36 [00:06<00:20,  1.26it/s]Processing item 11: frame=28, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 28
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 3: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 28, found 1 objects
# propagate in video:  31%|███████████████████████████████████████                                                                                         | 11/36 [00:07<00:20,  1.24it/s]Processing item 12: frame=29, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 29
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 3: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 29, found 1 objects
# propagate in video:  33%|██████████████████████████████████████████▋                                                                                     | 12/36 [00:08<00:19,  1.23it/s]Processing item 13: frame=30, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 30
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 3: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 30, found 1 objects
# propagate in video:  36%|██████████████████████████████████████████████▏                                                                                 | 13/36 [00:09<00:18,  1.22it/s]Processing item 14: frame=31, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 31
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [515.0, 281.0, 725.0, 336.0]
# Processed frame 31, found 1 objects
# propagate in video:  39%|█████████████████████████████████████████████████▊                                                                              | 14/36 [00:10<00:18,  1.22it/s]Processing item 15: frame=32, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 32
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 3: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 32, found 1 objects
# propagate in video:  42%|█████████████████████████████████████████████████████▎                                                                          | 15/36 [00:11<00:17,  1.21it/s]Processing item 16: frame=33, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 33
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [600.0, 102.0, 650.0, 186.0]
# Processed frame 33, found 1 objects
# propagate in video:  44%|████████████████████████████████████████████████████████▉                                                                       | 16/36 [00:11<00:16,  1.21it/s]Processing item 17: frame=34, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 34
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [323.0, 181.0, 408.0, 271.0]
# Processed frame 34, found 1 objects
# propagate in video:  47%|████████████████████████████████████████████████████████████▍                                                                   | 17/36 [00:12<00:15,  1.21it/s]Processing item 18: frame=35, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 35
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [579.0, 145.0, 641.0, 201.0]
# Processed frame 35, found 1 objects
# propagate in video:  50%|████████████████████████████████████████████████████████████████                                                                | 18/36 [00:13<00:14,  1.21it/s]Processing item 19: frame=36, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 36
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [488.0, 181.0, 527.0, 239.0]
# Processed frame 36, found 1 objects
# propagate in video:  53%|███████████████████████████████████████████████████████████████████▌                                                            | 19/36 [00:14<00:14,  1.21it/s]Processing item 20: frame=37, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 37
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [494.0, 202.0, 531.0, 253.0]
# Processed frame 37, found 1 objects
# propagate in video:  56%|███████████████████████████████████████████████████████████████████████                                                         | 20/36 [00:15<00:13,  1.21it/s]Processing item 21: frame=38, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 38
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [533.0, 176.0, 576.0, 225.0]
# Processed frame 38, found 1 objects
# propagate in video:  58%|██████████████████████████████████████████████████████████████████████████▋                                                     | 21/36 [00:15<00:12,  1.21it/s]Processing item 22: frame=39, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 39
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [469.0, 163.0, 504.0, 215.0]
# Processed frame 39, found 1 objects
# propagate in video:  61%|██████████████████████████████████████████████████████████████████████████████▏                                                 | 22/36 [00:16<00:11,  1.21it/s]Processing item 23: frame=40, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 40
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 3: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 40, found 1 objects
# propagate in video:  64%|█████████████████████████████████████████████████████████████████████████████████▊                                              | 23/36 [00:17<00:10,  1.21it/s]Processing item 24: frame=41, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 41
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [496.0, 182.0, 512.0, 219.0]
# Processed frame 41, found 1 objects
# propagate in video:  67%|█████████████████████████████████████████████████████████████████████████████████████▎                                          | 24/36 [00:18<00:09,  1.21it/s]Processing item 25: frame=42, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 42
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [395.0, 119.0, 422.0, 176.0]
# Processed frame 42, found 1 objects
# propagate in video:  69%|████████████████████████████████████████████████████████████████████████████████████████▉                                       | 25/36 [00:19<00:09,  1.21it/s]Processing item 26: frame=43, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 43
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [495.0, 159.0, 525.0, 207.0]
# Processed frame 43, found 1 objects
# propagate in video:  72%|████████████████████████████████████████████████████████████████████████████████████████████▍                                   | 26/36 [00:20<00:08,  1.21it/s]Processing item 27: frame=44, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 44
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [485.0, 143.0, 519.0, 188.0]
# Processed frame 44, found 1 objects
# propagate in video:  75%|████████████████████████████████████████████████████████████████████████████████████████████████                                | 27/36 [00:20<00:07,  1.21it/s]Processing item 28: frame=45, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 45
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [608.0, 242.0, 640.0, 283.0]
# Processed frame 45, found 1 objects
# propagate in video:  78%|███████████████████████████████████████████████████████████████████████████████████████████████████▌                            | 28/36 [00:21<00:06,  1.21it/s]Processing item 29: frame=46, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 46
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 3: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 46, found 1 objects
# propagate in video:  81%|███████████████████████████████████████████████████████████████████████████████████████████████████████                         | 29/36 [00:22<00:05,  1.21it/s]Processing item 30: frame=47, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 47
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 3: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 47, found 1 objects
# propagate in video:  83%|██████████████████████████████████████████████████████████████████████████████████████████████████████████▋                     | 30/36 [00:23<00:04,  1.21it/s]Processing item 31: frame=48, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 48
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 3: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 48, found 1 objects
# propagate in video:  86%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                 | 31/36 [00:24<00:04,  1.21it/s]Processing item 32: frame=49, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 49
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 3: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 49, found 1 objects
# propagate in video:  89%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊              | 32/36 [00:25<00:03,  1.20it/s]Processing item 33: frame=50, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 50
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 3: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 50, found 1 objects
# propagate in video:  92%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎          | 33/36 [00:25<00:02,  1.20it/s]Processing item 34: frame=51, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 51
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 3: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 51, found 1 objects
# propagate in video:  94%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉       | 34/36 [00:26<00:01,  1.20it/s]Processing item 35: frame=52, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 52
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 3: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 52, found 1 objects
# propagate in video:  97%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍   | 35/36 [00:27<00:00,  1.21it/s]Processing item 36: frame=53, objects=[3]
# Filtering objects to track: [3]
# After filtering: 1 objects remain
# Processing object 3 in frame 53
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 3: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 53, found 1 objects
# propagate in video: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 36/36 [00:28<00:00,  1.27it/s]
# Finished propagation, processed 36 frames, found 36 frames with objects
# DEBUG: propagate_masks returned type: <class 'tuple'>
# DEBUG: propagate_masks tuple length: 2
# DEBUG: Unpacked 2-element tuple - segments (36 frames) and boxes
# DEBUG: Segments contains 36 frames
# DEBUG: Frame 18 has 1 objects
# DEBUG: Frame 19 has 1 objects
# DEBUG: Frame 20 has 1 objects
# Successfully propagated masks for object 3, available in 36 frames

# ==== Processing object 4 (Sylvie's horned headpiece) separately ====
# Resetting SAM2 state for object 4...
# frame loading (JPEG): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 54/54 [00:01<00:00, 40.16it/s]
# Set video from directory: /home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/data/frames/Scenes 061-080__265H-2-_20230815215828529
# Adding box for object 4 at frame 33
# Running propagation for object 4...
# Starting propagate_masks with objects_to_track=[4]
# Using predictor type: SAM2VideoPredictor
# Propagate method: propagate_in_video from sam2.sam2_video_predictor
# Iterator type: generator
# propagate in video:   0%|                                                                                                                                         | 0/21 [00:00<?, ?it/s]First item type: tuple
# First item value: (33, [4], tensor([[[[-18.6186, -18.6186, -19.6051,  ..., -23.5010, -24.4846, -24.4846],
#           [-19.9732, -19.9732, -20.5185,  ..., -22.2657, -22.5937, -22.5937],
#           [-23.0149, -23.0149, -22.5695,  ..., -19.4916, -18.3476, -18.3476],
#           ...,
#           [-22.0225, -22.0225, -21.9425,  ..., -22.6875, -22.2009, -22.2009],
#           [-21.7383, -21.7383, -21.6572,  ..., -21.0145, -19.8140, -19.8140],
#           [-21.6117, -21.6117, -21.5302,  ..., -20.2694, -18.7511, -18.7511]]]],
#        device='cuda:0'))
# First item tuple length: 3
#   Element 0: type=int, value=33
#   Element 1: type=list, value=[4]
#   Element 2: type=Tensor, value=tensor([[[[-18.6186, -18.6186, -19.6051,  ..., -23.5010, -24.4846, -24.4846],
#           [-19.9732, -19.9732, -20.5185,  ..., -22.2657, -22.5937, -22.5937],
#           [-23.0149, -23.0149, -22.5695,  ..., -19.4916, -18.3476, -18.3476],
#           ...,
#           [-22.0225, -22.0225, -21.9425,  ..., -22.6875, -22.2009, -22.2009],
#           [-21.7383, -21.7383, -21.6572,  ..., -21.0145, -19.8140, -19.8140],
#           [-21.6117, -21.6117, -21.5302,  ..., -20.2694, -18.7511, -18.7511]]]],
#        device='cuda:0')
# Processing item 1: frame=33, objects=[4]
# Filtering objects to track: [4]
# After filtering: 1 objects remain
# Processing object 4 in frame 33
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [592.0, 102.0, 659.0, 198.0]
# Processed frame 33, found 1 objects
# Processing item 2: frame=34, objects=[4]
# Filtering objects to track: [4]
# After filtering: 1 objects remain
# Processing object 4 in frame 34
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [342.0, 180.0, 408.0, 278.0]
# Processed frame 34, found 1 objects
# propagate in video:  10%|████████████▎                                                                                                                    | 2/21 [00:00<00:06,  2.98it/s]Processing item 3: frame=35, objects=[4]
# Filtering objects to track: [4]
# After filtering: 1 objects remain
# Processing object 4 in frame 35
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [599.0, 145.0, 642.0, 202.0]
# Processed frame 35, found 1 objects
# propagate in video:  14%|██████████████████▍                                                                                                              | 3/21 [00:01<00:08,  2.07it/s]Processing item 4: frame=36, objects=[4]
# Filtering objects to track: [4]
# After filtering: 1 objects remain
# Processing object 4 in frame 36
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [481.0, 181.0, 527.0, 261.0]
# Processed frame 36, found 1 objects
# propagate in video:  19%|████████████████████████▌                                                                                                        | 4/21 [00:02<00:09,  1.75it/s]Processing item 5: frame=37, objects=[4]
# Filtering objects to track: [4]
# After filtering: 1 objects remain
# Processing object 4 in frame 37
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [494.0, 202.0, 531.0, 254.0]
# Processed frame 37, found 1 objects
# propagate in video:  24%|██████████████████████████████▋                                                                                                  | 5/21 [00:02<00:10,  1.58it/s]Processing item 6: frame=38, objects=[4]
# Filtering objects to track: [4]
# After filtering: 1 objects remain
# Processing object 4 in frame 38
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [534.0, 176.0, 576.0, 225.0]
# Processed frame 38, found 1 objects
# propagate in video:  29%|████████████████████████████████████▊                                                                                            | 6/21 [00:03<00:10,  1.47it/s]Processing item 7: frame=39, objects=[4]
# Filtering objects to track: [4]
# After filtering: 1 objects remain
# Processing object 4 in frame 39
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [465.0, 163.0, 505.0, 219.0]
# Processed frame 39, found 1 objects
# propagate in video:  33%|███████████████████████████████████████████                                                                                      | 7/21 [00:04<00:10,  1.39it/s]Processing item 8: frame=40, objects=[4]
# Filtering objects to track: [4]
# After filtering: 1 objects remain
# Processing object 4 in frame 40
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 4: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 40, found 1 objects
# propagate in video:  38%|█████████████████████████████████████████████████▏                                                                               | 8/21 [00:05<00:09,  1.33it/s]Processing item 9: frame=41, objects=[4]
# Filtering objects to track: [4]
# After filtering: 1 objects remain
# Processing object 4 in frame 41
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [495.0, 182.0, 512.0, 219.0]
# Processed frame 41, found 1 objects
# propagate in video:  43%|███████████████████████████████████████████████████████▎                                                                         | 9/21 [00:06<00:09,  1.28it/s]Processing item 10: frame=42, objects=[4]
# Filtering objects to track: [4]
# After filtering: 1 objects remain
# Processing object 4 in frame 42
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [388.0, 119.0, 422.0, 205.0]
# Processed frame 42, found 1 objects
# propagate in video:  48%|████████████████████████████████████████████████████████████▉                                                                   | 10/21 [00:06<00:08,  1.26it/s]Processing item 11: frame=43, objects=[4]
# Filtering objects to track: [4]
# After filtering: 1 objects remain
# Processing object 4 in frame 43
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [485.0, 159.0, 525.0, 209.0]
# Processed frame 43, found 1 objects
# propagate in video:  52%|███████████████████████████████████████████████████████████████████                                                             | 11/21 [00:07<00:08,  1.24it/s]Processing item 12: frame=44, objects=[4]
# Filtering objects to track: [4]
# After filtering: 1 objects remain
# Processing object 4 in frame 44
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [485.0, 142.0, 520.0, 189.0]
# Processed frame 44, found 1 objects
# propagate in video:  57%|█████████████████████████████████████████████████████████████████████████▏                                                      | 12/21 [00:08<00:07,  1.23it/s]Processing item 13: frame=45, objects=[4]
# Filtering objects to track: [4]
# After filtering: 1 objects remain
# Processing object 4 in frame 45
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Generated box: [606.0, 242.0, 640.0, 283.0]
# Processed frame 45, found 1 objects
# propagate in video:  62%|███████████████████████████████████████████████████████████████████████████████▏                                                | 13/21 [00:09<00:06,  1.22it/s]Processing item 14: frame=46, objects=[4]
# Filtering objects to track: [4]
# After filtering: 1 objects remain
# Processing object 4 in frame 46
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 4: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 46, found 1 objects
# propagate in video:  67%|█████████████████████████████████████████████████████████████████████████████████████▎                                          | 14/21 [00:10<00:05,  1.22it/s]Processing item 15: frame=47, objects=[4]
# Filtering objects to track: [4]
# After filtering: 1 objects remain
# Processing object 4 in frame 47
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 4: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 47, found 1 objects
# propagate in video:  71%|███████████████████████████████████████████████████████████████████████████████████████████▍                                    | 15/21 [00:11<00:04,  1.22it/s]Processing item 16: frame=48, objects=[4]
# Filtering objects to track: [4]
# After filtering: 1 objects remain
# Processing object 4 in frame 48
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 4: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 48, found 1 objects
# propagate in video:  76%|█████████████████████████████████████████████████████████████████████████████████████████████████▌                              | 16/21 [00:11<00:04,  1.21it/s]Processing item 17: frame=49, objects=[4]
# Filtering objects to track: [4]
# After filtering: 1 objects remain
# Processing object 4 in frame 49
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 4: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 49, found 1 objects
# propagate in video:  81%|███████████████████████████████████████████████████████████████████████████████████████████████████████▌                        | 17/21 [00:12<00:03,  1.21it/s]Processing item 18: frame=50, objects=[4]
# Filtering objects to track: [4]
# After filtering: 1 objects remain
# Processing object 4 in frame 50
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 4: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 50, found 1 objects
# propagate in video:  86%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                  | 18/21 [00:13<00:02,  1.21it/s]Processing item 19: frame=51, objects=[4]
# Filtering objects to track: [4]
# After filtering: 1 objects remain
# Processing object 4 in frame 51
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 4: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 51, found 1 objects
# propagate in video:  90%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊            | 19/21 [00:14<00:01,  1.21it/s]Processing item 20: frame=52, objects=[4]
# Filtering objects to track: [4]
# After filtering: 1 objects remain
# Processing object 4 in frame 52
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 4: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 52, found 1 objects
# propagate in video:  95%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉      | 20/21 [00:15<00:00,  1.21it/s]Processing item 21: frame=53, objects=[4]
# Filtering objects to track: [4]
# After filtering: 1 objects remain
# Processing object 4 in frame 53
# Mask logit type: Tensor, shape: torch.Size([1, 540, 960])
# Converted tensor mask, shape: torch.Size([1, 540, 960])
# Creating bounding box from mask shape: torch.Size([1, 540, 960])
# Expanded mask shape: torch.Size([1, 540, 960])
# Error creating box for object 4: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# Processed frame 53, found 1 objects
# propagate in video: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21/21 [00:16<00:00,  1.31it/s]
# Finished propagation, processed 21 frames, found 21 frames with objects
# DEBUG: propagate_masks returned type: <class 'tuple'>
# DEBUG: propagate_masks tuple length: 2
# DEBUG: Unpacked 2-element tuple - segments (21 frames) and boxes
# DEBUG: Segments contains 21 frames
# DEBUG: Frame 33 has 1 objects
# DEBUG: Frame 34 has 1 objects
# DEBUG: Frame 35 has 1 objects
# Successfully propagated masks for object 4, available in 21 frames
# Saving per-object segmentation visualizations...
# Processing object 1 visible from frame 3 to 3
#   Saved 1 visualizations for object #1 (Time Stick)
# Processing object 2 visible from frame 13 to 13
#   Saved 1 visualizations for object #2 (TVA Uniform)
# Processing object 3 visible from frame 18 to 43
#   Saved 26 visualizations for object #3 (Sylvie's horned headpiece)
# Processing object 4 visible from frame 33 to 33
#   Saved 1 visualizations for object #4 (Sylvie's horned headpiece)
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
#   Object #4 (Sylvie's horned headpiece) first detected in frame 33, using 33.json
#     JSON file contains 42 detections
#     Sample detection for Sylvie's horned headpiece: confidence=0.9580, coordinates=[582, 104, 671, 201]
#   Saved first detection frame for object #4 (Sylvie's horned headpiece)
# All first detection frames saved to cd_fsod_results/first_detections
# Creating object-to-frame mapping...
# Verifying consistency with saved mask images...
# Verification successful: All objects' masks are consistent with the mapping
# Saved object-to-frame mapping to cd_fsod_results/object_frame_mapping.json
# All results saved to: cd_fsod_results
# 2025-05-13 19:34:41,334 - INFO - Video processing completed in 130.45 seconds
# INFO:cd_fsod_test:Video processing completed in 130.45 seconds
# 2025-05-13 19:34:41,334 - DEBUG - First detection details:
# DEBUG:cd_fsod_test:First detection details:
# 2025-05-13 19:34:41,334 - DEBUG -   Object #1 (Time Stick) first detected at frame 3
# DEBUG:cd_fsod_test:  Object #1 (Time Stick) first detected at frame 3
# 2025-05-13 19:34:41,335 - DEBUG -     From 3.json: {'coordinates': [453, 224, 643, 473], 'label': 'Time Stick', 'confidence': 0.9931594133377075}
# DEBUG:cd_fsod_test:    From 3.json: {'coordinates': [453, 224, 643, 473], 'label': 'Time Stick', 'confidence': 0.9931594133377075}
# 2025-05-13 19:34:41,335 - DEBUG -   Object #2 (TVA Uniform) first detected at frame 13
# DEBUG:cd_fsod_test:  Object #2 (TVA Uniform) first detected at frame 13
# 2025-05-13 19:34:41,335 - DEBUG -     From 13.json: {'coordinates': [374, 115, 604, 490], 'label': 'TVA Uniform', 'confidence': 0.966902494430542}
# DEBUG:cd_fsod_test:    From 13.json: {'coordinates': [374, 115, 604, 490], 'label': 'TVA Uniform', 'confidence': 0.966902494430542}
# 2025-05-13 19:34:41,335 - DEBUG -   Object #3 (Sylvie's horned headpiece) first detected at frame 18
# DEBUG:cd_fsod_test:  Object #3 (Sylvie's horned headpiece) first detected at frame 18
# 2025-05-13 19:34:41,336 - DEBUG -     From 18.json: {'coordinates': [322, 128, 481, 202], 'label': "Sylvie's horned headpiece", 'confidence': 0.9855985045433044}
# DEBUG:cd_fsod_test:    From 18.json: {'coordinates': [322, 128, 481, 202], 'label': "Sylvie's horned headpiece", 'confidence': 0.9855985045433044}
# 2025-05-13 19:34:41,336 - DEBUG -   Object #4 (Sylvie's horned headpiece) first detected at frame 33
# DEBUG:cd_fsod_test:  Object #4 (Sylvie's horned headpiece) first detected at frame 33
# 2025-05-13 19:34:41,336 - DEBUG -     From 33.json: {'coordinates': [582, 104, 671, 201], 'label': "Sylvie's horned headpiece", 'confidence': 0.9580428600311279}
# DEBUG:cd_fsod_test:    From 33.json: {'coordinates': [582, 104, 671, 201], 'label': "Sylvie's horned headpiece", 'confidence': 0.9580428600311279}
# 2025-05-13 19:34:41,337 - INFO - Generated 68 output files
# INFO:cd_fsod_test:Generated 68 output files
# 2025-05-13 19:34:41,337 - DEBUG - Output files: ['frame_0049.jpg', 'frame_0045.jpg', 'cd_fsod_test_20250513_193227.log', 'frame_0014.jpg', 'frame_0025.jpg', 'frame_0051.jpg', 'frame_0039.jpg', 'frame_0030.jpg', 'frame_0008.jpg', 'frame_0002.jpg']
# DEBUG:cd_fsod_test:Output files: ['frame_0049.jpg', 'frame_0045.jpg', 'cd_fsod_test_20250513_193227.log', 'frame_0014.jpg', 'frame_0025.jpg', 'frame_0051.jpg', 'frame_0039.jpg', 'frame_0030.jpg', 'frame_0008.jpg', 'frame_0002.jpg']
# 2025-05-13 19:34:41,337 - INFO - ================================================================================
# INFO:cd_fsod_test:================================================================================
# 2025-05-13 19:34:41,337 - INFO - CD-FSOD Integration Test Completed Successfully
# INFO:cd_fsod_test:CD-FSOD Integration Test Completed Successfully
# 2025-05-13 19:34:41,337 - INFO - Total processing time: 134.21 seconds
# INFO:cd_fsod_test:Total processing time: 134.21 seconds
# 2025-05-13 19:34:41,337 - INFO - Results saved to: ./cd_fsod_results
# INFO:cd_fsod_test:Results saved to: ./cd_fsod_results
# 2025-05-13 19:34:41,337 - INFO - ================================================================================
# INFO:cd_fsod_test:================================================================================