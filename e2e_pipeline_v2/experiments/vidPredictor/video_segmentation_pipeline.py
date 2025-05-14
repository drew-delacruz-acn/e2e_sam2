#!/usr/bin/env python3
"""
Video Segmentation Pipeline

This script processes multiple video scenes from a directory structure, performing
object detection and segmentation using either CD-FSOD JSON detections or real-time 
detection with OWLv2, and segmentation with SAM2.

The script takes input directories containing scenes (subdirectories) of frames and
detections, and produces an output directory with the same structure containing
the processed results.
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
import shutil
import gc
import torch

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from object_tracking_pipeline import ObjectTrackingPipeline
from cd_fsod_detector import CDFSODDetector
from owlv2_detector import OWLv2Detector

# Add natural sorting function
def natural_sort_key(s):
    """Key function for natural sorting"""
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', str(s))]

# Configure logging
def setup_logger(name, log_level=logging.INFO, output_dir=None, scene_name=None):
    """Set up a logger with console and file handlers."""
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.handlers = []  # Clear existing handlers
    
    # Create console handler with formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Create formatter - include scene name if provided
    prefix = f"[{scene_name}] " if scene_name else ""
    formatter = logging.Formatter(f'%(asctime)s - %(levelname)s - {prefix}%(message)s')
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    # Add file handler if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name = f"{scene_name}_" if scene_name else ""
        log_file = os.path.join(output_dir, f"{log_name}video_segmentation_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    return logger

# Store original detector methods for enhanced logging
original_cd_fsod_load_detections = CDFSODDetector._load_detections
original_cd_fsod_process_detections = CDFSODDetector._process_detections
original_cd_fsod_detect = CDFSODDetector.detect

# Enhanced logging versions of detector methods
def enhanced_load_detections(self):
    """Enhanced logging version of CDFSODDetector._load_detections"""
    logger = logging.getLogger("video_segmentation")
    logger.info(f"Loading CD-FSOD detections from {self.json_dir}...")
    
    start_time = time.time()
    detections_by_frame = original_cd_fsod_load_detections(self)
    
    total_detections = sum(len(dets) for dets in detections_by_frame.values())
    frame_count = len(detections_by_frame)
    load_time = time.time() - start_time
    
    logger.info(f"Loaded {total_detections} detections across {frame_count} frames in {load_time:.2f} seconds")
    logger.info(f"Average {total_detections / max(1, frame_count):.2f} detections per frame")
    
    # Log confidence threshold
    logger.info(f"Using confidence threshold: {self.confidence_threshold}")
    
    # Log detection counts by frame (debug only)
    if logger.level <= logging.DEBUG:
        for frame_idx, dets in sorted(detections_by_frame.items())[:5]:  # First 5 frames only
            logger.debug(f"  Frame {frame_idx}: {len(dets)} detections")
    
    return detections_by_frame

def enhanced_process_detections(self):
    """Enhanced logging version of CDFSODDetector._process_detections"""
    logger = logging.getLogger("video_segmentation")
    logger.info("Processing CD-FSOD detections to identify first appearances and reappearances...")
    
    start_time = time.time()
    original_cd_fsod_process_detections(self)
    
    # Count the results
    first_appearances_count = sum(len(v) for v in self.first_appearances.values())
    reappearances_count = sum(len(v) for v in self.reappearances.values())
    object_count = len(self.object_tracks)
    
    process_time = time.time() - start_time
    
    logger.info(f"Processed detections in {process_time:.2f} seconds")
    logger.info(f"Identified {first_appearances_count} first appearances and {reappearances_count} reappearances")
    logger.info(f"Tracking {object_count} unique objects")
    
    # Log min gap frames parameter
    logger.info(f"Using minimum gap frames for reappearance: {self.min_gap_frames}")
    
    # Log first appearances by frame (debug only)
    if logger.level <= logging.DEBUG:
        logger.debug("First appearances by frame (first 5 frames only):")
        for frame_idx, appearances in sorted(self.first_appearances.items())[:5]:
            if appearances:
                logger.debug(f"  Frame {frame_idx}: {len(appearances)} first appearances: {[d['label'] for d in appearances]}")
    
    return

def enhanced_detect(self, image, text_queries, threshold=None):
    """Enhanced logging version of CDFSODDetector.detect"""
    logger = logging.getLogger("video_segmentation")
    
    # Extract frame index
    frame_idx = self._extract_frame_idx(image)
    
    if logger.level <= logging.DEBUG:
        if isinstance(image, dict):
            logger.debug(f"CD-FSOD detecting in frame {frame_idx} with queries: {text_queries}")
        else:
            logger.debug(f"CD-FSOD detecting with queries: {text_queries}")
    
    # Call original method
    results = original_cd_fsod_detect(self, image, text_queries, threshold)
    
    # Log results at debug level
    if logger.level <= logging.DEBUG:
        box_count = len(results.get("boxes", []))
        if box_count > 0:
            logger.debug(f"  Detected {box_count} objects: {results.get('labels', [])}")
        else:
            logger.debug(f"  No objects detected")
    
    return results

def apply_detector_patches():
    """Apply monkey patches to the CD-FSOD detector class to enhance logging."""
    logger = logging.getLogger("video_segmentation")
    logger.info("Applying enhanced logging patches to CD-FSOD detector...")
    
    # Patch the detector methods
    CDFSODDetector._load_detections = enhanced_load_detections
    CDFSODDetector._process_detections = enhanced_process_detections
    CDFSODDetector.detect = enhanced_detect
    
    logger.info("CD-FSOD detector patched with enhanced logging")

def log_pipeline_progress(pipeline, total_frames, frame_idx, interval=5):
    """Log pipeline progress at specified intervals."""
    logger = logging.getLogger("video_segmentation")
    
    # Log at interval percentages
    if total_frames > 0:
        progress = (frame_idx / total_frames) * 100
        if progress % interval < (1 / total_frames) * 100:
            logger.info(f"Processing progress: {progress:.1f}% ({frame_idx}/{total_frames} frames)")
            
            # Add stats if available
            if hasattr(pipeline, 'tracked_objects'):
                object_count = len(pipeline.tracked_objects)
                logger.info(f"Currently tracking {object_count} objects")

def process_scene(
    scene_path, 
    detections_path, 
    output_path, 
    args, 
    main_logger
):
    """
    Process a single scene directory.
    
    Args:
        scene_path: Path to the scene frames directory
        detections_path: Path to the scene detections directory
        output_path: Path to the scene output directory
        args: Command line arguments
        main_logger: Main logger instance
    """
    scene_name = scene_path.name
    
    # Create scene-specific logger
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(
        name=f"video_segmentation.{scene_name}",
        log_level=log_level,
        output_dir=output_path,
        scene_name=scene_name
    )
    
    # Log scene processing start
    logger.info("=" * 80)
    logger.info(f"Processing scene: {scene_name}")
    logger.info("=" * 80)
    
    # Check for frames in the scene directory
    try:
        frame_files = sorted([f for f in scene_path.glob("*.jpg") or scene_path.glob("*.png")], key=natural_sort_key)
        logger.info(f"Found {len(frame_files)} frames in {scene_path}")
        if len(frame_files) == 0:
            logger.error(f"No frames found in {scene_path}. Skipping scene.")
            return False
        logger.debug(f"First 5 frames: {[f.name for f in frame_files[:5]]}")
    except Exception as e:
        logger.error(f"Error accessing frames directory: {e}")
        return False
    
    # For CD-FSOD detector, check for JSON files
    if args.detector_type == "cd_fsod":
        try:
            json_files = sorted([f for f in detections_path.glob("*.json")], key=natural_sort_key)
            logger.info(f"Found {len(json_files)} JSON detection files in {detections_path}")
            if len(json_files) == 0:
                logger.error(f"No JSON files found in {detections_path}. Skipping scene.")
                return False
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
            return False
    
    # Apply detector patches for enhanced logging if requested
    if not args.no_enhanced_logging and args.detector_type == "cd_fsod":
        try:
            apply_detector_patches()
        except Exception as e:
            logger.error(f"Error applying detector patches: {e}")
            logger.warning("Continuing without enhanced detector logging")
    
    # Initialize timer
    start_time = time.time()
    logger.info(f"Initializing pipeline with {args.detector_type} detector...")
    
    try:
        # Initialize pipeline with appropriate detector
        if args.detector_type == "cd_fsod":
            pipeline = ObjectTrackingPipeline(
                owlv2_checkpoint=None,  # Not used with CD-FSOD detector
                sam2_checkpoint=args.sam2_checkpoint,
                sam2_config=args.sam2_config,
                output_dir=str(output_path),
                confidence_threshold=args.confidence,
                detector_type="cd_fsod",
                cd_fsod_path=str(detections_path),
                min_gap_frames=args.min_gap_frames,
                mask_quality_threshold=args.mask_quality_threshold
            )
        else:  # owlv2
            pipeline = ObjectTrackingPipeline(
                owlv2_checkpoint=args.owlv2_checkpoint,
                sam2_checkpoint=args.sam2_checkpoint,
                sam2_config=args.sam2_config,
                output_dir=str(output_path),
                confidence_threshold=args.confidence,
                detector_type="owlv2",
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
        return False
    
    # Process video
    processing_start = time.time()
    logger.info("Starting scene processing...")
    
    try:
        # Process video using the appropriate method
        if args.separate_objects:
            logger.info("Using separate object initialization method...")
            
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
                frames_dir=str(scene_path),
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
                frames_dir=str(scene_path),
                text_queries=args.text_queries
            )
        
        processing_time = time.time() - processing_start
        logger.info(f"Scene processing completed in {processing_time:.2f} seconds")
        
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
        results_files = list(Path(output_path).glob("*"))
        logger.info(f"Generated {len(results_files)} output files")
        if args.debug:
            logger.debug(f"Output files: {[f.name for f in results_files[:10]]}")
        
        # Log completion
        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"Scene Processing Completed Successfully: {scene_name}")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info(f"Results saved to: {output_path}")
        logger.info("=" * 80)
        
        # Clear GPU memory before moving to the next scene
        logger.info("Clearing GPU memory and cache...")
        # Log memory usage before clearing
        if torch.cuda.is_available():
            before_mem = torch.cuda.memory_allocated() / (1024 ** 3)
            before_cached = torch.cuda.memory_reserved() / (1024 ** 3)
            logger.info(f"Before clearing - GPU memory allocated: {before_mem:.2f} GB, reserved: {before_cached:.2f} GB")
        
        torch.cuda.empty_cache()
        gc.collect()
        
        # Log memory usage after clearing
        if torch.cuda.is_available():
            after_mem = torch.cuda.memory_allocated() / (1024 ** 3)
            after_cached = torch.cuda.memory_reserved() / (1024 ** 3)
            logger.info(f"After clearing - GPU memory allocated: {after_mem:.2f} GB, reserved: {after_cached:.2f} GB")
            logger.info(f"Memory freed: {before_mem - after_mem:.2f} GB allocated, {before_cached - after_cached:.2f} GB reserved")
        
        logger.info("Memory cleared successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during scene processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=" * 80)
        logger.info(f"Scene Processing Failed: {scene_name}")
        logger.info("=" * 80)
        
        # Also clear memory after failed processing
        logger.info("Clearing GPU memory and cache after failure...")
        # Log memory usage before clearing
        if torch.cuda.is_available():
            before_mem = torch.cuda.memory_allocated() / (1024 ** 3)
            before_cached = torch.cuda.memory_reserved() / (1024 ** 3)
            logger.info(f"Before clearing - GPU memory allocated: {before_mem:.2f} GB, reserved: {before_cached:.2f} GB")
        
        torch.cuda.empty_cache()
        gc.collect()
        
        # Log memory usage after clearing
        if torch.cuda.is_available():
            after_mem = torch.cuda.memory_allocated() / (1024 ** 3)
            after_cached = torch.cuda.memory_reserved() / (1024 ** 3)
            logger.info(f"After clearing - GPU memory allocated: {after_mem:.2f} GB, reserved: {after_cached:.2f} GB")
            logger.info(f"Memory freed: {before_mem - after_mem:.2f} GB allocated, {before_cached - after_cached:.2f} GB reserved")
        
        return False

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Video Segmentation Pipeline for multiple scenes")
    
    # Directory paths
    parser.add_argument("--frames-root", required=True, help="Root directory containing scene subdirectories with frames")
    parser.add_argument("--detections-root", required=True, help="Root directory containing scene subdirectories with detections")
    parser.add_argument("--output-root", default="./segmentation_results", help="Root output directory for results")
    
    # Detector selection
    parser.add_argument("--detector-type", default="cd_fsod", choices=["cd_fsod", "owlv2"], 
                        help="Type of detector to use (default: cd_fsod)")
    
    # Model paths
    parser.add_argument("--sam2-checkpoint", required=True, help="Path to SAM2 checkpoint")
    parser.add_argument("--sam2-config", required=True, help="Path to SAM2 config file")
    parser.add_argument("--owlv2-checkpoint", help="Path to OWLv2 checkpoint (required if using owlv2 detector)")
    
    # Detection parameters
    parser.add_argument("--text-queries", default=["all"], nargs="+", help="Text queries for object detection (default: 'all')")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for detections")
    parser.add_argument("--min-gap-frames", type=int, default=10, help="Minimum gap frames for CD-FSOD reappearances")
    
    # Scene selection
    parser.add_argument("--scene", help="Process only the specified scene name (optional)")
    
    # Processing options
    parser.add_argument("--separate-objects", action="store_true", help="Process each object separately to avoid dtype issues")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-enhanced-logging", action="store_true", help="Disable enhanced detector logging")
    
    # Mask quality options
    parser.add_argument("--mask-quality-threshold", type=int, default=0, help="Minimum pixel count for high-quality masks (default: 0)")
    
    args = parser.parse_args()
    
    # Validate detector-specific requirements
    if args.detector_type == "owlv2" and not args.owlv2_checkpoint:
        parser.error("--owlv2-checkpoint is required when using owlv2 detector")
    
    # Create output root directory
    output_root = Path(args.output_root)
    output_root.mkdir(exist_ok=True, parents=True)
    
    # Set up main logger
    log_level = logging.DEBUG if args.debug else logging.INFO
    main_logger = setup_logger(
        name="video_segmentation",
        log_level=log_level,
        output_dir=args.output_root
    )
    
    # Log script start and configuration
    main_logger.info("=" * 80)
    main_logger.info("Video Segmentation Pipeline Started")
    main_logger.info("=" * 80)
    main_logger.info("Configuration:")
    main_logger.info(f"  - Frames root directory: {args.frames_root}")
    main_logger.info(f"  - Detections root directory: {args.detections_root}")
    main_logger.info(f"  - Output root directory: {args.output_root}")
    main_logger.info(f"  - Detector type: {args.detector_type}")
    main_logger.info(f"  - SAM2 checkpoint: {args.sam2_checkpoint}")
    main_logger.info(f"  - SAM2 config: {args.sam2_config}")
    if args.detector_type == "owlv2":
        main_logger.info(f"  - OWLv2 checkpoint: {args.owlv2_checkpoint}")
    main_logger.info(f"  - Confidence threshold: {args.confidence}")
    main_logger.info(f"  - Minimum gap frames: {args.min_gap_frames}")
    main_logger.info(f"  - Text queries: {args.text_queries}")
    main_logger.info(f"  - Using separate objects: {args.separate_objects}")
    main_logger.info(f"  - Debug mode: {args.debug}")
    main_logger.info(f"  - Enhanced logging: {not args.no_enhanced_logging}")
    main_logger.info(f"  - Mask quality threshold: {args.mask_quality_threshold}")
    if args.scene:
        main_logger.info(f"  - Processing only scene: {args.scene}")
    
    # Get frames root directory
    frames_root = Path(args.frames_root)
    detections_root = Path(args.detections_root)
    
    # Find all scene directories
    if args.scene:
        # Process only the specified scene
        scene_dirs = [frames_root / args.scene]
        if not scene_dirs[0].exists() or not scene_dirs[0].is_dir():
            main_logger.error(f"Specified scene directory not found: {scene_dirs[0]}")
            return
    else:
        # Process all scene directories
        scene_dirs = [d for d in frames_root.iterdir() if d.is_dir()]
        scene_dirs.sort(key=lambda x: natural_sort_key(x.name))
    
    main_logger.info(f"Found {len(scene_dirs)} scene directories to process")
    
    # Process each scene
    successful_scenes = 0
    failed_scenes = 0
    
    for scene_idx, scene_dir in enumerate(scene_dirs):
        scene_name = scene_dir.name
        main_logger.info(f"[{scene_idx+1}/{len(scene_dirs)}] Processing scene: {scene_name}")
        
        # Check if matching detection directory exists
        detection_dir = detections_root / scene_name
        if not detection_dir.exists() or not detection_dir.is_dir():
            main_logger.error(f"Matching detection directory not found for scene {scene_name}: {detection_dir}")
            main_logger.error(f"Skipping scene: {scene_name}")
            failed_scenes += 1
            continue
        
        # Create scene output directory
        scene_output_dir = output_root / scene_name
        scene_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Process the scene
        start_time = time.time()
        success = process_scene(
            scene_path=scene_dir,
            detections_path=detection_dir,
            output_path=scene_output_dir,
            args=args,
            main_logger=main_logger
        )
        
        if success:
            main_logger.info(f"Scene {scene_name} processed successfully in {time.time() - start_time:.2f} seconds")
            successful_scenes += 1
        else:
            main_logger.error(f"Failed to process scene: {scene_name}")
            failed_scenes += 1
    
    # Log completion
    main_logger.info("=" * 80)
    main_logger.info("Video Segmentation Pipeline Completed")
    main_logger.info(f"Successfully processed {successful_scenes} scenes")
    if failed_scenes > 0:
        main_logger.warning(f"Failed to process {failed_scenes} scenes")
    main_logger.info(f"Results saved to: {args.output_root}")
    main_logger.info("=" * 80)

if __name__ == "__main__":
    main() 