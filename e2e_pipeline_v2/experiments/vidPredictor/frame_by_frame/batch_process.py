#!/usr/bin/env python3
"""
SAM2 Video Segmentation Batch Processing
----------------------------------------
This script runs the SAM2 video segmentation pipeline on multiple video sequences.
It looks for matching subdirectories in the detections and frames root directories,
then processes each pair using the main.py script.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import time
import logging
import gc  # For garbage collection
import torch  # For GPU memory management
from concurrent.futures import ProcessPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("batch_process.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("batch_processor")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SAM2 Video Segmentation Batch Processing')
    
    # Root directories
    parser.add_argument('--detections_root', type=str, required=True,
                        help='Root directory containing detection subdirectories')
    parser.add_argument('--frames_root', type=str, required=True,
                        help='Root directory containing frame subdirectories')
    parser.add_argument('--results_root', type=str, required=True,
                        help='Root directory to save all results')
    
    # SAM2 model paths (same for all runs)
    parser.add_argument('--sam2_checkpoint', type=str, required=True,
                        help='Path to SAM2 checkpoint file')
    parser.add_argument('--model_cfg', type=str, required=True,
                        help='Path to SAM2 model configuration file')
    
    # Optional parameters to pass to main.py
    parser.add_argument('--confidence_threshold', type=float, default=0.9,
                        help='Confidence threshold for filtering detections')
    parser.add_argument('--vis_stride', type=int, default=1,
                        help='Stride for visualization (display every nth frame)')
    parser.add_argument('--iou_threshold', type=float, default=0.3,
                        help='IoU threshold for matching detections to segments')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with additional logging')
    parser.add_argument('--no_vis', action='store_true',
                        help='Skip visualization generation')
    parser.add_argument('--save_masks', action='store_true',
                        help='Save binary masks for each object')
    
    # Batch processing options
    parser.add_argument('--parallel', action='store_true',
                        help='Enable parallel processing')
    parser.add_argument('--max_workers', type=int, default=2,
                        help='Maximum number of worker processes when parallel is enabled')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip video sequences with existing result directories')
    
    return parser.parse_args()

def get_matching_directories(detections_root, frames_root):
    """
    Find subdirectories that exist in both detections and frames roots
    
    Args:
        detections_root: Path to the root directory containing detection subdirectories
        frames_root: Path to the root directory containing frame subdirectories
        
    Returns:
        List of directory names that exist in both roots
    """
    # Convert to Path objects
    detections_path = Path(detections_root)
    frames_path = Path(frames_root)
    
    # Get subdirectories in both roots
    detection_dirs = [d.name for d in detections_path.iterdir() if d.is_dir()]
    frame_dirs = [d.name for d in frames_path.iterdir() if d.is_dir()]
    
    # Find matching directories
    matching_dirs = sorted(list(set(detection_dirs) & set(frame_dirs)))
    
    return matching_dirs

def process_video_sequence(args, video_name):
    """
    Process a single video sequence by calling main.py
    
    Args:
        args: Command-line arguments
        video_name: Name of the video sequence directory
        
    Returns:
        Tuple (success, output) where success is a boolean and output is the command output
    """
    # Create paths
    detections_dir = os.path.join(args.detections_root, video_name)
    frames_dir = os.path.join(args.frames_root, video_name)
    results_dir = os.path.join(args.results_root, video_name)
    
    # Skip if results directory exists and skip_existing is enabled
    if args.skip_existing and os.path.exists(results_dir):
        logger.info(f"Skipping {video_name} - results directory already exists")
        return True, "Skipped (results already exist)"
    
    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Build the command
    cmd = [
        "python", 
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py"),
        "--detections_dir", detections_dir,
        "--frames_dir", frames_dir,
        "--sam2_checkpoint", args.sam2_checkpoint,
        "--model_cfg", args.model_cfg,
        "--results_dir", results_dir,
        "--confidence_threshold", str(args.confidence_threshold),
        "--vis_stride", str(args.vis_stride),
        "--iou_threshold", str(args.iou_threshold)
    ]
    
    # Add optional flags
    if args.debug:
        cmd.append("--debug")
    if args.no_vis:
        cmd.append("--no_vis")
    if args.save_masks:
        cmd.append("--save_masks")
    
    # Log the command
    logger.info(f"Processing {video_name} with command: {' '.join(cmd)}")
    
    try:
        # Run the command and capture output
        start_time = time.time()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = process.communicate()
        
        # Check if the process was successful
        if process.returncode == 0:
            elapsed_time = time.time() - start_time
            logger.info(f"Completed {video_name} in {elapsed_time:.2f} seconds")
            success = True
            output = stdout
        else:
            logger.error(f"Failed to process {video_name}: {stderr}")
            success = False
            output = stderr
        
        # Clean up GPU memory after processing
        logger.info("Cleaning up GPU memory...")
        if torch.cuda.is_available():
            # Log memory usage before cleanup
            before_mem = torch.cuda.memory_allocated() / (1024 ** 3)
            before_cached = torch.cuda.memory_reserved() / (1024 ** 3)
            logger.info(f"Before cleanup - GPU memory allocated: {before_mem:.2f} GB, reserved: {before_cached:.2f} GB")
            
            # Empty CUDA cache and run garbage collection
            torch.cuda.empty_cache()
            gc.collect()
            
            # Log memory usage after cleanup
            after_mem = torch.cuda.memory_allocated() / (1024 ** 3)
            after_cached = torch.cuda.memory_reserved() / (1024 ** 3)
            logger.info(f"After cleanup - GPU memory allocated: {after_mem:.2f} GB, reserved: {after_cached:.2f} GB")
            logger.info(f"Memory freed: {max(0, before_mem - after_mem):.2f} GB allocated, {max(0, before_cached - after_cached):.2f} GB reserved")
            
            # Add a small delay to allow memory to be fully released
            time.sleep(2)
        
        return success, output
    
    except Exception as e:
        logger.exception(f"Error processing {video_name}: {str(e)}")
        
        # Ensure GPU memory is also cleaned up after an exception
        if torch.cuda.is_available():
            logger.info("Cleaning up GPU memory after exception...")
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(2)
            
        return False, str(e)

def process_all_sequences_sequential(args, video_names):
    """Process all video sequences sequentially"""
    results = {}
    
    for video_name in video_names:
        success, output = process_video_sequence(args, video_name)
        results[video_name] = {"success": success, "output": output}
    
    return results

def process_all_sequences_parallel(args, video_names):
    """Process all video sequences in parallel"""
    results = {}
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_video_sequence, args, video_name): video_name 
            for video_name in video_names
        }
        
        # Process results as they complete
        for future in futures:
            video_name = futures[future]
            try:
                success, output = future.result()
                results[video_name] = {"success": success, "output": output}
            except Exception as e:
                logger.exception(f"Error processing {video_name} in parallel: {str(e)}")
                results[video_name] = {"success": False, "output": str(e)}
    
    return results

def main():
    """Main function to run the batch processing pipeline"""
    # Parse command line arguments
    args = parse_args()
    
    # Create the results root directory
    os.makedirs(args.results_root, exist_ok=True)
    
    # Find matching directories
    logger.info("Finding matching video sequence directories...")
    video_names = get_matching_directories(args.detections_root, args.frames_root)
    
    if not video_names:
        logger.error("No matching video sequence directories found!")
        return
    
    logger.info(f"Found {len(video_names)} matching video sequences to process")
    
    # Process all video sequences
    start_time = time.time()
    
    if args.parallel:
        logger.info(f"Processing {len(video_names)} video sequences in parallel with {args.max_workers} workers")
        results = process_all_sequences_parallel(args, video_names)
    else:
        logger.info(f"Processing {len(video_names)} video sequences sequentially")
        results = process_all_sequences_sequential(args, video_names)
    
    total_time = time.time() - start_time
    
    # Print summary
    successful = sum(1 for result in results.values() if result["success"])
    failed = len(results) - successful
    
    logger.info("\n" + "="*50)
    logger.info(f"BATCH PROCESSING SUMMARY")
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info(f"Total video sequences: {len(results)}")
    logger.info(f"Successfully processed: {successful}")
    logger.info(f"Failed to process: {failed}")
    
    if failed > 0:
        logger.info("\nFailed video sequences:")
        for video_name, result in results.items():
            if not result["success"]:
                logger.info(f"  - {video_name}")
    
    logger.info("="*50)

if __name__ == "__main__":
    main() 