#!/usr/bin/env python3
"""
CD-FSOD SAM2 Voting Pipeline

This script processes multiple video scenes from a directory structure, performing
object detection and segmentation using CD-FSOD JSON detections and segmentation with SAM2.
It then applies an object-centric voting strategy to assign the most frequent label to each object
across all frames where it is present, improving temporal consistency.

The script takes input directories containing scenes (subdirectories) of frames and
detections, and produces an output directory with the same structure containing
temporally-smoothed, corrected results.
"""

import argparse
import os
import sys
import logging
import time
import json
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import torch

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from object_tracking_pipeline import ObjectTrackingPipeline
from cd_fsod_detector import CDFSODDetector

def setup_logger(name, log_level=logging.INFO, output_dir=None, scene_name=None):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.handlers = []
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    prefix = f"[{scene_name}] " if scene_name else ""
    formatter = logging.Formatter(f'%(asctime)s - %(levelname)s - {prefix}%(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name = f"{scene_name}_" if scene_name else ""
        log_file = os.path.join(output_dir, f"{log_name}cd_fsod_sam2_voting_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    return logger

def natural_sort_key(s):
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

class VotingPipeline(ObjectTrackingPipeline):
    """
    Extends ObjectTrackingPipeline to add object-centric voting for label smoothing.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.voting_logs = {}

    def perform_voting(self):
        """
        For each tracked object, assign the most frequent label across all frames where its mask exists.
        Update per-frame predictions to reflect this voted label. Log corrections and tie-breaks.
        """
        logger = logging.getLogger("cd_fsod_sam2_voting")
        for obj_id, obj_data in self.tracked_objects.items():
            # Gather all frames where mask exists for this object
            mask_frames = []
            for frame_idx, masks in self.propagation_results.items():
                if obj_id in masks and masks[obj_id] is not None and (hasattr(masks[obj_id], 'size') and masks[obj_id].size > 0):
                    mask_frames.append(frame_idx)
            if not mask_frames:
                continue
            # Gather all predicted labels for these frames
            labels = []
            for f in mask_frames:
                # Try to get the label from detections if available, else use object's class
                label = obj_data.get('class', None)
                if 'frame_results' in self.__dict__ and f in self.frame_results:
                    # Try to find detection for this object in this frame
                    dets = self.frame_results[f].get('detections', [])
                    for det in dets:
                        if det.get('label'):
                            labels.append(det['label'])
                            break
                    else:
                        if label:
                            labels.append(label)
                elif label:
                    labels.append(label)
            if not labels:
                continue
            # Majority vote
            label_counts = Counter(labels)
            most_common = label_counts.most_common()
            voted_label = most_common[0][0]
            tie = False
            if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                tie = True
                # Tie-breaker: use label from first frame, or highest average confidence if available
                voted_label = labels[0]
            # Update all frames in span
            for f in mask_frames:
                obj_data['class'] = voted_label
                # Optionally, update frame_results if you want per-frame output
                if 'frame_results' in self.__dict__ and f in self.frame_results:
                    for det in self.frame_results[f].get('detections', []):
                        det['label'] = voted_label
            # Log corrections and tie-breaks
            self.voting_logs[obj_id] = {
                'mask_frames': mask_frames,
                'labels': labels,
                'voted_label': voted_label,
                'label_counts': dict(label_counts),
                'tie': tie
            }
            logger.info(f"Object {obj_id}: voted label '{voted_label}' (votes: {dict(label_counts)}){' [TIE]' if tie else ''}")

    def save_voting_logs(self, output_dir):
        with open(os.path.join(output_dir, 'voting_logs.json'), 'w') as f:
            json.dump(self.voting_logs, f, indent=2)


def process_scene(scene_path, detections_path, output_path, args, main_logger):
    scene_name = scene_path.name
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(
        name=f"cd_fsod_sam2_voting.{scene_name}",
        log_level=log_level,
        output_dir=output_path,
        scene_name=scene_name
    )
    logger.info(f"Processing scene: {scene_name}")
    # Check for frames
    frame_files = sorted([f for f in scene_path.glob("*.jpg")] + [f for f in scene_path.glob("*.png")], key=natural_sort_key)
    if len(frame_files) == 0:
        logger.error(f"No frames found in {scene_path}. Skipping scene.")
        return False
    # Check for JSON files
    json_files = sorted([f for f in detections_path.glob("*.json")], key=natural_sort_key)
    if len(json_files) == 0:
        logger.error(f"No JSON files found in {detections_path}. Skipping scene.")
        return False
    # Initialize pipeline
    pipeline = VotingPipeline(
        owlv2_checkpoint=None,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        output_dir=str(output_path),
        confidence_threshold=args.confidence,
        detector_type="cd_fsod",
        cd_fsod_path=str(detections_path),
        min_gap_frames=args.min_gap_frames,
        mask_quality_threshold=args.mask_quality_threshold
    )
    # Process video (object tracking + mask propagation)
    pipeline.process_video(
        frames_dir=str(scene_path),
        text_queries=args.text_queries
    )
    # Perform voting
    pipeline.perform_voting()
    # Save logs
    pipeline.save_voting_logs(str(output_path))
    logger.info(f"Scene {scene_name} processed and voting results saved.")
    return True

def main():
    pipeline_start_time = time.time()
    parser = argparse.ArgumentParser(description="CD-FSOD SAM2 Voting Pipeline for multiple scenes")
    parser.add_argument("--frames-root", required=True, help="Root directory containing scene subdirectories with frames")
    parser.add_argument("--detections-root", required=True, help="Root directory containing scene subdirectories with detections")
    parser.add_argument("--output-root", default="./voting_results", help="Root output directory for results")
    parser.add_argument("--sam2-checkpoint", required=True, help="Path to SAM2 checkpoint")
    parser.add_argument("--sam2-config", required=True, help="Path to SAM2 config file")
    parser.add_argument("--text-queries", default=["all"], nargs="+", help="Text queries for object detection (default: 'all')")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for detections")
    parser.add_argument("--min-gap-frames", type=int, default=10, help="Minimum gap frames for CD-FSOD reappearances")
    parser.add_argument("--scene", help="Process only the specified scene name (optional)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--mask-quality-threshold", type=int, default=0, help="Minimum pixel count for high-quality masks (default: 0)")
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(exist_ok=True, parents=True)
    log_level = logging.DEBUG if args.debug else logging.INFO
    main_logger = setup_logger(
        name="cd_fsod_sam2_voting",
        log_level=log_level,
        output_dir=args.output_root
    )
    frames_root = Path(args.frames_root)
    detections_root = Path(args.detections_root)
    if args.scene:
        scene_dirs = [frames_root / args.scene]
        if not scene_dirs[0].exists() or not scene_dirs[0].is_dir():
            main_logger.error(f"Specified scene directory not found: {scene_dirs[0]}")
            return
    else:
        scene_dirs = [d for d in frames_root.iterdir() if d.is_dir()]
        scene_dirs.sort(key=lambda x: natural_sort_key(x.name))
    main_logger.info(f"Found {len(scene_dirs)} scene directories to process")
    successful_scenes = 0
    failed_scenes = 0
    for scene_idx, scene_dir in enumerate(scene_dirs):
        scene_name = scene_dir.name
        main_logger.info(f"[{scene_idx+1}/{len(scene_dirs)}] Processing scene: {scene_name}")
        detection_dir = detections_root / scene_name
        if not detection_dir.exists() or not detection_dir.is_dir():
            main_logger.error(f"Matching detection directory not found for scene {scene_name}: {detection_dir}")
            main_logger.error(f"Skipping scene: {scene_name}")
            failed_scenes += 1
            continue
        scene_output_dir = output_root / scene_name
        scene_output_dir.mkdir(exist_ok=True, parents=True)
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
    main_logger.info("=" * 80)
    main_logger.info("CD-FSOD SAM2 Voting Pipeline Completed")
    main_logger.info(f"Successfully processed {successful_scenes} scenes")
    if failed_scenes > 0:
        main_logger.warning(f"Failed to process {failed_scenes} scenes")
    main_logger.info(f"Results saved to: {args.output_root}")
    main_logger.info("=" * 80)
    total_pipeline_time = time.time() - pipeline_start_time
    hours, remainder = divmod(total_pipeline_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    main_logger.info(f"Total pipeline execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    main_logger.info("=" * 80)

if __name__ == "__main__":
    main() 