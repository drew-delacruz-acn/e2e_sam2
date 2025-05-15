#!/usr/bin/env python3
"""
CD-FSOD SAM2 Voting Pipeline

This script implements a video object detection and segmentation pipeline that fuses
CD-FSOD detections and SAM2 mask propagation using an object-centric voting strategy.
The goal is to improve temporal consistency and correct short-term misclassifications
by assigning the most frequent label to each object across all frames where it is present.
"""

import argparse
import os
import sys
import logging
import time
import json
from pathlib import Path
from datetime import datetime
from collections import Counter
import torch
import numpy as np

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
        log_file = os.path.join(output_dir, f"{log_name}voting_pipeline_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    return logger

class CDFSODSAM2VotingPipeline(ObjectTrackingPipeline):
    def __init__(
        self,
        sam2_checkpoint: str,
        sam2_config: str,
        output_dir: str,
        confidence_threshold: float = 0.5,
        min_gap_frames: int = 10,
        mask_quality_threshold: int = 0,
        iou_weight: float = 0.5,
        emb_weight: float = 0.5
    ):
        super().__init__(
            owlv2_checkpoint=None,  # Not used with CD-FSOD
            sam2_checkpoint=sam2_checkpoint,
            sam2_config=sam2_config,
            output_dir=output_dir,
            confidence_threshold=confidence_threshold,
            detector_type="cd_fsod",
            min_gap_frames=min_gap_frames,
            mask_quality_threshold=mask_quality_threshold,
            iou_weight=iou_weight,
            emb_weight=emb_weight
        )
        self.voted_labels = {}
        self.label_votes = {}

    def _perform_label_voting(self, obj_id, frame_indices):
        labels = []
        for frame_idx in frame_indices:
            # Try to get the label for this object in this frame
            # Use the tracked_objects' class as the label (could be improved if per-frame labels are stored)
            if obj_id in self.tracked_objects:
                labels.append(self.tracked_objects[obj_id]["class"])
        if not labels:
            return None
        label_counts = Counter(labels)
        most_common = label_counts.most_common(1)[0]
        if len(label_counts) > 1:
            max_count = most_common[1]
            tied_labels = [label for label, count in label_counts.items() if count == max_count]
            if len(tied_labels) > 1:
                voted_label = labels[0]
            else:
                voted_label = most_common[0]
        else:
            voted_label = most_common[0]
        self.voted_labels[obj_id] = voted_label
        self.label_votes[obj_id] = dict(label_counts)
        return voted_label

    def process_video(self, frames_dir: str, text_queries):
        super().process_video(frames_dir, text_queries)
        for obj_id, obj_data in self.tracked_objects.items():
            frame_indices = []
            for frame_idx, masks in self.propagation_results.items():
                if obj_id in masks:
                    frame_indices.append(frame_idx)
            voted_label = self._perform_label_voting(obj_id, frame_indices)
            if voted_label:
                self.tracked_objects[obj_id]["class"] = voted_label
                print(f"Object {obj_id} voted label: {voted_label}")
                print(f"Vote distribution: {self.label_votes[obj_id]}")
        self._save_voting_results()

    def _save_voting_results(self):
        results = {
            "voted_labels": self.voted_labels,
            "vote_distributions": self.label_votes,
            "objects": {
                str(obj_id): {
                    "final_label": obj_data["class"],
                    "first_detected": obj_data["first_detected"],
                    "last_seen": obj_data["last_seen"],
                    "frame_count": len(obj_data["boxes"])
                }
                for obj_id, obj_data in self.tracked_objects.items()
            }
        }
        output_path = Path(self.output_dir) / "voting_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved voting results to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="CD-FSOD SAM2 Voting Pipeline")
    parser.add_argument("--frames-root", required=True, help="Root directory containing scene subdirectories with frames")
    parser.add_argument("--detections-root", required=True, help="Root directory containing scene subdirectories with detections")
    parser.add_argument("--output-root", default="./voting_results", help="Root output directory for results")
    parser.add_argument("--sam2-checkpoint", required=True, help="Path to SAM2 checkpoint")
    parser.add_argument("--sam2-config", required=True, help="Path to SAM2 config file")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for detections")
    parser.add_argument("--min-gap-frames", type=int, default=10, help="Minimum gap frames for reappearances")
    parser.add_argument("--mask-quality-threshold", type=int, default=0, help="Minimum pixel count for high-quality masks")
    parser.add_argument("--iou-weight", type=float, default=0.5, help="Weight for IoU in object tracking (set to 1.0 for IoU-only)")
    parser.add_argument("--emb-weight", type=float, default=0.5, help="Weight for embedding similarity in object tracking (set to 0.0 for IoU-only)")
    parser.add_argument("--scene", help="Process only the specified scene name (optional)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(exist_ok=True, parents=True)
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(
        name="voting_pipeline",
        log_level=log_level,
        output_dir=args.output_root
    )
    frames_root = Path(args.frames_root)
    detections_root = Path(args.detections_root)
    if args.scene:
        scene_dirs = [frames_root / args.scene]
    else:
        scene_dirs = [d for d in frames_root.iterdir() if d.is_dir()]
    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        logger.info(f"Processing scene: {scene_name}")
        scene_output_dir = output_root / scene_name
        scene_output_dir.mkdir(exist_ok=True, parents=True)
        pipeline = CDFSODSAM2VotingPipeline(
            sam2_checkpoint=args.sam2_checkpoint,
            sam2_config=args.sam2_config,
            output_dir=str(scene_output_dir),
            confidence_threshold=args.confidence,
            min_gap_frames=args.min_gap_frames,
            mask_quality_threshold=args.mask_quality_threshold,
            iou_weight=args.iou_weight,
            emb_weight=args.emb_weight
        )
        try:
            pipeline.process_video(
                frames_dir=str(scene_dir),
                text_queries=["all"]
            )
            logger.info(f"Successfully processed scene: {scene_name}")
        except Exception as e:
            logger.error(f"Error processing scene {scene_name}: {e}")
            continue

if __name__ == "__main__":
    main() 