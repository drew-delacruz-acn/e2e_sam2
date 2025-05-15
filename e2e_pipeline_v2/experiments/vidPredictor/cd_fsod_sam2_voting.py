#!/usr/bin/env python3
"""
CDFSOD + SAM2 Voting Pipeline

This script processes multiple video scenes, performing object detection with CD-FSOD,
mask propagation with SAM2, and temporal label smoothing via object-centric voting.
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
from collections import Counter, defaultdict
import numpy as np
import torch
import gc

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from frame_loader import load_frames_from_directory
from cd_fsod_detector import CDFSODDetector
from sam2_wrapper import SAM2VideoWrapper
from object_tracker import ObjectTracker

# Natural sort
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

# Logger setup
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

def majority_vote(labels):
    if not labels:
        return None, False
    counter = Counter(labels)
    most_common = counter.most_common()
    if len(most_common) == 1:
        return most_common[0][0], False
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        return most_common[0][0], True  # Tie
    return most_common[0][0], False

def process_scene(scene_path, detections_path, output_path, args, main_logger):
    scene_name = scene_path.name
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(f"cd_fsod_sam2_voting.{scene_name}", log_level, output_path, scene_name)
    logger.info(f"Processing scene: {scene_name}")
    # Load frames
    frames, frame_paths = load_frames_from_directory(str(scene_path))
    logger.info(f"Loaded {len(frames)} frames from {scene_path}")
    # Load CD-FSOD detections
    detector = CDFSODDetector(json_dir=str(detections_path), confidence_threshold=args.confidence, min_gap_frames=args.min_gap_frames)
    logger.info(f"Loaded CD-FSOD detections from {detections_path}")
    # Log summary of detections for the scene
    det_by_frame = detector.detections_by_frame
    logger.info(f"Scene {scene_name}: {len(det_by_frame)} frames with detections.")
    for frame_idx, dets in list(det_by_frame.items())[:5]:
        logger.info(f"  Frame {frame_idx}: {len(dets)} detections.")
        for i, d in enumerate(dets):
            # Use 'coordinates' as the primary bounding box key, fallback to 'box'
            box = d.get('coordinates', d.get('box'))
            logger.info(f"    Detection {i}: box={box}, label={d.get('label')}, score={d.get('confidence', d.get('score', 'N/A'))}")
    # Track objects (IoU only)
    tracker = ObjectTracker(iou_weight=1.0, emb_weight=0.0)
    tracked_objects = defaultdict(list)  # obj_id -> list of (frame_idx, box, label, score)
    # Log raw detection JSONs for first few frames
    for idx, frame_path in enumerate(frame_paths[:5]):
        try:
            with open(frame_path.replace(str(scene_path), str(detections_path)).replace('.jpg', '.json').replace('.png', '.json'), 'r') as f:
                raw_json = f.read()
            logger.debug(f"Raw detection JSON for frame {idx}: {raw_json}")
        except Exception as e:
            logger.debug(f"Could not read detection JSON for frame {idx}: {e}")
    for idx, frame in enumerate(frames):
        dets = detector.detect(frame_paths[idx], args.text_queries)
        detections = []
        print('------------')
        print(dets)
        print('------------')
        for coordinates, label, score in zip(dets["coordinates"], dets["labels"], dets["scores"]):
            if coordinates is None or (isinstance(coordinates, (list, tuple)) and any(b is None for b in coordinates)):
                logger.warning(f"Skipping detection in frame {idx}: invalid coordinates: {coordinates}, label: {label}, score: {score}")
                continue
            detections.append({
                "coordinates": coordinates,
                "text": label,
                "score": score
            })
        current_tracks = tracker.update_tracks(frame, idx, detections, embedding_extractor=None)
        for obj_id, coordinates in current_tracks.items():
            label = tracker.tracked_objects[obj_id]["class"]
            score = None  # Optionally, you can store the detection score
            tracked_objects[obj_id].append((idx, coordinates, label, score))
    logger.info(f"Tracked {len(tracked_objects)} objects in scene {scene_name}")
    if len(tracked_objects) == 0:
        logger.warning(f"No objects were tracked in scene {scene_name}. Possible causes: empty detections, high confidence threshold, or invalid detection coordinates.")
    # Propagate masks with SAM2
    voting_results = defaultdict(dict)  # obj_id -> frame_idx -> {"label": ..., "mask": ...}
    corrections_log = []
    if args.separate_objects:
        logger.info("Processing each object separately for mask propagation and voting.")
        sam2 = SAM2VideoWrapper(args.sam2_checkpoint, args.sam2_config)
        for obj_id, track in tracked_objects.items():
            sam2.set_video(frames=frames)
            video_segments, _ = sam2.propagate_masks(objects_to_track=[obj_id])
            mask_frames = []
            labels = []
            for (frame_idx, coordinates, label, score) in track:
                mask = video_segments.get(frame_idx, {}).get(obj_id)
                if mask is not None and np.any(mask):
                    mask_frames.append(frame_idx)
                    labels.append(label)
            if not mask_frames:
                continue
            voted_label, is_tie = majority_vote(labels)
            if is_tie:
                voted_label = labels[0]
                corrections_log.append({"obj_id": obj_id, "frames": mask_frames, "reason": "tie", "chosen_label": voted_label})
            for fidx in mask_frames:
                voting_results[obj_id][fidx] = {
                    "label": voted_label,
                    "mask": video_segments[fidx][obj_id]
                }
    else:
        logger.info("Processing all objects together for mask propagation and voting.")
        sam2 = SAM2VideoWrapper(args.sam2_checkpoint, args.sam2_config)
        sam2.set_video(frames=frames)
        video_segments, _ = sam2.propagate_masks(objects_to_track=list(tracked_objects.keys()))
        for obj_id, track in tracked_objects.items():
            mask_frames = []
            labels = []
            for (frame_idx, coordinates, label, score) in track:
                mask = video_segments.get(frame_idx, {}).get(obj_id)
                if mask is not None and np.any(mask):
                    mask_frames.append(frame_idx)
                    labels.append(label)
            if not mask_frames:
                continue
            voted_label, is_tie = majority_vote(labels)
            if is_tie:
                voted_label = labels[0]
                corrections_log.append({"obj_id": obj_id, "frames": mask_frames, "reason": "tie", "chosen_label": voted_label})
            for fidx in mask_frames:
                voting_results[obj_id][fidx] = {
                    "label": voted_label,
                    "mask": video_segments[fidx][obj_id]
                }
    if len(tracked_objects) == 0:
        logger.warning(f"Skipping mask propagation and voting for scene {scene_name} because no objects were tracked.")
    # Save results
    results_dir = Path(output_path)
    results_dir.mkdir(exist_ok=True, parents=True)
    # Save per-object, per-frame masks and labels
    for obj_id, frames_dict in voting_results.items():
        obj_dir = results_dir / f"object_{obj_id}"
        obj_dir.mkdir(exist_ok=True, parents=True)
        for frame_idx, data in frames_dict.items():
            mask = data["mask"].astype(np.uint8) * 255
            mask_path = obj_dir / f"frame_{frame_idx:04d}_mask.png"
            label_path = obj_dir / f"frame_{frame_idx:04d}_label.txt"
            from PIL import Image
            Image.fromarray(mask).save(mask_path)
            with open(label_path, "w") as f:
                f.write(data["label"])

    # Output per-frame JSONs with updated fields
    frame_json_dir = results_dir / "updated_detections"
    frame_json_dir.mkdir(exist_ok=True, parents=True)
    num_frames = len(frames)
    # Build a lookup: obj_id -> {frame_idx -> voted_label, mask_path}
    obj_frame_lookup = defaultdict(dict)
    for obj_id, frames_dict in voting_results.items():
        for frame_idx, data in frames_dict.items():
            mask_path = str((results_dir / f"object_{obj_id}" / f"frame_{frame_idx:04d}_mask.png").relative_to(results_dir))
            obj_frame_lookup[obj_id][frame_idx] = {
                "label": data["label"],
                "mask_path": mask_path
            }
    # For each frame, build the updated detections list
    for idx, frame_path in enumerate(frame_paths):
        # Load original detections
        dets = detector.detect(frame_path, args.text_queries)
        logger.info(f"Frame {idx:04d}: Found {len(dets['coordinates'])} detections.")
        for i in range(len(dets["coordinates"])):
            logger.info(f"  Detection {i}: coordinates={dets['coordinates'][i]}, label={dets['labels'][i]}, score={dets['scores'][i]:.3f}")
        updated_dets = []
        used_obj_ids = set()
        for i in range(len(dets["coordinates"])):
            coordinates = dets["coordinates"][i]
            orig_label = dets["labels"][i]
            score = dets["scores"][i]
            # Find the tracked object for this detection (by IoU match)
            best_obj_id = None
            best_iou = 0.0
            for obj_id, track in tracked_objects.items():
                for (trk_idx, trk_coordinates, trk_label, _) in track:
                    if trk_idx == idx:
                        # Compute IoU
                        x1, y1, x2, y2 = coordinates
                        tx1, ty1, tx2, ty2 = trk_coordinates
                        xi1 = max(x1, tx1)
                        yi1 = max(y1, ty1)
                        xi2 = min(x2, tx2)
                        yi2 = min(y2, ty2)
                        if xi2 > xi1 and yi2 > yi1:
                            inter = (xi2 - xi1) * (yi2 - yi1)
                            area1 = (x2 - x1) * (y2 - y1)
                            area2 = (tx2 - tx1) * (ty2 - ty1)
                            union = area1 + area2 - inter
                            iou = inter / union if union > 0 else 0.0
                            if iou > best_iou:
                                best_iou = iou
                                best_obj_id = obj_id
            # If matched to a tracked object with a voted label, update
            if best_obj_id is not None and idx in obj_frame_lookup[best_obj_id]:
                voted_label = obj_frame_lookup[best_obj_id][idx]["label"]
                mask_path = obj_frame_lookup[best_obj_id][idx]["mask_path"]
                corrected = (voted_label != orig_label)
                updated_dets.append({
                    "coordinates": coordinates,
                    "label": voted_label,
                    "score": float(score),
                    "voted_label": voted_label,
                    "corrected": corrected,
                    "mask_path": mask_path,
                    "filled_gap": False
                })
                used_obj_ids.add(best_obj_id)
            else:
                updated_dets.append({
                    "coordinates": coordinates,
                    "label": orig_label,
                    "score": float(score),
                    "voted_label": orig_label,
                    "corrected": False,
                    "mask_path": None,
                    "filled_gap": False
                })
        # Now, fill in gaps: for each tracked object with a mask in this frame but not used above
        for obj_id, frame_info in obj_frame_lookup.items():
            if idx in frame_info and obj_id not in used_obj_ids:
                # Get mask and compute bounding box
                mask_path = frame_info[idx]["mask_path"]
                voted_label = frame_info[idx]["label"]
                # Load mask to get bounding box
                mask_img_path = results_dir / mask_path
                from PIL import Image
                mask = np.array(Image.open(mask_img_path))
                ys, xs = np.where(mask > 0)
                if len(xs) > 0 and len(ys) > 0:
                    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                    coordinates = [x1, y1, x2, y2]
                else:
                    coordinates = [0, 0, 0, 0]
                updated_dets.append({
                    "coordinates": coordinates,
                    "label": voted_label,
                    "score": -1.0,
                    "voted_label": voted_label,
                    "corrected": True,
                    "mask_path": mask_path,
                    "filled_gap": True
                })
        # Save updated detections JSON
        frame_json_path = frame_json_dir / f"frame_{idx:04d}_detections.json"
        with open(frame_json_path, "w") as f:
            json.dump(updated_dets, f, indent=2)

    # Save corrections log
    with open(results_dir / "corrections_log.json", "w") as f:
        json.dump(corrections_log, f, indent=2)
    logger.info(f"Saved results for scene {scene_name} to {results_dir}")

    # At the end of scene processing, add a summary
    num_frames_with_detections = sum(1 for dets in det_by_frame.values() if len(dets) > 0)
    logger.info(f"Scene summary: {len(tracked_objects)} objects tracked, {num_frames_with_detections} frames with detections out of {len(frames)} frames.")
    return True

def main():
    parser = argparse.ArgumentParser(description="CDFSOD + SAM2 Voting Pipeline for multiple scenes")
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
    parser.add_argument("--separate-objects", action="store_true", help="Process each object separately (for dtype safety)")
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(exist_ok=True, parents=True)
    log_level = logging.DEBUG if args.debug else logging.INFO
    main_logger = setup_logger("cd_fsod_sam2_voting", log_level, args.output_root)
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
    for scene_idx, scene_dir in enumerate(scene_dirs):
        scene_name = scene_dir.name
        main_logger.info(f"[{scene_idx+1}/{len(scene_dirs)}] Processing scene: {scene_name}")
        detection_dir = detections_root / scene_name
        if not detection_dir.exists() or not detection_dir.is_dir():
            main_logger.error(f"Matching detection directory not found for scene {scene_name}: {detection_dir}")
            main_logger.error(f"Skipping scene: {scene_name}")
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
        else:
            main_logger.error(f"Failed to process scene: {scene_name}")
    main_logger.info("CDFSOD + SAM2 Voting Pipeline Completed")

if __name__ == "__main__":
    main() 
