# owlv2_sam2_pipeline.py
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

# Import our modules
from frame_loader import load_frames_from_directory
from owlv2_detector import OWLv2Detector
from embedding_extractor import EmbeddingExtractor
from sam2_wrapper import SAM2VideoWrapper
from object_tracker import ObjectTracker

def run_pipeline(frames_dir, output_dir, text_queries, sam_checkpoint, sam_config, threshold=0.1):
    """Run the complete OWLv2 + SAM2 pipeline"""
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    
    # Step 1: Load video frames
    frames, frame_paths = load_frames_from_directory(frames_dir)
    
    # Step 2: Initialize components
    # OWLv2 Detector
    detector = OWLv2Detector()
    
    # Embedding Extractor for tracking
    embedding_extractor = EmbeddingExtractor()
    
    # SAM2 Video Predictor
    sam_wrapper = SAM2VideoWrapper(sam_checkpoint, sam_config)
    sam_wrapper.set_video(frames)
    
    # Object Tracker
    tracker = ObjectTracker()
    
    # Step 3: Process frames
    tracking_results = []
    
    for frame_idx, frame in enumerate(frames):
        print(f"\nProcessing frame {frame_idx+1}/{len(frames)}")
        
        # Run OWLv2 detection
        detections = detector.detect(frame, text_queries, threshold)
        
        # Update object tracking
        current_objects = tracker.update_tracks(
            frame, frame_idx, detections, embedding_extractor
        )
        
        # Run SAM2 segmentation for each tracked object
        for obj_id, box in current_objects.items():
            # Get segmentation mask
            mask = sam_wrapper.add_box(frame_idx, obj_id, box)
            
            # Save mask
            obj_dir = os.path.join(output_dir, "masks", f"object_{obj_id}")
            os.makedirs(obj_dir, exist_ok=True)
            
            import cv2
            mask_path = os.path.join(obj_dir, f"frame_{frame_idx:04d}.png")
            cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
            
            # Store tracking information
            tracking_results.append({
                "frame_idx": frame_idx,
                "object_id": obj_id,
                "class": tracker.tracked_objects[obj_id]["class"],
                "box": box,
                "mask_path": mask_path
            })
        
        # Create visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(frame)
        plt.title(f"Frame {frame_idx}")
        
        # Draw boxes and IDs for all current objects
        for obj_id, box in current_objects.items():
            obj_class = tracker.tracked_objects[obj_id]["class"]
            
            # Draw box
            x1, y1, x2, y2 = box
            plt.gca().add_patch(
                plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, 
                             edgecolor=f"C{obj_id % 10}", linewidth=2)
            )
            
            # Draw label
            plt.text(
                x1, y1-10, f"{obj_id}: {obj_class}", 
                bbox=dict(facecolor='white', alpha=0.8),
                color=f"C{obj_id % 10}", fontsize=12
            )
        
        # Save visualization
        plt.axis('off')
        plt.tight_layout()
        vis_path = os.path.join(output_dir, "visualizations", f"frame_{frame_idx:04d}.jpg")
        plt.savefig(vis_path)
        plt.close()
        
        # Optional: Run propagation periodically
        if frame_idx % 10 == 0 and frame_idx > 0:
            print("Running mask propagation...")
            sam_wrapper.propagate_masks()
    
    # Step 4: Save results
    results_json = {
        "video_info": {
            "total_frames": len(frames),
            "source_directory": frames_dir
        },
        "objects": {}
    }
    
    # Organize by object
    for obj_id, obj_data in tracker.tracked_objects.items():
        results_json["objects"][str(obj_id)] = {
            "class": obj_data["class"],
            "first_seen": obj_data["trajectory"][0][0],
            "last_seen": obj_data["last_seen"],
            "trajectory": {str(frame_idx): box for frame_idx, box in obj_data["trajectory"]}
        }
    
    # Save to file
    with open(os.path.join(output_dir, "tracking_results.json"), "w") as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nPipeline completed. Results saved to {output_dir}")
    print(f"Detected and tracked {len(tracker.tracked_objects)} objects across {len(frames)} frames")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OWLv2 + SAM2 Tracking Pipeline")
    parser.add_argument("--frames-dir", type=str, required=True, help="Directory with frames")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--text-queries", type=str, required=True, help="Comma-separated text queries")
    parser.add_argument("--sam-checkpoint", type=str, required=True, help="SAM2 checkpoint path")
    parser.add_argument("--sam-config", type=str, required=True, help="SAM2 config path")
    parser.add_argument("--threshold", type=float, default=0.1, help="Detection threshold")
    
    args = parser.parse_args()
    text_queries = [q.strip() for q in args.text_queries.split(",")]
    
    run_pipeline(
        args.frames_dir,
        args.output_dir,
        text_queries,
        args.sam_checkpoint,
        args.sam_config,
        args.threshold
    )