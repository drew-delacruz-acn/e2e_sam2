import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from PIL import Image


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

from sam2.build_sam import build_sam2_video_predictor
detections_dir = "/home/ubuntu/code/drew/e2e_sam2/data/detections_cdfsod/Scenes 061-080__265H-2-_20230815215828529/"
frames_dir = "/home/ubuntu/code/drew/e2e_sam2/data/frames/Scenes 061-080__265H-2-_20230815215828529/"


sam2_checkpoint = "../../../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(frames_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

inference_state = predictor.init_state(video_path=frames_dir)
predictor.reset_state(inference_state)

def natural_sort_key(s):
    """
    Sort strings containing numbers naturally (e.g. frame_1, frame_2, frame_10).
    """
    import re
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(s))]

def load_detection_files(detections_dir: str) -> Dict[int, List[Dict[str, Any]]]:
    """
    Load all detection JSON files from the directory
    
    Args:
        detections_dir: Path to directory containing detection JSON files
        
    Returns:
        Dictionary mapping frame indices to lists of detections
    """
    detections_path = Path(detections_dir)
    detection_files = sorted([f for f in detections_path.glob("*.json")], key=natural_sort_key)
    
    detections_by_frame = {}
    for file_path in detection_files:
        try:
            # Extract frame index from filename
            frame_idx = int(file_path.stem)
            
            # Load detections
            with open(file_path, 'r') as f:
                detections = json.load(f)
            
            # Store by frame
            detections_by_frame[frame_idx] = detections
            
        except ValueError:
            print(f"Skipping file {file_path} - could not parse frame index")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    print(f"Loaded {len(detections_by_frame)} detection files")
    return detections_by_frame

detections = load_detection_files(detections_dir)

def send_sam2_api(ann_frame_idx_VAR, ann_obj_id_VAR, box):
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx_VAR,
        obj_id=ann_obj_id_VAR,
        box=box,
        # labels=np.array(labelsVar, np.int32),
    )


def filter_detections_by_confidence(detections_by_frame, confidence_threshold=0.5):
    """
    Filter detections dictionary to keep only those with confidence above threshold
    
    Args:
        detections_by_frame: Dictionary mapping frame numbers to lists of detections
        confidence_threshold: Minimum confidence score to keep a detection (default: 0.5)
        
    Returns:
        Filtered dictionary with only high-confidence detections
    """
    filtered_detections = {}
    
    total_detections = 0
    kept_detections = 0
    
    for frame_num, detections in detections_by_frame.items():
        # Filter detections for this frame
        high_confidence_detections = []
        
        for detection in detections:
            total_detections += 1
            confidence = detection.get('confidence', 0.0)
            
            if confidence >= confidence_threshold:
                high_confidence_detections.append(detection)
                kept_detections += 1
        
        # Only add frame if it has any detections after filtering
        if high_confidence_detections:
            filtered_detections[frame_num] = high_confidence_detections
        else:
            # Keep the frame with empty list to maintain frame sequence
            filtered_detections[frame_num] = []
    
    # Print statistics
    if total_detections > 0:
        percentage_kept = (kept_detections / total_detections) * 100
        print(f"Filtered detections: kept {kept_detections}/{total_detections} ({percentage_kept:.1f}%)")
        print(f"Confidence threshold: {confidence_threshold}")
    else:
        print("No detections found to filter")
    
    return filtered_detections

def convert_detections_to_tracking_format(detections_by_frame):
    """
    Convert detection dictionary to tracking format with bounding boxes
    
    Args:
        detections_by_frame: Dictionary mapping frame numbers to lists of detections
        
    Returns:
        List of objects in tracking format
    """
    # Track objects by label to assign consistent IDs
    object_ids = {}
    next_id = 1
    
    objects = []
    
    # Process each frame's detections
    for frame_num, detections in sorted(detections_by_frame.items()):
        for detection in detections:
            label = detection.get('label', 'unknown')
            confidence = detection.get('confidence', 0.0)
            coords = detection.get('coordinates', [])
            
            # Skip detections without proper coordinates
            if len(coords) != 4:
                continue
                
            # Get or assign object ID
            if label not in object_ids:
                object_ids[label] = next_id
                next_id += 1
            
            object_id = object_ids[label]
            
            # Find or create object entry
            obj = next((o for o in objects if o['objectName'] == label and o['objectID'] == object_id), None)
            
            if obj is None:
                obj = {
                    'objectName': label,
                    'objectID': object_id,
                    'frameOccurences': []
                }
                objects.append(obj)
            
            # Add frame occurrence with bounding box
            obj['frameOccurences'].append({
                'frameNum': frame_num,
                'box': coords,  # [x1, y1, x2, y2]
                'confidence': confidence
            })
    
    return objects


# Load the detections
detections_by_frame = load_detection_files(detections_dir)

# Filter with 0.7 confidence threshold
filtered_detections = filter_detections_by_confidence(detections_by_frame, confidence_threshold=0.9)

# Convert to tracking format
tracking_objects = convert_detections_to_tracking_format(filtered_detections)

allFramesClasses = []
for objectClass in tracking_objects:
    for occurence in objectClass['frameOccurences']:
        if occurence['frameNum'] not in allFramesClasses:
            allFramesClasses.append(occurence['frameNum']) 

for frameNum in allFramesClasses:
    for objectClass in tracking_objects:
        print(f'object: {objectClass["objectName"]}')
        validOccurences = [w for w in objectClass['frameOccurences'] if w['frameNum'] == frameNum]
        if len(validOccurences):
            occ = validOccurences[0] # TODO - how would we handle multiple instances of a class - e.g., multiple temppads 
            print(f'occurence {occ["box"]}')
            send_sam2_api(frameNum, objectClass['objectID'], occ['box'])
        else:
            print(f"{objectClass['objectName']} NOT in frame {frameNum}")
            send_sam2_api(frameNum, objectClass['objectID'], [-1,-1,-1, -1])



# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse = False):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }


# render the segmentation results every few frames
vis_frame_stride = 1
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(frames_dir, frame_names[out_frame_idx])))
    if out_frame_idx in video_segments:
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)