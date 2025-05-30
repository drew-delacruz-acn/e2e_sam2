"""
Data utilities for SAM2 training and inference.
"""
import os
import json
import numpy as np
import cv2
import random


def load_dataset_splits(data_dir, splits_dir):
    """Load dataset splits for training and validation"""
    dataset = {}
    
    # Load train and validation splits
    for split in ['train', 'val']:
        split_file = os.path.join(splits_dir, f"{split}.json")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        # Prepend the data_dir to each path
        for item in split_data:
            item['image'] = os.path.join(data_dir, item['image'])
            item['annotation'] = os.path.join(data_dir, item['annotation'])
        
        dataset[split] = split_data
        print(f"Loaded {len(split_data)} samples for {split} split")
    
    return dataset


def read_batch(data):
    """Read random image and its annotation from the dataset"""
    # Select image
    entry = data[np.random.randint(len(data))]  # Choose random entry
    img = cv2.imread(entry["image"])[...,::-1]  # Read image as RGB
    ann_map = cv2.imread(entry["annotation"])  # Read annotation

    # Resize image
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])  # Scaling factor
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), 
                        interpolation=cv2.INTER_NEAREST)

    # Merge vessels and materials annotations
    mat_map = ann_map[:,:,0]  # Material annotation map
    ves_map = ann_map[:,:,2]  # Vessel annotation map
    mat_map[mat_map==0] = ves_map[mat_map==0]*(mat_map.max()+1)  # Merge maps

    # Get binary masks and points
    inds = np.unique(mat_map)[1:]  # Load all indices
    points = []
    masks = []
    for ind in inds:
        mask = (mat_map == ind).astype(np.uint8)  # Make binary mask for index ind
        masks.append(mask)
        coords = np.argwhere(mask > 0)  # Get all coordinates in mask
        yx = np.array(coords[np.random.randint(len(coords))])  # Choose random point
        points.append([[yx[1], yx[0]]])
    
    return img, np.array(masks), np.array(points), np.ones([len(masks), 1])


def get_points(mask, num_points):
    """Sample points inside the input mask"""
    points = []
    if np.sum(mask > 0) == 0:
        print("Warning: Mask is empty, cannot sample points")
        return np.array(points)
        
    for i in range(num_points):
        coords = np.argwhere(mask > 0)
        yx = np.array(coords[np.random.randint(len(coords))])
        points.append([[yx[1], yx[0]]])
    return np.array(points)


def load_dataset_paths(data_dir, sequence):
    """Load dataset file paths for a specific sequence"""
    data = []
    img_dir = os.path.join(data_dir, f"JPEGImages/480p/{sequence}/")
    ann_dir = os.path.join(data_dir, f"Annotations/480p/{sequence}/")
    
    # Check if directories exist
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not os.path.exists(ann_dir):
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")
    
    # Go over all files in the dataset
    for name in os.listdir(img_dir):
        if not name.endswith('.jpg'):
            continue
            
        ann_name = name[:-4] + ".png"
        if not os.path.exists(os.path.join(ann_dir, ann_name)):
            print(f"Warning: No annotation found for {name}")
            continue
            
        data.append({
            "image": os.path.join("JPEGImages/480p", sequence, name),
            "annotation": os.path.join("Annotations/480p", sequence, ann_name)
        })
    
    print(f"Found {len(data)} image-annotation pairs for sequence '{sequence}'")
    return data


def create_dataset_splits(data, val_split, seed):
    """Split dataset into training and validation sets"""
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Shuffle the data
    random.shuffle(data)
    
    # Calculate split index
    split_idx = int(len(data) * (1 - val_split))
    
    # Split data
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"Created splits: {len(train_data)} training, {len(val_data)} validation samples")
    
    return {
        "train": train_data,
        "val": val_data
    }


def save_splits(splits, output_dir):
    """Save dataset splits to disk"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits to JSON files
    for split_name, split_data in splits.items():
        output_file = os.path.join(output_dir, f"{split_name}.json")
        with open(output_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved {split_name} split to {output_file}")


def read_image(image_path, mask_path, gt_mask_path=None):
    """Read and resize image and mask for inference"""
    img = cv2.imread(image_path)[...,::-1]  # Read image as RGB
    mask = cv2.imread(mask_path, 0)  # Mask of the region to segment
    
    # Resize image to maximum size of 1024
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), 
                     interpolation=cv2.INTER_NEAREST)
    
    # Load ground truth mask if provided
    gt_mask = None
    if gt_mask_path:
        gt_mask = cv2.imread(gt_mask_path, 0)
        if gt_mask is not None:
            gt_mask = cv2.resize(gt_mask, (int(mask.shape[1]), int(mask.shape[0])),
                             interpolation=cv2.INTER_NEAREST)
            # Convert to binary mask if needed
            if gt_mask.max() > 1:
                gt_mask = (gt_mask > 0).astype(np.uint8)
    
    return img, mask, gt_mask 