import os
import numpy as np
import cv2
import torch

def get_dataset_files(data_dir, category="bear"):
    """Load dataset file paths from the DAVIS dataset"""
    data = []
    image_dir = os.path.join(data_dir, "JPEGImages/480p/", category)
    annot_dir = os.path.join(data_dir, "Annotations/480p/", category)
    
    for filename in os.listdir(image_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, filename)
            annotation_path = os.path.join(annot_dir, filename[:-4] + ".png")
            if os.path.exists(annotation_path):
                data.append({"image": image_path, "annotation": annotation_path})
    
    return data

def read_batch(data):
    """Read random image and its annotation from the dataset"""
    ent = data[np.random.randint(len(data))]  # choose random entry
    img = cv2.imread(ent["image"])[...,::-1]  # read image as RGB
    ann_map = cv2.imread(ent["annotation"])  # read annotation

    # resize image
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])  # scaling factor
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)

    # merge vessels and materials annotations
    mat_map = ann_map[:,:,0]  # material annotation map
    ves_map = ann_map[:,:,2]  # vessel annotation map
    mat_map[mat_map==0] = ves_map[mat_map==0]*(mat_map.max()+1)  # merge maps

    # Get binary masks and points
    inds = np.unique(mat_map)[1:]  # load all indices
    points = []
    masks = []
    for ind in inds:
        mask = (mat_map == ind).astype(np.uint8)  # make binary mask corresponding to index ind
        masks.append(mask)
        coords = np.argwhere(mask > 0)  # get all coordinates in mask
        if len(coords) > 0:
            yx = np.array(coords[np.random.randint(len(coords))])  # choose random point/coordinate
            points.append([[yx[1], yx[0]]])
        
    return img, np.array(masks), np.array(points), np.ones([len(masks), 1])

def read_image_for_inference(image_path, mask_path, max_size=1024):
    """Read and resize image and mask for inference"""
    img = cv2.imread(image_path)[...,::-1]  # read image as RGB
    mask = cv2.imread(mask_path, 0)  # mask of the region we want to segment
    
    # Resize image to maximum size
    r = np.min([max_size / img.shape[1], max_size / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
    
    return img, mask

def get_points_from_mask(mask, num_points):
    """Sample points inside the input mask"""
    points = []
    if np.sum(mask > 0) > 0:  # ensure the mask has positive pixels
        for i in range(num_points):
            coords = np.argwhere(mask > 0)
            if len(coords) > 0:
                yx = np.array(coords[np.random.randint(len(coords))])
                points.append([[yx[1], yx[0]]])
    
    return np.array(points)

def visualize_segmentation(seg_map):
    """Visualize segmentation map with random colors"""
    rgb_image = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    for id_class in range(1, seg_map.max()+1):
        rgb_image[seg_map == id_class] = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]
    
    return rgb_image 