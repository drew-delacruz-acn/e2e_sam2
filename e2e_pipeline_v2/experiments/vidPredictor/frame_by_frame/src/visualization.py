import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

def show_mask(mask, ax, obj_id=None, random_color=False):
    """
    Display a segmentation mask on the given axes
    
    Args:
        mask: Binary mask to display
        ax: Matplotlib axes to draw on
        obj_id: Object ID for color selection (optional)
        random_color: Whether to use random color instead of class color
    """
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
    """
    Display points (positive and negative) on the given axes
    
    Args:
        coords: Coordinates of points
        labels: Labels for points (1 for positive, 0 for negative)
        ax: Matplotlib axes to draw on
        marker_size: Size of point markers
    """
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    """
    Display a bounding box on the given axes
    
    Args:
        box: Bounding box coordinates [x1, y1, x2, y2]
        ax: Matplotlib axes to draw on
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def visualize_segmentation_results(frames_dir, frame_names, video_segments, vis_frame_stride=1, save_path=None):
    """
    Visualize segmentation results
    
    Args:
        frames_dir: Directory containing video frames
        frame_names: List of frame filenames
        video_segments: Dictionary mapping frame indices to segmentation results
        vis_frame_stride: Stride for visualization (display every nth frame)
        save_path: Directory to save visualization frames (optional)
    """
    plt.close("all")
    all_figures = []
    
    # Create save directory if needed
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        print(f"Saving visualizations to: {save_path}")
    
    # Total frames to process
    total_frames = len(range(0, len(frame_names), vis_frame_stride))
    processed_frames = 0
    
    try:
        for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
            processed_frames += 1
            
            # Progress update every 10%
            if total_frames > 10 and processed_frames % (total_frames // 10) == 0:
                print(f"Visualizing frames: {processed_frames}/{total_frames} ({processed_frames/total_frames*100:.1f}%)")
            
            # Create figure
            fig = plt.figure(figsize=(10, 8), dpi=150)
            plt.title(f"Frame {out_frame_idx}")
            
            # Load and display frame
            frame_path = os.path.join(frames_dir, frame_names[out_frame_idx])
            plt.imshow(Image.open(frame_path))
            
            # Add segmentation masks
            if out_frame_idx in video_segments:
                for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                    show_mask(out_mask, plt.gca(), obj_id=int(out_obj_id))
            
            # Save figure if save_path is provided
            if save_path:
                save_file = os.path.join(save_path, f"frame_{out_frame_idx:04d}.png")
                plt.savefig(save_file, dpi=150, bbox_inches='tight', pad_inches=0.1, transparent=False)
                plt.close(fig)  # Close immediately to free memory
            else:
                all_figures.append(fig)
        
        print(f"Visualization complete: {processed_frames} frames processed")
        
        # Show figures if not saved to disk
        if not save_path and all_figures:
            plt.show()
        
        return all_figures
    
    finally:
        # Make sure to close all figures to avoid memory leaks
        plt.close("all") 