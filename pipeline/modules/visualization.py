"""
Visualization utilities for detection and segmentation results.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2

class Visualizer:
    """Visualization utilities for detection and segmentation results"""
    
    def __init__(self, config):
        """
        Initialize the visualizer with configuration
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def show_detections(self, image, boxes, scores, labels, text_queries, save_path=None):
        """
        Display detection results with bounding boxes
        Args:
            image: PIL Image
            boxes: Bounding boxes
            scores: Confidence scores
            labels: Label indices
            text_queries: List of text queries
            save_path: Optional path to save the visualization
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(np.array(image))
        plt.axis('off')
        plt.title('Detected Objects')
        
        ax = plt.gca()
        colors = ['r', 'g', 'b', 'y', 'c', 'm']
        
        for box, score, label in zip(boxes, scores, labels):
            box = box.detach().cpu().numpy()
            x, y, x2, y2 = box
            width, height = x2 - x, y2 - y
            
            color = colors[label % len(colors)]
            rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            plt.text(x, y-10, f"{text_queries[label]}: {score:.2f}", color=color, fontsize=10, 
                     bbox=dict(facecolor='white', alpha=0.7))
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Detection visualization saved to {save_path}")
        
        plt.show()
    
    def show_mask(self, mask, ax, random_color=False, borders=True):
        """
        Display a single segmentation mask
        Args:
            mask: Binary mask
            ax: Matplotlib axis
            random_color: Whether to use random color
            borders: Whether to show borders
        """
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        
        if borders:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
        
        ax.imshow(mask_image)
    
    def show_points(self, coords, labels, ax, marker_size=375):
        """
        Display points used for segmentation
        Args:
            coords: Point coordinates
            labels: Point labels (0: negative, 1: positive)
            ax: Matplotlib axis
            marker_size: Size of markers
        """
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
                   s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
                   s=marker_size, edgecolor='white', linewidth=1.25)
    
    def show_box(self, box, ax):
        """
        Display a bounding box
        Args:
            box: Bounding box [x1, y1, x2, y2]
            ax: Matplotlib axis
        """
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    
    def show_segmentation(self, image, segmentation_results, score_threshold=0.5, save_dir=None):
        """
        Display segmentation results
        Args:
            image: PIL Image or numpy array
            segmentation_results: List of segmentation results
            score_threshold: Minimum score to display
            save_dir: Optional directory to save visualizations
        """
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for r_idx, result in enumerate(segmentation_results):
            for i, (mask, score) in enumerate(zip(result["masks"], result["scores"])):
                if score < score_threshold:
                    continue
                
                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                self.show_mask(mask, plt.gca(), borders=True)
                
                if "points" in result:
                    self.show_points(result["points"], result["point_labels"], plt.gca())
                
                if "box" in result:
                    self.show_box(result["box"], plt.gca())
                
                title = f"Mask {i+1}, Score: {score:.3f}"
                plt.title(title, fontsize=18)
                plt.axis('off')
                
                if save_dir:
                    save_path = os.path.join(save_dir, f"seg_result_{r_idx}_{i}.png")
                    plt.savefig(save_path, bbox_inches='tight')
                    print(f"Segmentation visualization saved to {save_path}")
                
                plt.show() 