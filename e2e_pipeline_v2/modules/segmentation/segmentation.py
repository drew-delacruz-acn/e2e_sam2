"""
Image segmentation module using SAM2.
"""
import torch
import numpy as np
import os

# Now we can import directly from the installed package
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class Segmenter:
    """Image segmentation module using SAM2"""
    
    def __init__(self, config):
        """
        Initialize the segmenter with configuration
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = self._get_device()
        
        # Get absolute paths to model config and checkpoint
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(current_dir, "../.."))
        sam2_dir = os.path.join(root_dir, "sam2")
        
        model_config = os.path.join(config["segmentation"]["model_config"])
        checkpoint = os.path.join(config["segmentation"]["checkpoint"])
        
        print(f"Using model config: {model_config}")
        print(f"Using checkpoint: {checkpoint}")
        
        # Initialize SAM2 model
        self.sam2_model = build_sam2(
            model_config,
            checkpoint,
            device=self.device
        )
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        print(f"Segmenter initialized with device: {self.device}")
    
    def _get_device(self):
        """Determine the appropriate device for computation"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # CUDA optimizations
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )
        else:
            device = torch.device("cpu")
        
        return device
    
    def segment_with_boxes(self, image_np, boxes):
        """
        Segment the image using bounding boxes
        Args:
            image_np: Image as numpy array
            boxes: List of bounding boxes
        Returns:
            List of segmentation results with masks as NumPy arrays
        """
        results = []
        self.predictor.set_image(image_np)
        
        for box in boxes:
            box_tensor = box[None, :]
            with torch.inference_mode():
                masks, scores, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box_tensor,
                    multimask_output=False
                )
            
            # Convert PyTorch tensors to NumPy arrays
            numpy_masks = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else masks
            numpy_scores = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else scores
            
            results.append({
                "masks": numpy_masks,
                "scores": numpy_scores,
                "box": box
            })
        
        return results
    
    def segment_with_points(self, image_np, boxes):
        """
        Segment the image using points at the center of each box
        Args:
            image_np: Image as numpy array
            boxes: List of bounding boxes
        Returns:
            List of segmentation results with masks as NumPy arrays
        """
        results = []
        self.predictor.set_image(image_np)
        
        for box in boxes:
            # Calculate center point
            x1, y1, x2, y2 = [float(coord) for coord in box]
            x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Create point coordinates
            point_coords = np.array([[x_center, y_center]])
            point_labels = np.array([1])  # 1 indicates a foreground point
            
            with torch.inference_mode():
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True
                )
            
            # Convert PyTorch tensors to NumPy arrays
            numpy_masks = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else masks
            numpy_scores = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else scores
            
            results.append({
                "masks": numpy_masks,
                "scores": numpy_scores,
                "points": point_coords,
                "point_labels": point_labels,
                "box": box
            })
        
        return results
    
    def predict_with_box(self, image, box):
        """
        Predict segmentation mask using SAM2 with a bounding box input
        Args:
            image: RGB image as numpy array
            box: Bounding box in format [x1, y1, x2, y2]
        Returns:
            masks: Predicted segmentation masks
            scores: Confidence scores
        """
        # Set image for prediction
        self.predictor.set_image(image)
        
        # Convert box to torch tensor and correct format
        box_torch = torch.tensor(box, device=self.predictor.device)[None, :]
        
        # Get prediction
        with torch.inference_mode():
            masks, scores, logits = self.predictor.predict(
                point_coords=None,
                point_labels=None, 
                boxes=box_torch,
                multimask_output=False
            )
        
        return masks[0], scores[0]
