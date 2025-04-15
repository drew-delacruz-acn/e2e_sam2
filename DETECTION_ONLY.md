# Object Detection Using Owlv2 (No SAM2)

If you're having issues with the SAM2 segmentation part of the pipeline, or simply want to run just the object detection component, this guide explains how to use only the Owlv2 detection functionality from our pipeline.

## Using Detection Only

### Option 1: Create a Simple Detection-Only Script

Create a file called `detect_only.py` with the following content:

```python
"""
Script to run only the object detection part of the pipeline
"""
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pipeline.config import get_config
from pipeline.modules.detection import ObjectDetector

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run object detection using Owlv2")
    parser.add_argument("--image", type=str, default="data/thor_hammer.jpeg", 
                        help="Path to input image")
    parser.add_argument("--queries", type=str, nargs="+", default=["hammer"], 
                        help="Text queries for detection")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save results to disk")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--detection-threshold", type=float, default=0.1,
                        help="Detection confidence threshold")
    return parser.parse_args()

def visualize_detections(image, boxes, scores, labels, text_queries, save_path=None):
    """Visualize detection results with bounding boxes"""
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Detection visualization saved to {save_path}")
    
    plt.show()

def save_crops(image, boxes, output_dir):
    """Save cropped images from bounding boxes"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    crop_paths = []
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cropped_image = image.crop((x1, y1, x2, y2))
        
        output_path = os.path.join(output_dir, f"box_{idx}.png")
        cropped_image.save(output_path)
        crop_paths.append(output_path)
    
    return crop_paths

def main():
    """Main function to run detection"""
    args = parse_args()
    
    # Custom configuration for detection
    config_override = {
        "detection": {
            "threshold": args.detection_threshold
        },
        "paths": {
            "results_dir": args.results_dir
        }
    }
    
    # Get configuration
    config = get_config(config_override)
    
    # Initialize the detector
    detector = ObjectDetector(config)
    
    # Run detection
    detection_results = detector.detect(
        image_path=args.image,
        text_queries=args.queries
    )
    
    # Extract results
    image = detection_results["image"]
    boxes = detection_results["boxes"]
    scores = detection_results["scores"]
    labels = detection_results["labels"]
    text_queries = detection_results["text_queries"]
    
    # Visualize and save results if requested
    if args.save:
        # Create directories
        vis_dir = os.path.join(args.results_dir, "visualizations")
        crops_dir = os.path.join(args.results_dir, "crops")
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(crops_dir, exist_ok=True)
        
        # Visualize detections
        detection_vis_path = os.path.join(vis_dir, "detection_results.png")
        visualize_detections(
            image, boxes, scores, labels, text_queries,
            save_path=detection_vis_path
        )
        
        # Save cropped objects
        crop_paths = save_crops(image, boxes, crops_dir)
        print(f"Saved {len(crop_paths)} cropped images to {crops_dir}")
    else:
        # Just visualize without saving
        visualize_detections(image, boxes, scores, labels, text_queries)
    
    print(f"Detection completed successfully!")

if __name__ == "__main__":
    main()
```

Run this script to perform detection only:

```bash
python detect_only.py --image data/my_image.jpg --queries "person" "dog"
```

### Option 2: Modify the Pipeline Configuration

You can also use the existing pipeline but configure it to skip the segmentation steps:

```python
from pipeline.pipeline import Pipeline

# Create a special config that will effectively disable segmentation
config_override = {
    "detection": {
        "threshold": 0.1,  # Adjust as needed
    },
    "segmentation": {
        "skip": True  # You'll need to add handling for this in pipeline.py
    }
}

# Initialize and run the pipeline
pipeline = Pipeline(config_override=config_override)
results = pipeline.run(
    image_path="data/my_image.jpg",
    text_queries=["person", "dog"],
    visualize=True,
    save_results=True
)
```

## Dependencies for Detection Only

The detection part only requires a subset of the full pipeline dependencies:

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.22.0
Pillow>=9.0.0
matplotlib>=3.5.0
transformers==4.39.0
```

You can install these with:

```bash
pip install torch torchvision numpy Pillow matplotlib transformers==4.39.0
```

## Benefits of Using Only Detection

1. **Simpler Dependencies**: You don't need to install or configure SAM2
2. **Faster Inference**: Object detection with Owlv2 is faster than running the full segmentation pipeline
3. **Lower Memory Usage**: The detection model requires less VRAM than SAM2
4. **Fewer Import Issues**: Avoids potential issues with SAM2 package imports

## How Detection Works

The `ObjectDetector` class from our pipeline:

1. Loads the Owlv2 model from Hugging Face Transformers
2. Processes the image with text queries
3. Obtains bounding box predictions
4. Filters overlapping boxes using IoU (Intersection over Union)
5. Returns the filtered results

The results include:
- Image data
- Bounding boxes for detected objects
- Confidence scores
- Label indices
- Text queries

## Example Detection Workflow

1. **Detect Objects**:
   ```bash
   python detect_only.py --image data/my_image.jpg --queries "person" "dog"
   ```

2. **Review Results**:
   - Check the visualizations showing detected objects with bounding boxes
   - Examine the cropped objects saved in the results directory
   - Look at the printed detection information in the terminal

3. **Adjust the Detection Threshold**:
   ```bash
   python detect_only.py --image data/my_image.jpg --queries "person" "dog" --detection-threshold 0.2
   ```

## Common Issues

1. **No Objects Detected**:
   - Try lowering the detection threshold (`--detection-threshold 0.05`)
   - Try different text queries (Owlv2 works best with simple object categories)
   - Make sure the objects are clearly visible in the image

2. **Missing Dependencies**:
   - Install the required packages: `pip install transformers==4.39.0 torch torchvision`

3. **Model Download Issues**:
   - Ensure you have internet connectivity (the model is downloaded from Hugging Face)
   - Check Hugging Face token permissions if using a private model

4. **File Not Found Errors**:
   - Make sure the image path is correct relative to your current directory 