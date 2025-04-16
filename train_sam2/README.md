# SAM2 Fine-Tuning and Inference

This package provides code for fine-tuning the Segment Anything 2 (SAM2) model and running inference with the fine-tuned model.

## Directory Structure

```
train_sam2/
├── config.py            # Configuration parameters
├── models/              # Model-related code
│   ├── __init__.py
│   └── model_utils.py   # Model utilities
├── train.py             # Training script
├── inference.py         # Inference script
└── utils/               # Utility functions
    ├── __init__.py
    ├── data_utils.py    # Data loading and processing
    └── losses.py        # Loss functions
```

## Requirements

- PyTorch
- OpenCV
- NumPy
- Matplotlib
- SAM2 model and dependencies

## Training the Model

To fine-tune SAM2 on the DAVIS dataset:

```bash
python train.py --data_dir /path/to/davis/dataset --category bear --max_iters 10000
```

Additional parameters:
- `--lr`: Learning rate (default: 1e-5)
- `--weight_decay`: Weight decay (default: 4e-5)
- `--save_interval`: Checkpoint save interval (default: 1000)
- `--save_dir`: Directory to save checkpoints (default: "results")

## Inference with the Model

To run inference with a fine-tuned model:

```bash
python inference.py --image /path/to/image.jpg --mask /path/to/mask.png --checkpoint /path/to/fine_tuned_model.pt
```

Required parameters:
- `--image`: Path to the input image
- `--mask`: Path to the mask indicating the region to segment

Optional parameters:
- `--checkpoint`: Path to the fine-tuned model (if not specified, uses the original SAM2)
- `--points`: Number of points to sample (default: 30)
- `--output`: Output file path (default: "segmentation_output.png")

## Example Usage

1. Fine-tune the model:
   ```bash
   python train.py --data_dir data/davis-2017/DAVIS/ --category bear --max_iters 5000
   ```

2. Run inference:
   ```bash
   python inference.py --image sample_image.jpg --mask sample_mask.png --checkpoint results/model_latest.pt
   ``` 