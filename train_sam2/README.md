# SAM2 Fine-tuning and Inference

Scripts for fine-tuning and running inference with the Segment Anything 2 (SAM2) model.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the SAM2 checkpoint:
Make sure you have the SAM2 model checkpoint files and config in appropriate locations.

## Training

The `train_sam.py` script fine-tunes the SAM2 model on your dataset.

```bash
python train_sam.py --data_dir /path/to/dataset \
                   --sam2_checkpoint /path/to/checkpoint \
                   --model_cfg /path/to/config \
                   --max_iterations 10000 \
                   --save_interval 1000 \
                   --output_model fine_tuned_model.torch
```

### Training Arguments

- `--data_dir`: Path to dataset
- `--sam2_checkpoint`: Path to SAM2 checkpoint
- `--model_cfg`: Path to model config
- `--learning_rate`: Learning rate for optimizer (default: 1e-5)
- `--weight_decay`: Weight decay for optimizer (default: 4e-5)
- `--max_iterations`: Number of training iterations (default: 10000)
- `--save_interval`: Save model every N iterations (default: 1000)
- `--output_model`: Output model path (default: model.torch)

## Inference

The `sam2_fintune_inference.py` script runs inference with the fine-tuned SAM2 model.

```bash
python sam2_fintune_inference.py --image_path /path/to/image.jpg \
                               --mask_path /path/to/mask.png \
                               --sam2_checkpoint /path/to/fine_tuned_model.torch \
                               --model_cfg /path/to/config \
                               --num_points 30 \
                               --output_path segmentation_result.png
```

### Inference Arguments

- `--image_path`: Path to input image
- `--mask_path`: Path to mask defining region to segment
- `--sam2_checkpoint`: Path to SAM2 checkpoint
- `--model_cfg`: Path to model config
- `--num_points`: Number of points to sample from the mask (default: 30)
- `--output_path`: Path to save output segmentation (default: output_segmentation.png)

## Dataset Format

The scripts are configured to work with the DAVIS dataset format. For other datasets, you may need to adjust the data loading functions.

## Notes

- The scripts default to using GPU with mixed precision if available
- For Apple Silicon Macs, MPS backend is used if CUDA is not available
- The training script logs IOU accuracy during training
- The inference script generates a visualization with colored segments 