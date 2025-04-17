# SAM2 Training Pipeline

This directory contains scripts for preparing datasets and fine-tuning the SAM2 (Segment Anything Model 2) on custom data.

## Directory Structure

- `prepare_dataset.py` - Script to prepare training and validation splits
- `train_sam.py` - Main training script with validation support
- `run_sam2_training.py` - Pipeline script to run data preparation and training
- `run_training.sh` - Bash wrapper that sets up a virtual environment and runs the pipeline

## Quick Start

The easiest way to run the training is with the bash wrapper script:

```bash
cd train_sam2
./run_training.sh
```

This will:
1. Create a virtual environment if it doesn't exist
2. Install dependencies
3. Check for .env file with API key
4. Run the training pipeline with default parameters

## Command Line Arguments

You can pass additional arguments to customize the training:

```bash
./run_training.sh --data_dir /path/to/davis/dataset --sequence bear --val_split 0.2 --max_iterations 500
```

### Available Arguments

- `--data_dir` - Path to DAVIS dataset (default: "../../../data/davis-2017/DAVIS/")
- `--sequence` - DAVIS sequence to use (default: "bear")
- `--val_split` - Validation split ratio (default: 0.2)
- `--output_dir` - Output directory for trained models (default: "./models")
- `--splits_dir` - Output directory for dataset splits (default: "./dataset_splits")
- `--sam2_checkpoint` - Path to SAM2 checkpoint (default: "../checkpoints/sam2.1_hiera_large.pt")
- `--model_cfg` - Path to model config (default: "configs/sam2.1/sam2.1_hiera_l.yaml")
- `--learning_rate` - Learning rate for optimizer (default: 1e-5)
- `--max_iterations` - Number of training iterations (default: 100)
- `--val_interval` - Validate model every N iterations (default: 10)
- `--save_interval` - Save model every N iterations (default: 50)

## Running Individual Scripts

If you prefer, you can run each script individually:

### 1. Prepare Dataset

```bash
python prepare_dataset.py --data_dir /path/to/davis/dataset --sequence bear --val_split 0.2
```

### 2. Train the Model

```bash
python train_sam.py --data_dir /path/to/davis/dataset --splits_dir ./dataset_splits
```

## Expected Dataset Structure

The scripts expect the DAVIS dataset with the following structure:

```
DAVIS/
├── JPEGImages/
│   └── 480p/
│       └── <sequence>/
│           └── *.jpg
└── Annotations/
    └── 480p/
        └── <sequence>/
            └── *.png
```

## Output

After training, the models will be saved in the `models` directory:
- `sam2_finetuned_<sequence>.torch` - Final model
- `sam2_finetuned_<sequence>_best.torch` - Best model according to validation IoU

Dataset splits are saved in the `dataset_splits` directory as JSON files. 