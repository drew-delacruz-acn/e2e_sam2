# Installation Guide

## Quick Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 2. Install Dependencies
```bash
pip install -r docs/requirements.txt
```

### 3. Install SAM2
```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### 4. Download Model Files
```bash
# From project root (e2e_sam2/)
mkdir -p checkpoints configs/sam2.1

# Download model checkpoint
wget -O checkpoints/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# Download config file
wget -O configs/sam2.1/sam2.1_hiera_l.yaml \
  https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml
```

### 5. Verify Installation
```bash
python -c "from sam2.build_sam import build_sam2_video_predictor; print('✅ Ready to go!')"
```

## Alternative Downloads
If `wget` doesn't work, download manually:
- [Model checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) → save as `checkpoints/sam2.1_hiera_large.pt`
- [Config file](https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml) → save as `configs/sam2.1/sam2.1_hiera_l.yaml`

That's it! See the main README for usage instructions. 