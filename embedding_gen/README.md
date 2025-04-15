# Vision vs Semantic Embedding Models Comparison

This project conducts a rapid experiment to evaluate and compare non-semantic vision embedding models against multimodal models, focusing on their ability to preserve visual similarity. The experiment specifically examines how different models handle cases where visual and semantic similarities diverge.

## Project Overview

The experiment compares three types of models:
1. **ResNet-50**: A pure vision model that focuses on visual features
2. **ViT-B/16**: Vision Transformer model for visual feature extraction
3. **CLIP-ViT-B-32**: A multimodal model that understands both visual and semantic relationships

We analyze these models using carefully selected image pairs that have either:
- High visual similarity but low semantic similarity (e.g., red apple vs red ball)
- High semantic similarity but low visual similarity (e.g., standard hammer vs Thor's hammer)

## Directory Structure

```
visionVSemantic/
├── data/
│   ├── raw_images/      # Original downloaded images
│   └── processed/       # Resized and preprocessed images
├── src/
│   ├── models/         # Model loading and embedding generation
│   ├── preprocessing/  # Image preprocessing utilities
│   └── evaluation/     # Similarity computation and analysis
├── results/            # Similarity matrices and visualizations
├── requirements.txt    # Project dependencies
└── README.md          # This file
```

## Setup Instructions

1. Clone the repository:
```bash
git clone [repository-url]
cd visionVSemantic
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Prepare Images**
   - Place your original images in `data/raw_images/`
   - Images will be automatically processed and saved to `data/processed/`

2. **Generate Embeddings**
   ```bash
   python src/models/generate_embeddings.py
   ```

3. **Compute Similarities**
   ```bash
   python src/evaluation/compute_similarities.py
   ```

4. **View Results**
   - Results will be saved in the `results/` directory
   - Includes similarity matrices and visualizations
   - Analysis report will be generated as `results/analysis.pdf`

## Dependencies

Key dependencies include:
- PyTorch and torchvision for deep learning
- Transformers for CLIP and ViT models
- OpenCV and Pillow for image processing
- NumPy, Pandas, and scikit-learn for data analysis
- Matplotlib and seaborn for visualization

For a complete list, see `requirements.txt`.

## Expected Outcomes

The experiment will produce:
1. Embedding vectors for each image from all three models
2. Similarity matrices comparing image pairs
3. Visualizations of similarity relationships
4. Analysis report comparing model performance

## Contributing

Feel free to contribute to this project by:
1. Adding more model comparisons
2. Expanding the test image dataset
3. Improving visualization and analysis methods
4. Enhancing documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details. 