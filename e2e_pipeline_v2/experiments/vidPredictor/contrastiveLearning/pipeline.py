# pipeline.py
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import LabelEncoder

from .data_io import load_frame
from .dataset import ContrastiveDataset
from .training import train_supcon
from .prototype import build_prototypes

def run_pipeline(
    src,
    method: str = "mean",
    log_metrics: bool = True,
    proj_dim: int = 128,
) -> pd.DataFrame:
    """
    Run the complete supervised contrastive learning pipeline.
    
    Parameters
    ----------
    src : pd.DataFrame or str
        Source DataFrame or path to pickle file containing embeddings.
    method : {"mean", "medoid"}, default="mean"
        Method to use for prototype generation.
    log_metrics : bool, default=True
        Whether to compute and log clustering metrics.
    proj_dim : int, default=128
        Output dimension for the projection head.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with representative embeddings for each class.
    """
    # Load data
    frame = load_frame(src)
    
    # Get embedding dimension
    # Use 'embedding' column for consistency throughout the pipeline
    embedding_col = "embedding" if "embedding" in frame.columns else "finetuned_embedding"
    vec_dim = len(frame[embedding_col].iloc[0])
    
    # Create dataset
    ds = ContrastiveDataset(frame)
    
    # Train projection head
    model = train_supcon(ds, vec_dim, epochs=10, proj_dim=proj_dim)

    # Get projected embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        proj = model(ds.vecs.to(device)).cpu().numpy()

    # Build prototypes
    reps = build_prototypes(frame, proj, method)

    # Compute metrics if requested
    if log_metrics:
        # Convert string labels to integers for metrics computation
        label_encoder = LabelEncoder()
        numeric_labels = label_encoder.fit_transform(frame["class"].values)
        
        try:
            sil = silhouette_score(proj, numeric_labels)      # cohesion/separation
            dbi = davies_bouldin_score(proj, numeric_labels)  # lower = better
            print(f"Silhouette={sil:.3f}  Davies-Bouldin={dbi:.3f}")
        except Exception as e:
            print(f"Error computing metrics: {e}")

    return reps 