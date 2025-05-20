# prototype.py
import numpy as np
import pandas as pd
from typing import Literal
from sklearn.metrics import pairwise_distances_argmin

def build_prototypes(
    frame: pd.DataFrame,
    proj_vecs: np.ndarray,
    method: Literal["mean", "medoid"] = "mean",
) -> pd.DataFrame:
    """
    Build representative class prototypes from projected embeddings.
    
    Parameters
    ----------
    frame : pd.DataFrame
        DataFrame with 'class' column.
    proj_vecs : np.ndarray
        Projected embeddings corresponding to each row in the DataFrame.
    method : {"mean", "medoid"}, default="mean"
        Method to use for prototype generation.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['class', 'representative_embedding'].
    """
    df = frame[["class"]].copy()
    df["proj"] = list(proj_vecs)

    reps = []
    for cid, group in df.groupby("class"):
        arr = np.vstack(group["proj"].values)
        if method == "mean":
            proto = arr.mean(axis=0)
        else:  # medoid
            c_idx = pairwise_distances_argmin(arr.mean(0, keepdims=True), arr)[0]
            proto = arr[c_idx]
        # Normalize the prototype
        proto /= np.linalg.norm(proto) + 1e-9
        reps.append({"class": cid, "representative_embedding": proto})
    return pd.DataFrame(reps) 