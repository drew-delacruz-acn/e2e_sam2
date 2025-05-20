from pathlib import Path
from typing import Union
import pandas as pd

def load_frame(src: Union[pd.DataFrame, str | Path]) -> pd.DataFrame:
    """
    Parameters
    ----------
    src : DataFrame | str | Path
        Either an in-memory DataFrame or a path to a Pickle file.

    Returns
    -------
    pd.DataFrame
        Must contain columns ['class', 'embedding'] where
        'embedding' holds a 1-D numpy array or list.
    """
    if isinstance(src, pd.DataFrame):
        return src.copy()
    src = Path(src)
    if src.suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(src)
    raise ValueError(f"Unsupported source: {src}") 