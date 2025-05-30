import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple

# Column name constants
COL_CLASS = 'class'
COL_EMBEDDING = 'finetuned_embedding'
COL_VIDEO = 'video'
COL_FRAME = 'frame'
COL_VISUAL_PRED_OBJECT = 'visual_predicted_object'
COL_VISUAL_MAX_SCORE = 'visual_max_score'


def cosine_similarity_prediction(embedding: np.ndarray, 
                               class_embeddings: List[np.ndarray], 
                               class_names: List[str]) -> Tuple[str, float]:
    """
    Predict class using cosine similarity.
    """
    if not class_embeddings or not class_names: 
        print("⚠️ cosine_similarity_prediction called with empty class_embeddings or class_names.")
        return "unknown", 0.0 
    
    try:
        input_tensor = F.normalize(torch.tensor(embedding, dtype=torch.float32).unsqueeze(0), dim=1)
        if not class_embeddings: 
             return "unknown", 0.0
        class_tensor = F.normalize(torch.tensor(np.vstack(class_embeddings), dtype=torch.float32), dim=1)
        
        similarities = F.cosine_similarity(input_tensor, class_tensor).numpy().flatten()
        
        if similarities.size == 0: 
            print("⚠️ cosine_similarity_prediction produced empty similarities array.")
            return "unknown", 0.0

        best_idx = np.argmax(similarities)
        return class_names[best_idx], float(similarities[best_idx]) 
    except Exception as e:
        print(f"❌ Error in cosine_similarity_prediction: {e}")
        print(f"   Input embedding shape: {embedding.shape if isinstance(embedding, np.ndarray) else type(embedding)}")
        print(f"   Number of class_embeddings: {len(class_embeddings)}")
        if class_embeddings:
            print(f"   Shape of first class_embedding: {class_embeddings[0].shape if isinstance(class_embeddings[0], np.ndarray) else type(class_embeddings[0])}")
        return "unknown", 0.0


def generate_predictions(resnet_data: pd.DataFrame,
                        representatives_path: Path,
                        threshold: float,
                        iteration_number: int) -> pd.DataFrame:
    """
    Generate predictions using ONLY POSITIVE CLASS representatives.
    """
    print(f"🔮 Generating predictions (Iteration {iteration_number}) with threshold {threshold} using ONLY POSITIVE representatives...")
    
    try:
        with open(representatives_path, 'rb') as f:
            representatives_data = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ Representatives file not found: {representatives_path}")
        raise
    except Exception as e:
        print(f"❌ Error loading representatives file {representatives_path}: {e}")
        raise

    current_reps_df = None
    if isinstance(representatives_data, dict):
        try:
            current_reps_df = pd.DataFrame(representatives_data)
        except Exception as e:
            raise ValueError(f"Cannot convert dict representatives to DataFrame: {e}. Keys: {list(representatives_data.keys())}")
    elif isinstance(representatives_data, pd.DataFrame):
        current_reps_df = representatives_data.copy()
    else:
        raise ValueError(f"Unknown representatives format: {type(representatives_data)}. Expected dict or DataFrame.")

    if 'representative_embedding' in current_reps_df.columns and COL_EMBEDDING not in current_reps_df.columns:
        current_reps_df = current_reps_df.rename(columns={'representative_embedding': COL_EMBEDDING})
    
    if not all(col in current_reps_df.columns for col in [COL_CLASS, COL_EMBEDDING]):
        raise ValueError(f"Representatives DataFrame missing '{COL_CLASS}' or '{COL_EMBEDDING}'. Columns found: {list(current_reps_df.columns)}")

    positive_representatives_df = current_reps_df[~current_reps_df[COL_CLASS].str.startswith('not_', na=False)].copy()
    
    if positive_representatives_df.empty:
        print("⚠️ No positive class representatives found after filtering! Cannot make predictions.")
        expected_cols = list(resnet_data.columns) + [COL_VISUAL_PRED_OBJECT, COL_VISUAL_MAX_SCORE]
        return pd.DataFrame(columns=expected_cols)

    class_embeddings = list(positive_representatives_df[COL_EMBEDDING])
    class_names = list(positive_representatives_df[COL_CLASS])
    
    if not class_names: 
        print("⚠️ No class names available from positive representatives. Cannot make predictions.")
        expected_cols = list(resnet_data.columns) + [COL_VISUAL_PRED_OBJECT, COL_VISUAL_MAX_SCORE]
        return pd.DataFrame(columns=expected_cols)
        
    print(f"🏷️  Using {len(class_names)} POSITIVE representatives for prediction: {class_names[:10]}{ '...' if len(class_names)>10 else '' }")
    
    predictions_list = []
    if COL_EMBEDDING not in resnet_data.columns:
        raise ValueError(f"'{COL_EMBEDDING}' column missing from resnet_data. Columns: {list(resnet_data.columns)}")

    for _, row in resnet_data.iterrows():
        row_embedding = row[COL_EMBEDDING]
        if not isinstance(row_embedding, np.ndarray):
            print(f"⚠️ Skipping row due to invalid embedding type: {type(row_embedding)}. Video: {row.get(COL_VIDEO)}, Frame: {row.get(COL_FRAME)}")
            pred_class, confidence = "unknown", 0.0
        else:
            pred_class, confidence = cosine_similarity_prediction(
                row_embedding, 
                class_embeddings, 
                class_names
            )
        
        predictions_list.append({
            COL_VISUAL_PRED_OBJECT: pred_class, 
            COL_VISUAL_MAX_SCORE: confidence
        })
    
    pred_df = pd.DataFrame(predictions_list)
    result_df = pd.concat([resnet_data.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
    print(f"✅ Generated {len(result_df)} initial predictions using positive reps.")

    deduplicated_df = result_df 
    if not result_df.empty and COL_VISUAL_PRED_OBJECT in result_df.columns:
        valid_predictions_for_grouping = result_df[result_df[COL_VISUAL_PRED_OBJECT].notna()].reset_index(drop=True)
        if not valid_predictions_for_grouping.empty:
            try:
                idx = valid_predictions_for_grouping.groupby([COL_VIDEO, COL_VISUAL_PRED_OBJECT])[COL_VISUAL_MAX_SCORE].idxmax()
                deduplicated_df = valid_predictions_for_grouping.loc[idx].reset_index(drop=True)
                print(f"✅ Deduplicated to {len(deduplicated_df)} unique video-class predictions.")
            except KeyError as e:
                print(f"⚠️ KeyError during deduplication groupby: {e}")
                deduplicated_df = result_df 
        else:
            print("⚠️ No valid predictions for deduplication.")
            deduplicated_df = pd.DataFrame(columns=result_df.columns)
    else:
        print("⚠️ result_df is empty or 'visual_predicted_object' missing before deduplication.")
        deduplicated_df = pd.DataFrame(columns=list(resnet_data.columns) + [COL_VISUAL_PRED_OBJECT, COL_VISUAL_MAX_SCORE])

    final_df = deduplicated_df[deduplicated_df[COL_VISUAL_MAX_SCORE] >= threshold].copy()
    print(f"✅ {len(final_df)} predictions at/above threshold {threshold}.")
    return final_df 