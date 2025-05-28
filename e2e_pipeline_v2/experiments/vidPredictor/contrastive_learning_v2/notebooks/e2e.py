import os
import json
import time
import requests
import numpy as np
import twelvelabs
from dotenv import load_dotenv
from twelvelabs.models.embed import SegmentEmbedding
from typing import List
import requests
import io
from PIL import Image, ImageOps
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pickle
load_dotenv()


with open("/home/ubuntu/code/libby/pipeline/data/finetuned_may19.pkl", 'rb') as file:
    resnet_df = pickle.load(file)
    
resnet_df = resnet_df[['video', 'frame', 'owl_label', 'finetuned_embedding']]
resnet_df

with open("/home/ubuntu/code/contrastive_results/contrastive_embeddings_viz_e75_t0.7_dim2048/prototypes.pkl", 'rb') as file:
    defObjects = pickle.load(file)

class_emb_matrix = list(defObjects['representative_embedding'])
objs = list(defObjects['class'])

def cosine_scores(input_emb, class_emb_matrix):
    input_tensor = F.normalize(torch.tensor(input_emb, dtype=torch.float32).unsqueeze(0), dim=1)
    class_tensor = F.normalize(torch.tensor(np.vstack(class_emb_matrix), dtype=torch.float32), dim=1)
    return F.cosine_similarity(input_tensor, class_tensor).numpy()



def visual_prediction_from_text_filtered(row):
    visual_scores = cosine_scores(row['finetuned_embedding'], class_emb_matrix)
    best_idx = np.argmax(visual_scores)
    return pd.Series({
        'visual_predicted_object': objs[best_idx],
        'visual_max_score': visual_scores[best_idx],
    })

visual_preds = resnet_df.apply(visual_prediction_from_text_filtered, axis=1)
resnet_df[['visual_predicted_object', 'visual_max_score']] = visual_preds


visual_threshold = 0.45 # THIS WILL CHANGE -- NEED TO TEST FOR YOUR STUFF
resnet_df['prediction'] = 1
resnet_df['frame'] = resnet_df['frame'].apply(lambda row: int(row))

# idx = filtered_df_copy.groupby(['video', 'visual_predicted_object'])['prediction'].idxmax()
idx = resnet_df.groupby(['video', 'visual_predicted_object'])['visual_max_score'].idxmax()
resnet_df = resnet_df.loc[idx].reset_index(drop=True)

resnet_df['visual_predicted_object'] = resnet_df.apply(lambda row: row['visual_predicted_object'] if row['visual_max_score'] > visual_threshold else 'No Class', axis=1)
resnet_df = resnet_df[resnet_df['visual_predicted_object'] != 'No Class']


with open("/home/ubuntu/code/libby/pipeline/data/sourceTruth_jeremiah.pkl", 'rb') as file:
    final_SOT = pickle.load(file)
    
final_SOT = final_SOT.rename(columns={'frame': 'second', 'second': 'frame'})
final_SOT = final_SOT.groupby(['video', 'tag'])['actual'].max().reset_index()
# final_SOT['frame'] = final_SOT['frame'].apply(lambda row: int(row))

merged = pd.merge(resnet_df, final_SOT, left_on=['video', 'visual_predicted_object'], right_on=['video', 'tag'], how='right')

def classify_answer(answer, pred):
    if answer == 0 and pred == 0:
        return 'TN'
    elif answer == 1 and pred == 1:
        return 'TP'
    elif answer == 1 and pred == 0:
        return 'FN'
    return 'FP'
merged['prediction'] = merged['prediction'].fillna(0)
merged['answerClass'] =  merged.apply(lambda row: classify_answer(row['actual'], row['prediction']), axis = 1)


counts = merged['answerClass'].value_counts()
TP = counts.get('TP', 0)
FP = counts.get('FP', 0)
FN = counts.get('FN', 0)
TN = counts.get('TN', 0)

# Calculate metrics
print('TP: ', TP)
print('FP: ', FP)
print('FN: ', FN)
print('TN: ', TN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# Display
print(f'Precision: {precision:.4f}')
print(f'Recall:    {recall:.4f}')
print(f'F1 Score:  {f1_score:.4f}')

alx = merged.groupby(["tag","answerClass"]).size().reset_index().pivot(columns='answerClass', values=0, index = 'tag').fillna(0).reset_index().sort_values('TP')
alx['precision'] = alx['TP'] / (alx['TP'] + alx['FP']) 
alx['recall'] = alx['TP'] / (alx['TP'] + alx['FN']) 
alx['f-score'] = (alx['precision']*alx['recall']/(alx['precision'] + alx['recall']))*2
 
alx['Total Labels Across Videos'] = alx['TP'] + alx['FN']
 
alx.fillna(0).sort_values('f-score', ascending = False)