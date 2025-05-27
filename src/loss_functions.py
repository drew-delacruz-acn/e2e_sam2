"""
Loss functions for contrastive learning experiment.

This module implements:
1. Cosine similarity computation between representatives and embeddings
2. Contrastive loss function with pull and push terms
3. Mathematical operations for representative learning
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def cosine_similarity_matrix(representatives: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity matrix between representatives and embeddings.
    
    Args:
        representatives: Tensor of shape (num_classes, embedding_dim)
        embeddings: Tensor of shape (num_samples, embedding_dim)
        
    Returns:
        Similarity matrix of shape (num_classes, num_samples)
        where sim_matrix[i, j] = cosine_similarity(representatives[i], embeddings[j])
    """
    # Normalize vectors to unit length
    reps_normalized = F.normalize(representatives, p=2, dim=1)  # (num_classes, embedding_dim)
    embs_normalized = F.normalize(embeddings, p=2, dim=1)      # (num_samples, embedding_dim)
    
    # Compute cosine similarity via matrix multiplication
    # (num_classes, embedding_dim) @ (embedding_dim, num_samples) = (num_classes, num_samples)
    similarity_matrix = torch.mm(reps_normalized, embs_normalized.t())
    
    return similarity_matrix


def contrastive_loss(representatives: torch.Tensor, 
                    embeddings: torch.Tensor, 
                    labels: torch.Tensor, 
                    margin: float = 0.15, 
                    lambda_push: float = 0.25) -> torch.Tensor:
    """
    Compute contrastive loss for representative learning.
    
    The loss function is:
    L = Σ_c [ -mean(cos(r_c, P^c)) + λ * mean(max(0, cos(r_c, N^c) - margin)) ]
    
    Where:
    - r_c is the representative for class c
    - P^c are positive samples (same class as c)
    - N^c are negative samples (different class from c)
    - margin is the cosine similarity margin for negatives
    - λ (lambda_push) controls the strength of the push term
    
    Args:
        representatives: Tensor of shape (num_classes, embedding_dim)
        embeddings: Tensor of shape (num_samples, embedding_dim)
        labels: Tensor of shape (num_samples,) with class indices
        margin: Cosine similarity margin for negative samples
        lambda_push: Weight for the push term
        
    Returns:
        Scalar loss tensor
    """
    num_classes = representatives.shape[0]
    device = representatives.device
    
    # Compute cosine similarity matrix: (num_classes, num_samples)
    sim_matrix = cosine_similarity_matrix(representatives, embeddings)
    
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Compute loss for each class
    for class_idx in range(num_classes):
        # Get positive and negative masks for this class
        positive_mask = (labels == class_idx)
        negative_mask = (labels != class_idx)
        
        # Skip if no samples for this class
        if not positive_mask.any():
            continue
            
        # Get similarities for this class representative
        class_similarities = sim_matrix[class_idx]  # (num_samples,)
        
        # Pull term: maximize similarity with positives
        # -mean(cos(r_c, P^c))
        positive_similarities = class_similarities[positive_mask]
        pull_term = -torch.mean(positive_similarities)
        
        # Push term: minimize similarity with negatives beyond margin
        # mean(max(0, cos(r_c, N^c) - margin))
        push_term = torch.tensor(0.0, device=device)
        if negative_mask.any():
            negative_similarities = class_similarities[negative_mask]
            push_violations = torch.clamp(negative_similarities - margin, min=0.0)
            push_term = torch.mean(push_violations)
        
        # Combine terms for this class
        class_loss = pull_term + lambda_push * push_term
        total_loss = total_loss + class_loss
    
    return total_loss


def compute_loss_components(representatives: torch.Tensor, 
                           embeddings: torch.Tensor, 
                           labels: torch.Tensor, 
                           margin: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute pull and push loss components separately for analysis.
    
    Args:
        representatives: Tensor of shape (num_classes, embedding_dim)
        embeddings: Tensor of shape (num_samples, embedding_dim)
        labels: Tensor of shape (num_samples,) with class indices
        margin: Cosine similarity margin for negative samples
        
    Returns:
        Tuple of (pull_loss, push_loss)
    """
    num_classes = representatives.shape[0]
    device = representatives.device
    
    # Compute cosine similarity matrix
    sim_matrix = cosine_similarity_matrix(representatives, embeddings)
    
    total_pull = torch.tensor(0.0, device=device)
    total_push = torch.tensor(0.0, device=device)
    
    for class_idx in range(num_classes):
        positive_mask = (labels == class_idx)
        negative_mask = (labels != class_idx)
        
        if not positive_mask.any():
            continue
            
        class_similarities = sim_matrix[class_idx]
        
        # Pull term
        positive_similarities = class_similarities[positive_mask]
        pull_term = -torch.mean(positive_similarities)
        total_pull = total_pull + pull_term
        
        # Push term
        if negative_mask.any():
            negative_similarities = class_similarities[negative_mask]
            push_violations = torch.clamp(negative_similarities - margin, min=0.0)
            push_term = torch.mean(push_violations)
            total_push = total_push + push_term
    
    return total_pull, total_push 