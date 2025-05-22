"""
Trainer module for contrastive learning experiment.

This module implements:
1. ContrastiveTrainer class for managing training process
2. Representative initialization and optimization
3. Training loop with loss computation and backpropagation
4. Evaluation using F1 score with nearest representative classification
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.metrics import f1_score
import warnings

from .loss_functions import contrastive_loss, cosine_similarity_matrix


class ContrastiveTrainer:
    """
    Trainer for contrastive representative learning.
    
    This class manages the training process including:
    - Representative initialization as class means
    - Optimization with Adam optimizer
    - Training loop with loss computation
    - Evaluation using nearest representative classification
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trainer with configuration.
        
        Args:
            config: Dictionary containing training parameters:
                - lr: Learning rate (required)
                - margin: Cosine similarity margin (required)
                - lambda_push: Weight for push term (required)
                - epochs: Number of training epochs (optional)
                
        Raises:
            ValueError: If required config parameters are missing or invalid
        """
        self.config = config.copy()
        
        # Validate required parameters
        required_params = ['lr', 'margin', 'lambda_push']
        missing_params = [param for param in required_params if param not in config]
        if missing_params:
            raise ValueError(f"Missing required config parameters: {missing_params}")
        
        # Validate parameter values
        if config['lr'] <= 0:
            raise ValueError("Learning rate must be positive")
        if config['margin'] < 0:
            raise ValueError("Margin must be non-negative")
        if config['lambda_push'] < 0:
            raise ValueError("Lambda push must be non-negative")
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize placeholders
        self.representatives = None
        self.optimizer = None
        
    def initialize_representatives(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Initialize representatives as class means.
        
        Args:
            embeddings: Tensor of shape (num_samples, embedding_dim)
            labels: Tensor of shape (num_samples,) with class indices
        """
        embeddings = embeddings.to(self.device)
        labels = labels.to(self.device)
        
        num_classes = len(torch.unique(labels))
        embedding_dim = embeddings.shape[1]
        
        # Initialize representatives as class means
        self.representatives = torch.zeros(num_classes, embedding_dim, 
                                         device=self.device, requires_grad=True)
        
        for class_idx in range(num_classes):
            class_mask = (labels == class_idx)
            if class_mask.any():
                class_embeddings = embeddings[class_mask]
                self.representatives.data[class_idx] = class_embeddings.mean(dim=0)
        
        # Initialize optimizer
        self.optimizer = optim.Adam([self.representatives], lr=self.config['lr'])
    
    def train_step(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Perform one training step.
        
        Args:
            embeddings: Tensor of shape (num_samples, embedding_dim)
            labels: Tensor of shape (num_samples,) with class indices
            
        Returns:
            Loss value for this step
        """
        embeddings = embeddings.to(self.device)
        labels = labels.to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Compute loss
        loss = contrastive_loss(
            self.representatives, 
            embeddings, 
            labels, 
            margin=self.config['margin'],
            lambda_push=self.config['lambda_push']
        )
        
        # Backpropagation
        loss.backward()
        
        # Update representatives
        self.optimizer.step()
        
        return loss.detach()
    
    def train(self, embeddings: torch.Tensor, labels: torch.Tensor, epochs: int) -> Dict[str, List[float]]:
        """
        Train representatives for multiple epochs.
        
        Args:
            embeddings: Training embeddings tensor
            labels: Training labels tensor
            epochs: Number of training epochs
            
        Returns:
            Dictionary with training history containing 'losses' list
        """
        history = {'losses': []}
        
        for epoch in range(epochs):
            loss = self.train_step(embeddings, labels)
            history['losses'].append(loss.item())
        
        return history
    
    def evaluate(self, embeddings: torch.Tensor, labels: torch.Tensor, 
                representatives: Optional[torch.Tensor] = None) -> float:
        """
        Evaluate performance using F1 score with nearest representative classification.
        
        Args:
            embeddings: Evaluation embeddings tensor
            labels: True labels tensor
            representatives: Representatives to use (default: self.representatives)
            
        Returns:
            Macro-averaged F1 score
        """
        if representatives is None:
            representatives = self.representatives
        
        embeddings = embeddings.to(self.device)
        labels = labels.to(self.device)
        representatives = representatives.to(self.device)
        
        with torch.no_grad():
            # Compute distances to all representatives
            distances = torch.cdist(embeddings, representatives)
            
            # Classify using nearest representative
            predictions = torch.argmin(distances, dim=1)
        
        # Convert to numpy for sklearn
        y_true = labels.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        
        # Handle edge case: only one class in labels
        unique_labels = np.unique(y_true)
        if len(unique_labels) == 1:
            # If all samples are same class, check if predictions match
            if len(np.unique(y_pred)) == 1 and y_pred[0] == y_true[0]:
                return 1.0
            else:
                return 0.0
        
        # Compute macro F1 score
        try:
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        except Exception:
            # Fallback for edge cases
            f1 = 0.0
        
        return float(f1)
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.config['lr']
    
    def get_margin(self) -> float:
        """Get current margin."""
        return self.config['margin']
    
    def get_lambda_push(self) -> float:
        """Get current lambda push."""
        return self.config['lambda_push'] 