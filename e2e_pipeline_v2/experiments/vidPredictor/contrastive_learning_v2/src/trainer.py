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
import pandas as pd

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
        
    def initialize_representatives(self, embeddings: torch.Tensor, labels: torch.Tensor, 
                                 init_method: str = 'class_means') -> None:
        """
        Initialize representatives using different strategies.
        
        Args:
            embeddings: Tensor of shape (num_samples, embedding_dim)
            labels: Tensor of shape (num_samples,) with class indices
            init_method: Initialization method. Options:
                - 'class_means': Initialize as class means (default)
                - 'random': Random initialization from standard normal
                - 'bounded_random': Random initialization within embedding bounds
                - 'perturbed_means': Class means + small random perturbation
        """
        embeddings = embeddings.to(self.device)
        labels = labels.to(self.device)
        
        num_classes = len(torch.unique(labels))
        embedding_dim = embeddings.shape[1]
        
        # Initialize representatives tensor
        self.representatives = torch.zeros(num_classes, embedding_dim, 
                                         device=self.device, requires_grad=True)
        
        if init_method == 'class_means':
            # Initialize as class means (original method)
            for class_idx in range(num_classes):
                class_mask = (labels == class_idx)
                if class_mask.any():
                    class_embeddings = embeddings[class_mask]
                    self.representatives.data[class_idx] = class_embeddings.mean(dim=0)
                    
        elif init_method == 'random':
            # Random initialization from standard normal distribution
            torch.nn.init.normal_(self.representatives.data, mean=0.0, std=1.0)
            
        elif init_method == 'bounded_random':
            # Random initialization within the bounds of the embedding space
            emb_min = embeddings.min(dim=0)[0]
            emb_max = embeddings.max(dim=0)[0]
            
            for class_idx in range(num_classes):
                # Uniform random within embedding bounds
                random_vals = torch.rand(embedding_dim, device=self.device)
                self.representatives.data[class_idx] = emb_min + random_vals * (emb_max - emb_min)
                
        elif init_method == 'perturbed_means':
            # Class means with small random perturbation
            for class_idx in range(num_classes):
                class_mask = (labels == class_idx)
                if class_mask.any():
                    class_embeddings = embeddings[class_mask]
                    class_mean = class_embeddings.mean(dim=0)
                    
                    # Add small random perturbation (10% of std)
                    perturbation = torch.randn_like(class_mean) * 0.1 * class_embeddings.std(dim=0)
                    self.representatives.data[class_idx] = class_mean + perturbation
                    
        else:
            raise ValueError(f"Unknown initialization method: {init_method}. "
                           f"Choose from: 'class_means', 'random', 'bounded_random', 'perturbed_means'")
        
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
    
    def set_hard_negatives(self, hard_negatives_dict: Dict[str, List[Dict]]) -> None:
        """
        Set hard negatives for enhanced training.
        
        Args:
            hard_negatives_dict: Dict mapping class_name -> list of embedding dicts
                Each embedding dict should have 'embedding' key with torch.Tensor
        """
        self.hard_negatives = {}
        
        # Convert class names to indices if we have a mapping
        if hasattr(self, 'class_to_idx'):
            for class_name, negatives in hard_negatives_dict.items():
                if class_name in self.class_to_idx:
                    class_idx = self.class_to_idx[class_name]
                    self.hard_negatives[class_idx] = [neg['embedding'].to(self.device) for neg in negatives]
        else:
            # For now, assume class names are the indices or we'll handle this differently
            print("⚠️  Warning: No class_to_idx mapping found. Hard negatives may not work correctly.")
            self.hard_negatives = {}
        
        total_hard_negatives = sum(len(negs) for negs in self.hard_negatives.values())
        print(f"🎯 Set {total_hard_negatives} hard negatives across {len(self.hard_negatives)} classes")
    
    def train_step_with_hard_negatives(self, embeddings: torch.Tensor, labels: torch.Tensor, 
                                     lambda_hard: float = 0.5) -> tuple:
        """
        Enhanced training step with hard negatives.
        
        Args:
            embeddings: Tensor of shape (num_samples, embedding_dim)
            labels: Tensor of shape (num_samples,) with class indices
            lambda_hard: Weight for hard negative loss
            
        Returns:
            Tuple of (total_loss, standard_loss, hard_loss)
        """
        embeddings = embeddings.to(self.device)
        labels = labels.to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Standard contrastive loss
        standard_loss = contrastive_loss(
            self.representatives, 
            embeddings, 
            labels, 
            margin=self.config['margin'],
            lambda_push=self.config['lambda_push']
        )
        
        # Hard negative loss
        hard_loss = torch.tensor(0.0, device=self.device)
        if hasattr(self, 'hard_negatives') and self.hard_negatives:
            for class_idx, neg_embeddings in self.hard_negatives.items():
                if len(neg_embeddings) > 0:
                    # Stack hard negatives for this class
                    neg_tensor = torch.stack(neg_embeddings).to(self.device)
                    
                    # Get representative for this class
                    rep = self.representatives[class_idx:class_idx+1]  # Keep batch dim
                    
                    # Compute similarity
                    sim_matrix = cosine_similarity_matrix(rep, neg_tensor)
                    
                    # Push hard negatives away (beyond margin)
                    margin = self.config['margin']
                    violations = torch.clamp(sim_matrix - margin, min=0.0)
                    hard_loss += torch.mean(violations)
        
        # Total loss
        total_loss = standard_loss + lambda_hard * hard_loss
        
        # Backpropagation
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.detach(), standard_loss.detach(), hard_loss.detach()
    
    def load_representatives_from_dataframe(self, representatives_df, class_names: List[str]) -> None:
        """
        Load representatives from a pandas DataFrame.
        
        Args:
            representatives_df: DataFrame with 'class' and 'finetuned_embedding' columns
            class_names: List of class names in the correct order
        """
        # Create class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Extract embeddings in the correct order
        embeddings_list = []
        for class_name in class_names:
            class_rows = representatives_df[representatives_df['class'] == class_name]
            if len(class_rows) == 0:
                raise ValueError(f"Class '{class_name}' not found in representatives DataFrame")
            embedding = class_rows.iloc[0]['finetuned_embedding']
            embeddings_list.append(embedding)
        
        # Convert to tensor
        embeddings_array = np.stack(embeddings_list)
        self.representatives = torch.tensor(embeddings_array, dtype=torch.float32, 
                                          device=self.device, requires_grad=True)
        
        # Initialize optimizer
        self.optimizer = optim.Adam([self.representatives], lr=self.config['lr'])
        
        print(f"✅ Loaded representatives for {len(class_names)} classes")
    
    def save_representatives_to_dataframe(self, class_names: List[str]):
        """
        Save current representatives to a pandas DataFrame.
        
        Args:
            class_names: List of class names in the correct order
            
        Returns:
            DataFrame with 'class' and 'finetuned_embedding' columns
        """
        if self.representatives is None:
            raise ValueError("No representatives to save")
        
        representatives_data = []
        for i, class_name in enumerate(class_names):
            representatives_data.append({
                'class': class_name,
                'finetuned_embedding': self.representatives[i].detach().cpu().numpy()
            })
        
        return pd.DataFrame(representatives_data) 