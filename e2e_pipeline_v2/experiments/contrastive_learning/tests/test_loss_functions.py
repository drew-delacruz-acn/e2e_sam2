import pytest
import torch
import numpy as np

# Note: These imports will fail until we create the actual modules
# For now, we're writing the tests first (TDD approach)
# from src.loss_functions import contrastive_loss, cosine_similarity_matrix


class TestCosineSimilarity:
    """Test cosine similarity computation."""
    
    def test_cosine_similarity_known_vectors(self):
        """Test cosine similarity with vectors of known similarity."""
        # Test with simple 2D vectors
        v1 = torch.tensor([1.0, 0.0])  # Unit vector along x-axis
        v2 = torch.tensor([0.0, 1.0])  # Unit vector along y-axis
        v3 = torch.tensor([1.0, 0.0])  # Same as v1
        
        # Expected similarities:
        # cos(v1, v2) = 0 (orthogonal)
        # cos(v1, v3) = 1 (identical)
        # cos(v2, v3) = 0 (orthogonal)
        
        # Manual calculation for verification
        manual_sim_12 = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
        manual_sim_13 = torch.dot(v1, v3) / (torch.norm(v1) * torch.norm(v3))
        
        assert torch.isclose(manual_sim_12, torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(manual_sim_13, torch.tensor(1.0), atol=1e-6)
    
    def test_cosine_similarity_matrix_shape(self):
        """Test cosine similarity matrix has correct shape."""
        # Test data: 3 representatives, 5 embeddings
        reps = torch.randn(3, 10)  # 3 classes, 10-dim embeddings
        embeddings = torch.randn(5, 10)  # 5 samples, 10-dim embeddings
        
        # Expected output shape: (3, 5) - each rep vs each embedding
        # When implemented:
        # sim_matrix = cosine_similarity_matrix(reps, embeddings)
        # assert sim_matrix.shape == (3, 5)
        
        # For now, verify our test setup
        assert reps.shape == (3, 10)
        assert embeddings.shape == (5, 10)
    
    def test_cosine_similarity_range(self):
        """Test cosine similarity values are in [-1, 1] range."""
        # Random vectors should produce similarities in valid range
        reps = torch.randn(2, 5)
        embeddings = torch.randn(3, 5)
        
        # Manual calculation to verify expected behavior
        for i in range(reps.shape[0]):
            for j in range(embeddings.shape[0]):
                rep = reps[i]
                emb = embeddings[j]
                sim = torch.dot(rep, emb) / (torch.norm(rep) * torch.norm(emb))
                assert -1.0 <= sim.item() <= 1.0, f"Similarity {sim} out of range"


class TestContrastiveLoss:
    """Test the contrastive loss function."""
    
    def test_loss_is_non_negative(self):
        """Test that loss is always >= 0."""
        # Create simple test case
        torch.manual_seed(42)
        
        # 2 classes, 2D embeddings for simplicity
        representatives = torch.randn(2, 2, requires_grad=True)
        embeddings = torch.randn(4, 2)  # 2 samples per class
        labels = torch.tensor([0, 0, 1, 1])  # Class labels
        
        margin = 0.1
        lambda_push = 0.5
        
        # When implemented:
        # loss = contrastive_loss(representatives, embeddings, labels, margin, lambda_push)
        # assert loss >= 0, "Loss should be non-negative"
        
        # For now, verify test setup
        assert representatives.shape == (2, 2)
        assert embeddings.shape == (4, 2)
        assert len(labels) == 4
    
    def test_perfect_separation_gives_low_push_loss(self):
        """Test that perfect separation results in minimal push loss."""
        # Create perfectly separated data
        # Class 0: embeddings at (1, 0)
        # Class 1: embeddings at (-1, 0)
        # Representatives should be at class centers
        
        representatives = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], requires_grad=True)
        embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
        labels = torch.tensor([0, 0, 1, 1])
        
        margin = 0.5  # Cosine similarity between (1,0) and (-1,0) is -1, well below margin
        lambda_push = 1.0
        
        # Expected behavior:
        # - Pull term should be 0 (reps are exactly at class means)
        # - Push term should be 0 (cosine similarity is -1, well below margin)
        # - Total loss should be very small
        
        # Manual verification of cosine similarity
        cos_sim = torch.dot(representatives[0], representatives[1])
        cos_sim = cos_sim / (torch.norm(representatives[0]) * torch.norm(representatives[1]))
        assert cos_sim.item() == -1.0, "Opposite unit vectors should have cosine similarity -1"
    
    def test_loss_decreases_when_rep_moves_toward_positives(self):
        """CRITICAL: Test loss decreases when representative moves toward its positives."""
        torch.manual_seed(42)
        
        # Simple 2D case: Class 0 has samples at (1, 1) and (1.1, 1.1)
        embeddings = torch.tensor([[1.0, 1.0], [1.1, 1.1], [-1.0, -1.0]])
        labels = torch.tensor([0, 0, 1])  # First two are class 0, last is class 1
        
        # Start with representative far from its positives
        rep_far = torch.tensor([[0.0, 0.0], [-1.0, -1.0]], requires_grad=True)
        
        # Move representative closer to its positives
        rep_close = torch.tensor([[1.05, 1.05], [-1.0, -1.0]], requires_grad=True)
        
        margin = 0.1
        lambda_push = 0.5
        
        # When implemented:
        # loss_far = contrastive_loss(rep_far, embeddings, labels, margin, lambda_push)
        # loss_close = contrastive_loss(rep_close, embeddings, labels, margin, lambda_push)
        # assert loss_close < loss_far, "Loss should decrease when rep moves toward positives"
        
        # For now, verify the setup makes sense
        # rep_close should be closer to class 0 samples than rep_far
        class_0_mean = embeddings[labels == 0].mean(dim=0)
        dist_far = torch.norm(rep_far[0] - class_0_mean)
        dist_close = torch.norm(rep_close[0] - class_0_mean)
        assert dist_close < dist_far, "rep_close should be closer to class 0 samples"
    
    def test_loss_increases_when_rep_moves_toward_negatives(self):
        """Test loss increases when representative moves toward negatives."""
        torch.manual_seed(42)
        
        # Class 0 at (1, 1), Class 1 at (-1, -1)
        embeddings = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
        labels = torch.tensor([0, 1])
        
        # Representative starts at its class center
        rep_good = torch.tensor([[1.0, 1.0], [-1.0, -1.0]], requires_grad=True)
        
        # Representative moves toward the other class
        rep_bad = torch.tensor([[0.0, 0.0], [-1.0, -1.0]], requires_grad=True)
        
        margin = 0.5
        lambda_push = 1.0
        
        # When implemented:
        # loss_good = contrastive_loss(rep_good, embeddings, labels, margin, lambda_push)
        # loss_bad = contrastive_loss(rep_bad, embeddings, labels, margin, lambda_push)
        # assert loss_bad > loss_good, "Loss should increase when rep moves toward negatives"
        
        # Verify setup: rep_bad should be closer to the negative class
        negative_sample = embeddings[labels == 1][0]  # Class 1 sample
        dist_good = torch.norm(rep_good[0] - negative_sample)
        dist_bad = torch.norm(rep_bad[0] - negative_sample)
        assert dist_bad < dist_good, "rep_bad should be closer to negative sample"
    
    def test_margin_enforcement(self):
        """Test that margin parameter works correctly."""
        # Create case where cosine similarity is exactly at margin
        rep = torch.tensor([[1.0, 0.0]], requires_grad=True)
        # Use exact value for 60 degrees: cos(60°) = 0.5, sin(60°) = √3/2
        sqrt_3_over_2 = torch.sqrt(torch.tensor(3.0)) / 2.0
        negative_emb = torch.tensor([[0.5, sqrt_3_over_2]])  # Exact 60 degrees from rep
        positive_emb = torch.tensor([[1.0, 0.0]])    # Same as rep, cos=1.0
        
        embeddings = torch.cat([positive_emb, negative_emb])
        labels = torch.tensor([0, 1])  # First is positive, second is negative
        
        # Test with margin = 0.5 (exactly at the cosine similarity)
        margin = 0.5
        lambda_push = 1.0
        
        # Expected: push loss should be 0 since cos(rep, negative) = margin exactly
        # When implemented:
        # loss = contrastive_loss(rep, embeddings, labels, margin, lambda_push)
        
        # Verify our test setup
        manual_cos = torch.dot(rep[0], negative_emb[0])
        manual_cos = manual_cos / (torch.norm(rep[0]) * torch.norm(negative_emb[0]))
        assert torch.isclose(manual_cos, torch.tensor(0.5), atol=1e-6)
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the loss."""
        torch.manual_seed(42)
        
        representatives = torch.randn(2, 3, requires_grad=True)
        embeddings = torch.randn(4, 3)
        labels = torch.tensor([0, 0, 1, 1])
        
        margin = 0.1
        lambda_push = 0.5
        
        # When implemented:
        # loss = contrastive_loss(representatives, embeddings, labels, margin, lambda_push)
        # loss.backward()
        # 
        # # Check that gradients exist and are non-zero
        # assert representatives.grad is not None
        # assert not torch.allclose(representatives.grad, torch.zeros_like(representatives.grad))
        
        # For now, verify setup
        assert representatives.requires_grad
        assert representatives.shape == (2, 3)


class TestLossComponents:
    """Test individual components of the loss function."""
    
    def test_pull_term_calculation(self):
        """Test the positive pulling term calculation."""
        # Simple case: representative exactly at class mean should give max pull
        class_samples = torch.tensor([[1.0, 0.0], [1.0, 0.0]])  # Identical samples
        rep = torch.tensor([1.0, 0.0])  # Same as samples
        
        # Expected pull term: -mean(cos(rep, samples)) = -1.0
        expected_pull = -1.0
        
        # Manual calculation
        similarities = []
        for sample in class_samples:
            cos_sim = torch.dot(rep, sample) / (torch.norm(rep) * torch.norm(sample))
            similarities.append(cos_sim)
        
        actual_pull = -torch.mean(torch.stack(similarities))
        assert torch.isclose(actual_pull, torch.tensor(expected_pull), atol=1e-6)
    
    def test_push_term_with_margin(self):
        """Test the negative pushing term with margin."""
        rep = torch.tensor([1.0, 0.0])
        negative_samples = torch.tensor([[0.8, 0.6], [-1.0, 0.0]])  # cos=0.8, cos=-1.0
        margin = 0.5
        
        # Expected push terms:
        # Sample 1: max(0, 0.8 - 0.5) = 0.3
        # Sample 2: max(0, -1.0 - 0.5) = 0.0
        # Mean: (0.3 + 0.0) / 2 = 0.15
        
        push_terms = []
        for sample in negative_samples:
            cos_sim = torch.dot(rep, sample) / (torch.norm(rep) * torch.norm(sample))
            push_term = torch.max(torch.tensor(0.0), cos_sim - margin)
            push_terms.append(push_term)
        
        expected_push = torch.mean(torch.stack(push_terms))
        
        # Verify individual calculations
        cos_1 = torch.dot(rep, negative_samples[0]) / (torch.norm(rep) * torch.norm(negative_samples[0]))
        cos_2 = torch.dot(rep, negative_samples[1]) / (torch.norm(rep) * torch.norm(negative_samples[1]))
        
        assert torch.isclose(cos_1, torch.tensor(0.8), atol=1e-6)
        assert torch.isclose(cos_2, torch.tensor(-1.0), atol=1e-6)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_sample_per_class(self):
        """Test behavior with only one sample per class."""
        representatives = torch.randn(2, 3, requires_grad=True)
        embeddings = torch.randn(2, 3)  # One sample per class
        labels = torch.tensor([0, 1])
        
        margin = 0.1
        lambda_push = 0.5
        
        # Should handle this gracefully
        # When implemented:
        # loss = contrastive_loss(representatives, embeddings, labels, margin, lambda_push)
        # assert torch.isfinite(loss), "Loss should be finite even with single samples"
        
        # Verify test setup
        unique_labels = torch.unique(labels)
        assert len(unique_labels) == 2
        for label in unique_labels:
            assert torch.sum(labels == label) == 1, "Should have exactly one sample per class"
    
    def test_zero_vectors(self):
        """Test behavior with zero vectors."""
        representatives = torch.zeros(1, 3, requires_grad=True)
        embeddings = torch.zeros(2, 3)
        labels = torch.tensor([0, 0])
        
        margin = 0.1
        lambda_push = 0.5
        
        # Zero vectors have undefined cosine similarity (0/0)
        # Implementation should handle this gracefully
        # When implemented, should either:
        # 1. Raise informative error
        # 2. Handle with special case (e.g., treat as similarity = 0)
        
        # Verify test setup creates the problematic case
        assert torch.allclose(representatives, torch.zeros_like(representatives))
        assert torch.allclose(embeddings, torch.zeros_like(embeddings))
    
    def test_identical_representatives(self):
        """Test behavior when representatives are identical."""
        # This could happen during initialization or if learning rate is too high
        identical_rep = torch.tensor([1.0, 0.0])
        representatives = torch.stack([identical_rep, identical_rep], dim=0)
        representatives.requires_grad_(True)
        
        embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([0, 1])
        
        margin = 0.1
        lambda_push = 0.5
        
        # Should handle identical representatives gracefully
        # When implemented:
        # loss = contrastive_loss(representatives, embeddings, labels, margin, lambda_push)
        # assert torch.isfinite(loss), "Should handle identical representatives"
        
        # Verify setup
        assert torch.allclose(representatives[0], representatives[1])


class TestLossIntegration:
    """Integration tests for the complete loss computation."""
    
    def test_loss_with_toy_separable_data(self, toy_separable_data):
        """Test loss computation with perfectly separable toy data."""
        # Convert DataFrame to tensors
        embeddings_list = toy_separable_data['finetuned_embedding'].tolist()
        embeddings = torch.tensor(np.stack(embeddings_list), dtype=torch.float32)
        
        # Create label mapping
        unique_classes = toy_separable_data['class'].unique()
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        labels = torch.tensor([class_to_idx[cls] for cls in toy_separable_data['class']])
        
        # Initialize representatives as class means
        num_classes = len(unique_classes)
        representatives = torch.zeros(num_classes, embeddings.shape[1], requires_grad=True)
        
        for i, cls in enumerate(unique_classes):
            class_mask = labels == i
            class_embeddings = embeddings[class_mask]
            representatives.data[i] = class_embeddings.mean(dim=0)
        
        margin = 0.1
        lambda_push = 0.5
        
        # When implemented:
        # loss = contrastive_loss(representatives, embeddings, labels, margin, lambda_push)
        # assert torch.isfinite(loss), "Loss should be finite"
        # assert loss >= 0, "Loss should be non-negative"
        
        # Verify our setup
        assert embeddings.shape[0] == len(toy_separable_data)
        assert len(labels) == len(toy_separable_data)
        assert representatives.shape[0] == num_classes 