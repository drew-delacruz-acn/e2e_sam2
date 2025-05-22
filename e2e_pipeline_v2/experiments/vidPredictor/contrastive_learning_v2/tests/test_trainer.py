import pytest
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# Note: These imports will fail until we create the actual modules
# For now, we're writing the tests first (TDD approach)
# from src.trainer import ContrastiveTrainer


class TestTrainerInitialization:
    """Test trainer initialization and setup."""
    
    def test_initialization_with_class_means(self, sample_dataframe):
        """Test representatives are initialized as class means."""
        # Convert DataFrame to expected format
        embeddings_list = sample_dataframe['fine_tuned_embeddings'].tolist()
        embeddings = torch.tensor(np.stack(embeddings_list), dtype=torch.float32)
        
        # Create label mapping
        unique_classes = sample_dataframe['class'].unique()
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        labels = torch.tensor([class_to_idx[cls] for cls in sample_dataframe['class']])
        
        # Expected behavior: representatives should start as class means
        expected_reps = torch.zeros(len(unique_classes), embeddings.shape[1])
        for i, cls in enumerate(unique_classes):
            class_mask = labels == i
            class_embeddings = embeddings[class_mask]
            expected_reps[i] = class_embeddings.mean(dim=0)
        
        # When implemented:
        # config = {'lr': 0.001, 'margin': 0.15, 'lambda_push': 0.25}
        # trainer = ContrastiveTrainer(config)
        # trainer.initialize_representatives(embeddings, labels)
        # 
        # assert torch.allclose(trainer.representatives, expected_reps, atol=1e-6)
        # assert trainer.representatives.requires_grad
        
        # For now, verify our test setup
        assert expected_reps.shape == (len(unique_classes), embeddings.shape[1])
        assert len(unique_classes) == 3  # mirror, lamp, wall
    
    def test_optimizer_setup(self):
        """Test Adam optimizer is properly configured."""
        # When implemented:
        # config = {'lr': 0.001, 'margin': 0.15, 'lambda_push': 0.25}
        # trainer = ContrastiveTrainer(config)
        # 
        # # Should have Adam optimizer with correct learning rate
        # assert isinstance(trainer.optimizer, torch.optim.Adam)
        # assert trainer.optimizer.param_groups[0]['lr'] == 0.001
        
        # For now, verify config structure
        config = {'lr': 0.001, 'margin': 0.15, 'lambda_push': 0.25}
        assert 'lr' in config
        assert config['lr'] > 0
    
    def test_device_handling(self, device):
        """Test proper device selection and tensor placement."""
        # When implemented:
        # config = {'lr': 0.001, 'margin': 0.15, 'lambda_push': 0.25}
        # trainer = ContrastiveTrainer(config)
        # 
        # assert trainer.device == device
        # # Representatives should be on the correct device
        # assert trainer.representatives.device == device
        
        # For now, verify device is available
        assert device.type in ['cpu', 'cuda']


class TestTrainingLoop:
    """Test the training loop functionality."""
    
    def test_representatives_update_during_training(self, toy_separable_data):
        """Test that representatives actually change after optimizer step."""
        # Convert toy data to tensors
        embeddings_list = toy_separable_data['fine_tuned_embeddings'].tolist()
        embeddings = torch.tensor(np.stack(embeddings_list), dtype=torch.float32)
        
        unique_classes = toy_separable_data['class'].unique()
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        labels = torch.tensor([class_to_idx[cls] for cls in toy_separable_data['class']])
        
        # When implemented:
        # config = {'lr': 0.01, 'margin': 0.15, 'lambda_push': 0.25}  # Higher LR for visible change
        # trainer = ContrastiveTrainer(config)
        # trainer.initialize_representatives(embeddings, labels)
        # 
        # # Store initial state
        # initial_reps = trainer.representatives.clone()
        # 
        # # Run one training step
        # loss = trainer.train_step(embeddings, labels)
        # 
        # # Representatives should have changed
        # assert not torch.allclose(trainer.representatives, initial_reps)
        # assert torch.isfinite(loss)
        # assert loss >= 0
        
        # For now, verify test data setup
        assert embeddings.shape[0] == len(toy_separable_data)
        assert len(unique_classes) == 2  # A and B
    
    def test_loss_decreases_over_epochs(self, toy_separable_data):
        """Test that loss generally decreases during training."""
        # This is a key test - training should improve the loss
        
        # When implemented:
        # config = {'lr': 0.01, 'margin': 0.15, 'lambda_push': 0.25, 'epochs': 10}
        # trainer = ContrastiveTrainer(config)
        # 
        # embeddings_list = toy_separable_data['fine_tuned_embeddings'].tolist()
        # embeddings = torch.tensor(np.stack(embeddings_list), dtype=torch.float32)
        # 
        # unique_classes = toy_separable_data['class'].unique()
        # class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        # labels = torch.tensor([class_to_idx[cls] for cls in toy_separable_data['class']])
        # 
        # trainer.initialize_representatives(embeddings, labels)
        # 
        # # Train for several epochs
        # losses = []
        # for epoch in range(10):
        #     loss = trainer.train_step(embeddings, labels)
        #     losses.append(loss.item())
        # 
        # # Loss should generally decrease (allow some fluctuation)
        # assert losses[-1] < losses[0], "Final loss should be less than initial loss"
        
        # Verify test data is suitable for this test
        embeddings_list = toy_separable_data['fine_tuned_embeddings'].tolist()
        embeddings = torch.tensor(np.stack(embeddings_list), dtype=torch.float32)
        
        # Data should be separable (class A around (2,2), class B around (-2,-2))
        class_a_mask = toy_separable_data['class'] == 'A'
        class_b_mask = toy_separable_data['class'] == 'B'
        
        class_a_embeddings = embeddings[class_a_mask.values]
        class_b_embeddings = embeddings[class_b_mask.values]
        
        # Verify classes are well-separated
        a_mean = class_a_embeddings.mean(dim=0)
        b_mean = class_b_embeddings.mean(dim=0)
        separation = torch.norm(a_mean - b_mean)
        
        assert separation > 1.0, "Classes should be well-separated for this test"
    
    def test_gradient_accumulation(self):
        """Test gradient computation and accumulation."""
        torch.manual_seed(42)
        
        # Simple test case
        representatives = torch.randn(2, 3, requires_grad=True)
        embeddings = torch.randn(4, 3)
        labels = torch.tensor([0, 0, 1, 1])
        
        # When implemented:
        # config = {'lr': 0.001, 'margin': 0.15, 'lambda_push': 0.25}
        # trainer = ContrastiveTrainer(config)
        # trainer.representatives = representatives
        # trainer.optimizer = torch.optim.Adam([trainer.representatives], lr=config['lr'])
        # 
        # # Run training step
        # loss = trainer.train_step(embeddings, labels)
        # 
        # # Check gradients exist and are reasonable
        # assert trainer.representatives.grad is not None
        # assert not torch.allclose(trainer.representatives.grad, torch.zeros_like(trainer.representatives.grad))
        # assert torch.all(torch.isfinite(trainer.representatives.grad))
        
        # For now, verify setup
        assert representatives.requires_grad
        assert embeddings.shape[0] == len(labels)


class TestEvaluation:
    """Test evaluation and metrics computation."""
    
    def test_f1_calculation_baseline_vs_learned(self, toy_separable_data):
        """Test F1 score calculation for baseline vs learned representatives."""
        # Convert data
        embeddings_list = toy_separable_data['fine_tuned_embeddings'].tolist()
        embeddings = torch.tensor(np.stack(embeddings_list), dtype=torch.float32)
        
        unique_classes = toy_separable_data['class'].unique()
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        labels = torch.tensor([class_to_idx[cls] for cls in toy_separable_data['class']])
        
        # Compute baseline (class means)
        baseline_reps = torch.zeros(len(unique_classes), embeddings.shape[1])
        for i in range(len(unique_classes)):
            class_mask = labels == i
            baseline_reps[i] = embeddings[class_mask].mean(dim=0)
        
        # When implemented:
        # config = {'lr': 0.001, 'margin': 0.15, 'lambda_push': 0.25}
        # trainer = ContrastiveTrainer(config)
        # 
        # # Evaluate baseline
        # baseline_f1 = trainer.evaluate(embeddings, labels, baseline_reps)
        # 
        # # For perfectly separable data, F1 should be high
        # assert baseline_f1 > 0.8, "Baseline should perform well on separable data"
        # 
        # # Test with learned representatives (should be at least as good)
        # learned_reps = baseline_reps + 0.01 * torch.randn_like(baseline_reps)  # Slight perturbation
        # learned_f1 = trainer.evaluate(embeddings, labels, learned_reps)
        # 
        # assert 0.0 <= learned_f1 <= 1.0, "F1 score should be in valid range"
        
        # For now, test manual F1 calculation
        # Classify using nearest representative
        distances = torch.cdist(embeddings, baseline_reps)
        predictions = torch.argmin(distances, dim=1)
        
        # Convert to numpy for sklearn
        y_true = labels.numpy()
        y_pred = predictions.numpy()
        
        f1 = f1_score(y_true, y_pred, average='macro')
        assert 0.0 <= f1 <= 1.0, "F1 score should be in valid range"
        assert f1 > 0.5, "Should perform better than random on separable data"
    
    def test_nearest_representative_classification(self):
        """Test classification using nearest representative."""
        # Simple 2D case where we know the answer
        representatives = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])  # Class 0 at (1,0), Class 1 at (-1,0)
        test_embeddings = torch.tensor([[0.9, 0.1], [-0.8, 0.2]])  # Should classify as [0, 1]
        
        # Manual nearest neighbor classification
        distances = torch.cdist(test_embeddings, representatives)
        predictions = torch.argmin(distances, dim=1)
        
        expected_predictions = torch.tensor([0, 1])
        assert torch.equal(predictions, expected_predictions)
    
    def test_evaluation_with_single_class(self):
        """Test evaluation edge case with only one class."""
        # Edge case: what happens if validation set has only one class?
        embeddings = torch.randn(3, 5)
        labels = torch.zeros(3, dtype=torch.long)  # All same class
        representatives = torch.randn(2, 5)  # But we have 2 representatives
        
        # When implemented:
        # config = {'lr': 0.001, 'margin': 0.15, 'lambda_push': 0.25}
        # trainer = ContrastiveTrainer(config)
        # 
        # # Should handle gracefully (maybe return NaN or 0)
        # f1 = trainer.evaluate(embeddings, labels, representatives)
        # assert torch.isfinite(torch.tensor(f1)) or torch.isnan(torch.tensor(f1))
        
        # For now, verify this is indeed an edge case
        unique_labels = torch.unique(labels)
        assert len(unique_labels) == 1, "Should have only one class"


class TestTrainerConfiguration:
    """Test trainer configuration and parameter handling."""
    
    def test_config_validation(self):
        """Test configuration parameter validation."""
        # Valid config
        valid_config = {
            'lr': 0.001,
            'margin': 0.15,
            'lambda_push': 0.25,
            'epochs': 50
        }
        
        # When implemented:
        # trainer = ContrastiveTrainer(valid_config)
        # assert trainer.config['lr'] == 0.001
        # assert trainer.config['margin'] == 0.15
        
        # Invalid configs should raise errors
        invalid_configs = [
            {'lr': -0.001},  # Negative learning rate
            {'margin': -0.1},  # Negative margin
            {'lambda_push': -0.5},  # Negative lambda
            {}  # Missing required parameters
        ]
        
        # When implemented:
        # for invalid_config in invalid_configs:
        #     with pytest.raises(ValueError):
        #         ContrastiveTrainer(invalid_config)
        
        # For now, verify our test configs
        assert valid_config['lr'] > 0
        assert valid_config['margin'] >= 0
        assert valid_config['lambda_push'] >= 0
    
    def test_hyperparameter_access(self):
        """Test access to hyperparameters during training."""
        config = {
            'lr': 0.001,
            'margin': 0.15,
            'lambda_push': 0.25,
            'epochs': 50
        }
        
        # When implemented:
        # trainer = ContrastiveTrainer(config)
        # 
        # # Should be able to access hyperparameters
        # assert trainer.get_lr() == 0.001
        # assert trainer.get_margin() == 0.15
        # assert trainer.get_lambda_push() == 0.25
        
        # For now, verify config structure
        required_keys = ['lr', 'margin', 'lambda_push']
        for key in required_keys:
            assert key in config


class TestTrainerIntegration:
    """Integration tests for the complete trainer workflow."""
    
    def test_full_training_pipeline(self, toy_separable_data):
        """Test the complete training pipeline from start to finish."""
        # This test will verify the entire workflow once implemented
        
        # When implemented:
        # config = {
        #     'lr': 0.01,
        #     'margin': 0.15,
        #     'lambda_push': 0.25,
        #     'epochs': 20
        # }
        # 
        # trainer = ContrastiveTrainer(config)
        # 
        # # Prepare data
        # embeddings_list = toy_separable_data['fine_tuned_embeddings'].tolist()
        # embeddings = torch.tensor(np.stack(embeddings_list), dtype=torch.float32)
        # 
        # unique_classes = toy_separable_data['class'].unique()
        # class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        # labels = torch.tensor([class_to_idx[cls] for cls in toy_separable_data['class']])
        # 
        # # Initialize
        # trainer.initialize_representatives(embeddings, labels)
        # baseline_f1 = trainer.evaluate(embeddings, labels, trainer.representatives.detach())
        # 
        # # Train
        # history = trainer.train(embeddings, labels, epochs=config['epochs'])
        # 
        # # Evaluate
        # final_f1 = trainer.evaluate(embeddings, labels, trainer.representatives.detach())
        # 
        # # Assertions
        # assert len(history['losses']) == config['epochs']
        # assert final_f1 >= baseline_f1, "Training should improve or maintain F1"
        # assert all(loss >= 0 for loss in history['losses']), "All losses should be non-negative"
        
        # For now, verify test data is suitable
        assert len(toy_separable_data) == 10  # 5 samples per class
        assert len(toy_separable_data['class'].unique()) == 2  # 2 classes
    
    def test_training_convergence_on_toy_data(self, toy_separable_data):
        """Test that training converges on perfectly separable data."""
        # On perfectly separable data, the algorithm should converge to a good solution
        
        # When implemented:
        # config = {
        #     'lr': 0.01,
        #     'margin': 0.15,
        #     'lambda_push': 0.5,
        #     'epochs': 50
        # }
        # 
        # trainer = ContrastiveTrainer(config)
        # 
        # # Prepare data
        # embeddings_list = toy_separable_data['fine_tuned_embeddings'].tolist()
        # embeddings = torch.tensor(np.stack(embeddings_list), dtype=torch.float32)
        # 
        # unique_classes = toy_separable_data['class'].unique()
        # class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        # labels = torch.tensor([class_to_idx[cls] for cls in toy_separable_data['class']])
        # 
        # # Train
        # trainer.initialize_representatives(embeddings, labels)
        # history = trainer.train(embeddings, labels, epochs=config['epochs'])
        # 
        # # Should achieve perfect or near-perfect classification
        # final_f1 = trainer.evaluate(embeddings, labels, trainer.representatives.detach())
        # assert final_f1 > 0.95, "Should achieve high F1 on perfectly separable data"
        # 
        # # Loss should have decreased significantly
        # initial_loss = history['losses'][0]
        # final_loss = history['losses'][-1]
        # assert final_loss < initial_loss, "Loss should decrease during training"
        
        # Verify the data is indeed perfectly separable
        embeddings_list = toy_separable_data['fine_tuned_embeddings'].tolist()
        embeddings = torch.tensor(np.stack(embeddings_list), dtype=torch.float32)
        
        class_a_mask = toy_separable_data['class'] == 'A'
        class_b_mask = toy_separable_data['class'] == 'B'
        
        class_a_embeddings = embeddings[class_a_mask.values]
        class_b_embeddings = embeddings[class_b_mask.values]
        
        # Check separation
        a_center = class_a_embeddings.mean(dim=0)
        b_center = class_b_embeddings.mean(dim=0)
        
        # All class A samples should be closer to A center than B center
        for emb in class_a_embeddings:
            dist_to_a = torch.norm(emb - a_center)
            dist_to_b = torch.norm(emb - b_center)
            assert dist_to_a < dist_to_b, "Class A samples should be closer to A center" 