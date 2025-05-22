import pytest
import torch
import numpy as np
import pandas as pd
import tempfile
import json
from pathlib import Path

# Note: These imports will fail until we create the actual modules
# For now, we're writing the tests first (TDD approach)
# from src.data_loader import load_and_split_data
# from src.trainer import ContrastiveTrainer
# from src.visualizer import create_plots


class TestEndToEndWorkflow:
    """Test the complete end-to-end workflow."""
    
    def test_full_pipeline_runs_without_errors(self, temp_pkl_file, temp_output_dir):
        """Test complete workflow: load → split → train → evaluate → save."""
        # This is the master integration test
        
        # When implemented, this should run the entire pipeline:
        # 1. Load PKL file
        # 2. Split into train/val
        # 3. Initialize trainer
        # 4. Train representatives
        # 5. Evaluate performance
        # 6. Generate plots
        # 7. Save results
        
        # Expected workflow:
        # train_df, val_df, class_names = load_and_split_data(temp_pkl_file, val_frac=0.3)
        # 
        # config = {
        #     'lr': 0.01,
        #     'margin': 0.15,
        #     'lambda_push': 0.25,
        #     'epochs': 10  # Short for testing
        # }
        # 
        # trainer = ContrastiveTrainer(config)
        # 
        # # Convert to tensors
        # train_embeddings = torch.tensor(np.stack(train_df['finetuned_embedding'].tolist()))
        # val_embeddings = torch.tensor(np.stack(val_df['finetuned_embedding'].tolist()))
        # 
        # class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
        # train_labels = torch.tensor([class_to_idx[cls] for cls in train_df['class']])
        # val_labels = torch.tensor([class_to_idx[cls] for cls in val_df['class']])
        # 
        # # Train
        # trainer.initialize_representatives(train_embeddings, train_labels)
        # baseline_f1 = trainer.evaluate(val_embeddings, val_labels, trainer.representatives.detach())
        # 
        # history = trainer.train(train_embeddings, train_labels, epochs=config['epochs'])
        # final_f1 = trainer.evaluate(val_embeddings, val_labels, trainer.representatives.detach())
        # 
        # # Save results
        # results = {
        #     'baseline_f1': baseline_f1,
        #     'final_f1': final_f1,
        #     'config': config,
        #     'history': history
        # }
        # 
        # results_file = temp_output_dir / 'results.json'
        # with open(results_file, 'w') as f:
        #     json.dump(results, f)
        # 
        # # Generate plots
        # create_plots(history, trainer.representatives, train_embeddings, train_labels, temp_output_dir)
        # 
        # # Verify outputs
        # assert results_file.exists()
        # assert final_f1 >= 0.0
        # assert len(history['losses']) == config['epochs']
        
        # For now, verify test setup
        assert Path(temp_pkl_file).exists()
        assert temp_output_dir.exists()
    
    def test_reproducibility_with_seed(self, toy_separable_data):
        """Test same seed gives identical results."""
        # Critical for scientific reproducibility
        
        # When implemented:
        # def run_experiment(seed):
        #     torch.manual_seed(seed)
        #     np.random.seed(seed)
        #     
        #     config = {
        #         'lr': 0.01,
        #         'margin': 0.15,
        #         'lambda_push': 0.25,
        #         'epochs': 5
        #     }
        #     
        #     trainer = ContrastiveTrainer(config)
        #     
        #     embeddings_list = toy_separable_data['finetuned_embedding'].tolist()
        #     embeddings = torch.tensor(np.stack(embeddings_list), dtype=torch.float32)
        #     
        #     unique_classes = toy_separable_data['class'].unique()
        #     class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        #     labels = torch.tensor([class_to_idx[cls] for cls in toy_separable_data['class']])
        #     
        #     trainer.initialize_representatives(embeddings, labels)
        #     history = trainer.train(embeddings, labels, epochs=config['epochs'])
        #     
        #     return history['losses']
        # 
        # # Run twice with same seed
        # losses_1 = run_experiment(42)
        # losses_2 = run_experiment(42)
        # 
        # # Should be identical
        # assert len(losses_1) == len(losses_2)
        # for l1, l2 in zip(losses_1, losses_2):
        #     assert abs(l1 - l2) < 1e-6, "Results should be identical with same seed"
        
        # For now, verify test data consistency
        assert len(toy_separable_data) == 10
        assert len(toy_separable_data['class'].unique()) == 2
    
    def test_output_files_created(self, temp_output_dir):
        """Test all expected output files are generated."""
        # When implemented, should create:
        # - results.json (metrics and config)
        # - representatives.pkl (learned embeddings)
        # - loss_curve.png
        # - tsne_plot.png
        
        expected_files = [
            'results.json',
            'representatives.pkl',
            'loss_curve.png',
            'tsne_plot.png'
        ]
        
        # When implemented:
        # # Run full pipeline
        # run_full_experiment(output_dir=temp_output_dir)
        # 
        # # Check all files exist
        # for filename in expected_files:
        #     filepath = temp_output_dir / filename
        #     assert filepath.exists(), f"Missing output file: {filename}"
        #     assert filepath.stat().st_size > 0, f"Empty output file: {filename}"
        
        # For now, verify output directory is writable
        test_file = temp_output_dir / 'test.txt'
        test_file.write_text('test')
        assert test_file.exists()
    
    def test_learned_reps_better_than_baseline(self, toy_separable_data):
        """Test on data where contrastive learning should help."""
        # This is the key success metric for the experiment
        
        # When implemented:
        # config = {
        #     'lr': 0.01,
        #     'margin': 0.15,
        #     'lambda_push': 0.5,  # Higher push for better separation
        #     'epochs': 20
        # }
        # 
        # trainer = ContrastiveTrainer(config)
        # 
        # embeddings_list = toy_separable_data['finetuned_embedding'].tolist()
        # embeddings = torch.tensor(np.stack(embeddings_list), dtype=torch.float32)
        # 
        # unique_classes = toy_separable_data['class'].unique()
        # class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        # labels = torch.tensor([class_to_idx[cls] for cls in toy_separable_data['class']])
        # 
        # # Split data
        # train_size = int(0.7 * len(embeddings))
        # train_embeddings = embeddings[:train_size]
        # train_labels = labels[:train_size]
        # val_embeddings = embeddings[train_size:]
        # val_labels = labels[train_size:]
        # 
        # # Initialize and get baseline
        # trainer.initialize_representatives(train_embeddings, train_labels)
        # baseline_f1 = trainer.evaluate(val_embeddings, val_labels, trainer.representatives.detach())
        # 
        # # Train
        # trainer.train(train_embeddings, train_labels, epochs=config['epochs'])
        # learned_f1 = trainer.evaluate(val_embeddings, val_labels, trainer.representatives.detach())
        # 
        # # Key assertion: learning should improve performance
        # assert learned_f1 >= baseline_f1, f"Learned F1 ({learned_f1:.3f}) should be >= baseline F1 ({baseline_f1:.3f})"
        # 
        # # On perfectly separable data, should achieve high performance
        # assert learned_f1 > 0.8, f"Should achieve high F1 on separable data, got {learned_f1:.3f}"
        
        # Verify test data is suitable for this test
        embeddings_list = toy_separable_data['finetuned_embedding'].tolist()
        embeddings = torch.tensor(np.stack(embeddings_list), dtype=torch.float32)
        
        # Check that classes are well-separated
        class_a_mask = toy_separable_data['class'] == 'A'
        class_b_mask = toy_separable_data['class'] == 'B'
        
        class_a_embeddings = embeddings[class_a_mask.values]
        class_b_embeddings = embeddings[class_b_mask.values]
        
        a_center = class_a_embeddings.mean(dim=0)
        b_center = class_b_embeddings.mean(dim=0)
        separation = torch.norm(a_center - b_center)
        
        assert separation > 2.0, "Classes should be well-separated for this test"


class TestErrorHandling:
    """Test error handling and edge cases in the full pipeline."""
    
    def test_handles_missing_pkl_file(self):
        """Test graceful handling of missing input file."""
        nonexistent_file = "/path/that/does/not/exist.pkl"
        
        # When implemented:
        # with pytest.raises(FileNotFoundError, match="PKL file not found"):
        #     load_and_split_data(nonexistent_file)
        
        # For now, verify the file doesn't exist
        assert not Path(nonexistent_file).exists()
    
    def test_handles_corrupted_pkl_file(self, temp_output_dir):
        """Test handling of corrupted or invalid PKL files."""
        # Create a corrupted PKL file
        corrupted_file = temp_output_dir / "corrupted.pkl"
        corrupted_file.write_text("This is not a valid PKL file")
        
        # When implemented:
        # with pytest.raises(Exception, match="Failed to load PKL file"):
        #     load_and_split_data(str(corrupted_file))
        
        # For now, verify the corrupted file exists
        assert corrupted_file.exists()
    
    def test_handles_insufficient_data(self, single_sample_per_class):
        """Test handling when there's insufficient data for splitting."""
        # When implemented:
        # # Should either raise error or handle gracefully
        # try:
        #     train_df, val_df, class_names = load_and_split_data(single_sample_per_class, val_frac=0.5)
        #     # If it succeeds, should warn about insufficient validation data
        #     assert len(val_df) == 0 or len(train_df) >= len(val_df)
        # except ValueError as e:
        #     assert "insufficient data" in str(e).lower()
        
        # Verify this is indeed an edge case
        class_counts = single_sample_per_class['class'].value_counts()
        assert class_counts.max() == 1, "Should have only one sample per class"
    
    def test_handles_empty_dataframe(self):
        """Test handling of empty input data."""
        empty_df = pd.DataFrame(columns=['class', 'finetuned_embedding'])
        
        # When implemented:
        # with pytest.raises(ValueError, match="Empty dataset"):
        #     load_and_split_data(empty_df)
        
        # Verify test setup
        assert len(empty_df) == 0
        assert 'class' in empty_df.columns
        assert 'finetuned_embedding' in empty_df.columns


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling behavior."""
    
    def test_training_time_reasonable(self, sample_dataframe):
        """Test that training completes in reasonable time."""
        import time
        
        # When implemented:
        # start_time = time.time()
        # 
        # config = {
        #     'lr': 0.01,
        #     'margin': 0.15,
        #     'lambda_push': 0.25,
        #     'epochs': 10
        # }
        # 
        # trainer = ContrastiveTrainer(config)
        # 
        # embeddings_list = sample_dataframe['finetuned_embedding'].tolist()
        # embeddings = torch.tensor(np.stack(embeddings_list), dtype=torch.float32)
        # 
        # unique_classes = sample_dataframe['class'].unique()
        # class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        # labels = torch.tensor([class_to_idx[cls] for cls in sample_dataframe['class']])
        # 
        # trainer.initialize_representatives(embeddings, labels)
        # trainer.train(embeddings, labels, epochs=config['epochs'])
        # 
        # elapsed_time = time.time() - start_time
        # 
        # # Should complete in reasonable time (adjust threshold as needed)
        # assert elapsed_time < 30.0, f"Training took too long: {elapsed_time:.2f} seconds"
        
        # For now, verify test data size
        assert len(sample_dataframe) == 30  # 3 classes × 10 samples
        embedding_dim = len(sample_dataframe['finetuned_embedding'].iloc[0])
        assert embedding_dim == 2048  # Standard embedding dimension
    
    def test_memory_usage_reasonable(self, sample_dataframe):
        """Test that memory usage stays within reasonable bounds."""
        # When implemented, could monitor memory usage during training
        # For now, just verify data structures are reasonable size
        
        embeddings_list = sample_dataframe['finetuned_embedding'].tolist()
        embeddings = torch.tensor(np.stack(embeddings_list), dtype=torch.float32)
        
        # Calculate memory usage
        embedding_memory = embeddings.numel() * embeddings.element_size()  # bytes
        embedding_memory_mb = embedding_memory / (1024 * 1024)  # MB
        
        # Should be reasonable for test data
        assert embedding_memory_mb < 100, f"Test embeddings use too much memory: {embedding_memory_mb:.2f} MB"
    
    def test_handles_large_number_of_classes(self):
        """Test behavior with many classes (stress test)."""
        # Create synthetic data with many classes
        torch.manual_seed(42)
        num_classes = 50
        samples_per_class = 2
        embedding_dim = 128  # Smaller for speed
        
        data = []
        for class_id in range(num_classes):
            # Create class-specific center
            center = torch.randn(embedding_dim)
            for sample_id in range(samples_per_class):
                embedding = center + 0.1 * torch.randn(embedding_dim)
                data.append({
                    'class': f'class_{class_id}',
                    'finetuned_embedding': embedding.numpy()
                })
        
        large_df = pd.DataFrame(data)
        
        # When implemented:
        # config = {
        #     'lr': 0.01,
        #     'margin': 0.15,
        #     'lambda_push': 0.25,
        #     'epochs': 5  # Short for stress test
        # }
        # 
        # trainer = ContrastiveTrainer(config)
        # 
        # embeddings_list = large_df['finetuned_embedding'].tolist()
        # embeddings = torch.tensor(np.stack(embeddings_list), dtype=torch.float32)
        # 
        # unique_classes = large_df['class'].unique()
        # class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        # labels = torch.tensor([class_to_idx[cls] for cls in large_df['class']])
        # 
        # # Should handle many classes without crashing
        # trainer.initialize_representatives(embeddings, labels)
        # history = trainer.train(embeddings, labels, epochs=config['epochs'])
        # 
        # assert len(history['losses']) == config['epochs']
        # assert all(torch.isfinite(torch.tensor(loss)) for loss in history['losses'])
        
        # For now, verify test data structure
        assert len(large_df) == num_classes * samples_per_class
        assert len(large_df['class'].unique()) == num_classes 