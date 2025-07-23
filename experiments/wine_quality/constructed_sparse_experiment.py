#!/usr/bin/env python3
"""
Wine Quality Constructed Sparse Network Experiment
==================================================

Location: /mnt/projects/afl/experiments/wine_quality/constructed_sparse_experiment.py

This experiment tests networks built at target sparsity from initialization
to compare against pruned networks. This validates whether the AFL framework
is meaningful by checking if pruning ≈ constructed sparse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import os
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from shared.forge_config import get_afl_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConstructedSparseWineQualityMLP(nn.Module):
    """MLP constructed with specified layer sizes from initialization."""
    
    def __init__(self, layer_sizes: List[int], input_size: int = 11, 
                 num_classes: int = 1, dropout: float = 0.2):
        super(ConstructedSparseWineQualityMLP, self).__init__()
        
        # Build network with specified sizes
        self.fc1 = nn.Linear(input_size, layer_sizes[0])
        self.fc2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.fc3 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.fc4 = nn.Linear(layer_sizes[2], num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


def calculate_sparse_architecture(original_sizes: List[int], sparsity: float) -> List[int]:
    """
    Calculate layer sizes for constructed sparse network.
    
    Args:
        original_sizes: Original layer sizes [256, 128, 64]
        sparsity: Target sparsity level (0.0 to 1.0)
        
    Returns:
        New layer sizes maintaining proportions
    """
    total_original = sum(original_sizes)
    target_total = int(total_original * (1 - sparsity))
    
    # Ensure at least 3 neurons (one per layer)
    if target_total < len(original_sizes):
        return [1] * len(original_sizes)
    
    # Distribute proportionally
    proportions = [size / total_original for size in original_sizes]
    
    new_sizes = []
    allocated = 0
    
    for i, prop in enumerate(proportions[:-1]):
        size = max(1, int(target_total * prop))
        new_sizes.append(size)
        allocated += size
    
    # Last layer gets remainder
    new_sizes.append(max(1, target_total - allocated))
    
    return new_sizes


def load_wine_quality_data(batch_size: int = 32) -> Tuple[DataLoader, DataLoader, StandardScaler]:
    """Load and preprocess Wine Quality dataset."""
    
    # Load dataset
    config = get_afl_config()
    dataset_path = config.get_dataset_path("wine_quality") / "winequality-red.csv"
    
    logger.info(f"Loading dataset from: {dataset_path}")
    data = pd.read_csv(dataset_path, delimiter=';')
    
    # Prepare features and target
    X = data.drop('quality', axis=1).values
    y = data['quality'].values.astype(float)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create datasets and loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler


def train_constructed_sparse_model(model: nn.Module, train_loader: DataLoader, 
                                 epochs: int = 50, learning_rate: float = 0.001) -> float:
    """Train a constructed sparse model from scratch."""
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    return loss.item()


def evaluate_model(model: nn.Module, test_loader: DataLoader) -> Tuple[float, float]:
    """
    Evaluate model on test set.
    Returns: (accuracy, mse)
    """
    model.eval()
    correct = 0
    total = 0
    mse_sum = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            outputs = model(data)
            
            # Round predictions for accuracy
            predicted = torch.round(outputs)
            
            # Clip to valid range [3, 9]
            predicted = torch.clamp(predicted, 3, 9)
            targets_rounded = torch.round(targets)
            
            total += targets.size(0)
            correct += (predicted == targets_rounded).sum().item()
            
            # Calculate MSE
            mse_sum += ((outputs - targets) ** 2).sum().item()
    
    accuracy = 100 * correct / total
    mse = mse_sum / total
    
    return accuracy, mse


def train_single_trial(args):
    """Train a single trial - for parallel execution."""
    trial_id, sparse_sizes, sparsity, train_loader, test_loader, epochs = args
    
    # Set seed
    torch.manual_seed(42 + trial_id + int(sparsity * 1000))
    np.random.seed(42 + trial_id + int(sparsity * 1000))
    
    # Create and train model
    model = ConstructedSparseWineQualityMLP(sparse_sizes).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Train model
    train_start = time.time()
    final_loss = train_constructed_sparse_model(model, train_loader, epochs=epochs)
    
    # Evaluate
    accuracy, mse = evaluate_model(model, test_loader)
    trial_time = time.time() - train_start
    
    # Clean GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "trial": trial_id,
        "accuracy": accuracy,
        "mse": mse,
        "training_time": trial_time,
        "final_loss": final_loss,
        "total_parameters": total_params
    }


def run_constructed_sparse_experiment():
    """Run complete constructed sparse experiment for Wine Quality."""
    
    logger.info("🚀 Starting Wine Quality Constructed Sparse Experiment")
    logger.info("=" * 50)
    
    # Check if GPU available for parallel execution
    use_parallel = torch.cuda.is_available() and int(os.environ.get('AFL_MAX_PARALLEL', '1')) > 1
    max_parallel = int(os.environ.get('AFL_MAX_PARALLEL', '8')) if use_parallel else 1
    
    if use_parallel:
        logger.info(f"🔥 GPU detected - running {max_parallel} trials in parallel")
    else:
        logger.info("🐌 Running trials sequentially")
    
    experiment_start = time.time()
    
    # Configuration
    original_architecture = [256, 128, 64]
    sparsity_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    trials_per_level = 30
    training_epochs = 50
    
    # Setup paths
    config = get_afl_config()
    experiment_name = "wine_quality_constructed_sparse_256_128_64"
    report_path = config.get_experiment_report_path(experiment_name)
    
    # Load data
    logger.info("📂 Loading Wine Quality dataset...")
    train_loader, test_loader, scaler = load_wine_quality_data()
    
    # Results storage
    results = {
        "experiment": experiment_name,
        "dataset": "Wine Quality",
        "original_architecture": original_architecture,
        "sparsity_levels": sparsity_levels,
        "trials_per_level": trials_per_level,
        "training_epochs": training_epochs,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sparsity_results": {},
        "detailed_results": []
    }
    
    # Run experiments for each sparsity level
    for sparsity in sparsity_levels:
        logger.info(f"\n📊 Testing sparsity: {sparsity:.0%}")
        logger.info("-" * 40)
        
        # Calculate sparse architecture
        sparse_sizes = calculate_sparse_architecture(original_architecture, sparsity)
        actual_sparsity = 1 - sum(sparse_sizes) / sum(original_architecture)
        
        logger.info(f"Original architecture: {original_architecture} (total: {sum(original_architecture)})")
        logger.info(f"Sparse architecture: {sparse_sizes} (total: {sum(sparse_sizes)})")
        logger.info(f"Actual sparsity: {actual_sparsity:.1%}")
        
        # Run multiple trials
        if use_parallel:
            # Parallel execution
            trial_args = [(trial, sparse_sizes, sparsity, train_loader, test_loader, training_epochs) 
                         for trial in range(trials_per_level)]
            
            # Run in batches to avoid overwhelming GPU
            trial_results = []
            for i in range(0, trials_per_level, max_parallel):
                batch = trial_args[i:i+max_parallel]
                with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                    batch_results = list(executor.map(train_single_trial, batch))
                    trial_results.extend(batch_results)
                
                # Log progress
                logger.info(f"  Progress: {len(trial_results)}/{trials_per_level} trials completed")
                if torch.cuda.is_available():
                    mem_used = torch.cuda.memory_allocated() / 1024**3
                    mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    logger.info(f"  GPU Memory: {mem_used:.1f}/{mem_total:.1f} GB")
        else:
            # Sequential execution
            trial_results = []
            for trial in range(trials_per_level):
                result = train_single_trial((trial, sparse_sizes, sparsity, 
                                           train_loader, test_loader, training_epochs))
                trial_results.append(result)
                
                if (trial + 1) % 10 == 0:
                    logger.info(f"  Progress: {trial + 1}/{trials_per_level} trials completed")
        
        # Extract accuracies, MSEs and times
        trial_accuracies = [r["accuracy"] for r in trial_results]
        trial_mses = [r["mse"] for r in trial_results]
        trial_times = [r["training_time"] for r in trial_results]
        
        # Store detailed results
        for result in trial_results:
            results["detailed_results"].append({
                "sparsity": sparsity,
                "trial": result["trial"],
                "accuracy": result["accuracy"],
                "mse": result["mse"],
                "architecture": sparse_sizes,
                "total_parameters": result["total_parameters"],
                "training_time": result["training_time"],
                "final_loss": result["final_loss"]
            })
        
        # Compute statistics for this sparsity level
        results["sparsity_results"][f"{sparsity:.1f}"] = {
            "architecture": sparse_sizes,
            "actual_sparsity": actual_sparsity,
            "accuracies": trial_accuracies,
            "mean_accuracy": np.mean(trial_accuracies),
            "std_accuracy": np.std(trial_accuracies),
            "min_accuracy": min(trial_accuracies),
            "max_accuracy": max(trial_accuracies),
            "mean_mse": np.mean(trial_mses),
            "std_mse": np.std(trial_mses),
            "mean_training_time": np.mean(trial_times),
            "total_parameters": total_params  # This will use the last model's params
        }
        
        logger.info(f"\n📈 Results for {sparsity:.0%} sparsity:")
        logger.info(f"  Mean accuracy: {np.mean(trial_accuracies):.2f}% ± {np.std(trial_accuracies):.2f}%")
        logger.info(f"  Range: [{min(trial_accuracies):.2f}%, {max(trial_accuracies):.2f}%]")
        logger.info(f"  Mean MSE: {np.mean(trial_mses):.4f} ± {np.std(trial_mses):.4f}")
        logger.info(f"  Mean training time: {np.mean(trial_times):.1f}s")
    
    # Add experiment metadata
    experiment_duration = time.time() - experiment_start
    results["experiment_duration_hours"] = experiment_duration / 3600
    results["device"] = str(DEVICE)
    
    # Save results
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Experiment complete!")
    logger.info(f"📄 Results saved to: {report_path}")
    logger.info(f"⏱️  Total duration: {experiment_duration / 3600:.2f} hours")
    
    # Print summary
    logger.info("\n📊 SUMMARY:")
    logger.info(f"{'Sparsity':>10} | {'Architecture':>20} | {'Mean Accuracy':>15} | {'Mean MSE':>10}")
    logger.info("-" * 60)
    
    for sparsity in sparsity_levels:
        stats = results["sparsity_results"][f"{sparsity:.1f}"]
        arch_str = f"{stats['architecture']}"
        logger.info(f"{sparsity:>9.0%} | {arch_str:>20} | {stats['mean_accuracy']:>14.2f}% | {stats['mean_mse']:>9.4f}")
    
    return results


if __name__ == "__main__":
    # Check for GPU optimization flag
    if os.environ.get('AFL_MAX_PARALLEL', '1') != '1':
        logger.info("🔥 GPU parallelization enabled")
    
    run_constructed_sparse_experiment()