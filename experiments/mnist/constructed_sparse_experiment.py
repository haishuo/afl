#!/usr/bin/env python3
"""
MNIST Constructed Sparse Network Experiment
===========================================

Location: /mnt/projects/afl/experiments/mnist/constructed_sparse_experiment.py

This experiment tests networks built at target sparsity from initialization
to compare against pruned networks. This validates whether the AFL framework
is meaningful by checking if pruning ≈ constructed sparse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from shared.forge_config import get_afl_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConstructedSparseMLP(nn.Module):
    """MLP constructed with specified layer sizes from initialization."""
    
    def __init__(self, layer_sizes: List[int], input_size: int = 784, 
                 num_classes: int = 10, dropout: float = 0.2):
        super(ConstructedSparseMLP, self).__init__()
        self.flatten = nn.Flatten()
        
        # Build network with specified sizes
        self.fc1 = nn.Linear(input_size, layer_sizes[0])
        self.fc2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.fc3 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.fc4 = nn.Linear(layer_sizes[2], num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.flatten(x)
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


def train_constructed_sparse_model(model: nn.Module, train_loader: DataLoader, 
                                 epochs: int = 50, learning_rate: float = 0.001) -> float:
    """Train a constructed sparse model from scratch."""
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    return loss.item()


def evaluate_model(model: nn.Module, test_loader: DataLoader) -> float:
    """Evaluate model accuracy on test set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    return 100 * correct / total


def load_mnist_data(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return train_loader, test_loader


def run_constructed_sparse_experiment():
    """Run complete constructed sparse experiment for MNIST."""
    
    logger.info("🚀 Starting MNIST Constructed Sparse Experiment")
    logger.info("=" * 50)
    
    experiment_start = time.time()
    
    # Configuration
    original_architecture = [256, 128, 64]
    sparsity_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    trials_per_level = 30
    training_epochs = 50
    
    # Setup paths
    config = get_afl_config()
    experiment_name = "mnist_constructed_sparse_256_128_64"
    report_path = config.get_experiment_report_path(experiment_name)
    
    # Load data
    logger.info("📂 Loading MNIST dataset...")
    train_loader, test_loader = load_mnist_data()
    
    # Results storage
    results = {
        "experiment": experiment_name,
        "dataset": "MNIST",
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
        
        trial_accuracies = []
        trial_times = []
        
        # Run multiple trials
        for trial in range(trials_per_level):
            trial_start = time.time()
            
            # Set seed for reproducibility
            torch.manual_seed(42 + trial + int(sparsity * 1000))
            np.random.seed(42 + trial + int(sparsity * 1000))
            
            # Create and train model
            model = ConstructedSparseMLP(sparse_sizes).to(DEVICE)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Train model
            final_loss = train_constructed_sparse_model(model, train_loader, epochs=training_epochs)
            
            # Evaluate
            accuracy = evaluate_model(model, test_loader)
            trial_time = time.time() - trial_start
            
            trial_accuracies.append(accuracy)
            trial_times.append(trial_time)
            
            # Store detailed result
            results["detailed_results"].append({
                "sparsity": sparsity,
                "trial": trial,
                "accuracy": accuracy,
                "architecture": sparse_sizes,
                "total_parameters": total_params,
                "training_time": trial_time,
                "final_loss": final_loss
            })
            
            if (trial + 1) % 10 == 0:
                logger.info(f"  Progress: {trial + 1}/{trials_per_level} trials completed")
                logger.info(f"  Current mean accuracy: {np.mean(trial_accuracies):.2f}%")
        
        # Compute statistics for this sparsity level
        results["sparsity_results"][f"{sparsity:.1f}"] = {
            "architecture": sparse_sizes,
            "actual_sparsity": actual_sparsity,
            "accuracies": trial_accuracies,
            "mean_accuracy": np.mean(trial_accuracies),
            "std_accuracy": np.std(trial_accuracies),
            "min_accuracy": min(trial_accuracies),
            "max_accuracy": max(trial_accuracies),
            "mean_training_time": np.mean(trial_times),
            "total_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        logger.info(f"\n📈 Results for {sparsity:.0%} sparsity:")
        logger.info(f"  Mean accuracy: {np.mean(trial_accuracies):.2f}% ± {np.std(trial_accuracies):.2f}%")
        logger.info(f"  Range: [{min(trial_accuracies):.2f}%, {max(trial_accuracies):.2f}%]")
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
    logger.info(f"{'Sparsity':>10} | {'Architecture':>20} | {'Mean Accuracy':>15}")
    logger.info("-" * 50)
    
    for sparsity in sparsity_levels:
        stats = results["sparsity_results"][f"{sparsity:.1f}"]
        arch_str = f"{stats['architecture']}"
        logger.info(f"{sparsity:>9.0%} | {arch_str:>20} | {stats['mean_accuracy']:>14.2f}%")
    
    return results


if __name__ == "__main__":
    run_constructed_sparse_experiment()