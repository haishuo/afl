#!/usr/bin/env python3
"""
MNIST MLP AFL Experiment
========================

AFL determination experiment for MNIST dataset with MLP architecture.
Location: /mnt/projects/afl/experiments/mnist/mnist_mlp_afl.py

This module implements:
- MNIST dataset loading and preprocessing
- MLP architecture definition (256->128->64->10)
- Model training to convergence
- AFL determination through systematic random pruning
- Results saving and analysis
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
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import copy

# Import AFL framework
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from shared.forge_config import get_afl_config
from core.afl_framework import AFLFramework, AFLConfig, AFLResult

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST MLP Architecture
class MNISTMLP(nn.Module):
    """Simple MLP for MNIST classification (256->128->64->10)."""
    
    def __init__(self, input_size=784, hidden1=256, hidden2=128, hidden3=64, num_classes=10):
        super(MNISTMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, num_classes)
        self.dropout = nn.Dropout(0.2)
        
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


def load_mnist_data(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    """Load MNIST dataset with standard preprocessing."""
    
    # Transform for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Get dataset path from AFL config
    config = get_afl_config()
    
    # Try to load from our manually downloaded data first
    try:
        # Load datasets - let PyTorch download if our manual download doesn't work
        train_dataset = datasets.MNIST(
            root=str(config.common_datasets_dir),  # Use common_datasets as root
            train=True,
            download=True,  # Allow download if needed
            transform=transform
        )
        
        test_dataset = datasets.MNIST(
            root=str(config.common_datasets_dir),
            train=False,
            download=True,  # Allow download if needed
            transform=transform
        )
        
    except Exception as e:
        logger.warning(f"Failed to load manually downloaded MNIST: {str(e)}")
        logger.info("Falling back to PyTorch automatic download...")
        
        # Fallback: let PyTorch handle everything
        train_dataset = datasets.MNIST(
            root=str(config.common_datasets_dir / "mnist_pytorch"),
            train=True,
            download=True,
            transform=transform
        )
        
        test_dataset = datasets.MNIST(
            root=str(config.common_datasets_dir / "mnist_pytorch"),
            train=False,
            download=True,
            transform=transform
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    logger.info(f"MNIST loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
    return train_loader, test_loader


def train_mnist_mlp(train_loader: DataLoader, test_loader: DataLoader, 
                   epochs: int = 20) -> MNISTMLP:
    """Train MNIST MLP to convergence."""
    
    logger.info(f"Training MNIST MLP for {epochs} epochs on {DEVICE}")
    
    # Create model
    model = MNISTMLP().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        
        # Evaluate on test set
        test_acc = evaluate_model(model, test_loader)
        
        logger.info(f"Epoch {epoch+1}/{epochs}: "
                   f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
                   f"Test Acc: {test_acc:.2f}%")
    
    logger.info("Training completed")
    return model


def evaluate_model(model: nn.Module, test_loader: DataLoader) -> float:
    """Evaluate model accuracy on test set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def apply_random_pruning(model: nn.Module, sparsity: float, seed: int) -> nn.Module:
    """Apply random pruning to FC layers at specified sparsity level."""
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create a copy of the model
    pruned_model = copy.deepcopy(model)
    
    # Apply random pruning to each linear layer
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Linear):
            # Get weight tensor
            weight = module.weight.data
            
            # Create random mask
            num_weights = weight.numel()
            num_prune = int(num_weights * sparsity)
            
            # Random indices to prune
            flat_weight = weight.view(-1)
            indices = torch.randperm(num_weights)[:num_prune]
            
            # Apply pruning mask
            flat_weight[indices] = 0.0
            
            # Reshape back
            module.weight.data = flat_weight.view(weight.shape)
    
    return pruned_model


def fine_tune_model(model: nn.Module, train_loader: DataLoader, epochs: int = 5) -> nn.Module:
    """Fine-tune pruned model for recovery."""
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for data, targets in train_loader:
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    return model


class MNISTMLFAFL(AFLFramework):
    """AFL Framework implementation for MNIST MLP."""
    
    def _evaluate_model(self, model: nn.Module, test_loader) -> float:
        """Evaluate model accuracy on test set."""
        return evaluate_model(model, test_loader)
    
    def _count_parameters(self, model: nn.Module) -> int:
        """Count total trainable parameters in model."""
        return count_parameters(model)
    
    def _copy_model(self, model: nn.Module) -> nn.Module:
        """Create deep copy of model."""
        return copy.deepcopy(model)
    
    def _apply_random_pruning(self, model: nn.Module, sparsity: float, seed: int) -> nn.Module:
        """Apply random pruning at specified sparsity level."""
        return apply_random_pruning(model, sparsity, seed)
    
    def _fine_tune_model(self, model: nn.Module, train_loader, epochs: int) -> nn.Module:
        """Fine-tune pruned model."""
        return fine_tune_model(model, train_loader, epochs)


def run_mnist_mlp_afl() -> AFLResult:
    """Run complete MNIST MLP AFL determination experiment."""
    
    logger.info("🚀 Starting MNIST MLP AFL Experiment")
    logger.info("=" * 50)
    
    start_time = time.time()
    
    # Setup paths
    config = get_afl_config()
    experiment_name = "mnist_mlp_256_128_64"
    models_dir = config.get_experiment_models_dir(experiment_name)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("📂 Loading MNIST dataset...")
    train_loader, test_loader = load_mnist_data()
    
    # Check if baseline model already exists
    baseline_model_path = models_dir / "baseline_model.pth"
    
    if baseline_model_path.exists():
        logger.info("📦 Loading existing baseline model...")
        model = MNISTMLP().to(DEVICE)
        checkpoint = torch.load(baseline_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        baseline_accuracy = checkpoint['test_accuracy']
        logger.info(f"Loaded model with {baseline_accuracy:.2f}% test accuracy")
    else:
        logger.info("🏋️ Training baseline model...")
        model = train_mnist_mlp(train_loader, test_loader)
        
        # Evaluate and save
        baseline_accuracy = evaluate_model(model, test_loader)
        
        # Save baseline model
        torch.save({
            'model_state_dict': model.state_dict(),
            'test_accuracy': baseline_accuracy,
            'architecture': 'MLP_256_128_64',
            'dataset': 'MNIST',
            'parameters': count_parameters(model)
        }, baseline_model_path)
        
        logger.info(f"✅ Baseline model saved: {baseline_accuracy:.2f}% accuracy")
    
    # Run AFL determination
    logger.info("🔬 Starting AFL determination...")
    
    afl_config = AFLConfig(
        sparsity_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        trials_per_level=30,
        confidence_level=0.95,
        fine_tune_epochs=5
    )
    
    afl_framework = MNISTMLFAFL(afl_config)
    
    result = afl_framework.determine_afl(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        dataset_name="MNIST",
        architecture_name="MLP_256_128_64"
    )
    
    # Update experiment duration
    result.experiment_duration = time.time() - start_time
    
    # Save results
    results_path = config.get_experiment_report_path(experiment_name)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert result to dict for JSON serialization
    results_dict = {
        'dataset': result.dataset,
        'architecture': result.architecture,
        'parameter_count': result.parameter_count,
        'baseline_accuracy': result.baseline_accuracy,
        'afl_value': result.afl_value,
        'sparsity_results': result.sparsity_results,
        'experiment_duration': result.experiment_duration,
        'recommendation': result.recommendation,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info(f"💾 Results saved to: {results_path}")
    
    # Print summary
    logger.info("\n📊 MNIST MLP AFL EXPERIMENT COMPLETE")
    logger.info("=" * 45)
    logger.info(f"Dataset: {result.dataset}")
    logger.info(f"Architecture: {result.architecture}")
    logger.info(f"Parameters: {result.parameter_count:,}")
    logger.info(f"Baseline Accuracy: {result.baseline_accuracy:.2f}%")
    logger.info(f"AFL Value: {result.afl_value}")
    logger.info(f"Recommendation: {result.recommendation}")
    logger.info(f"Duration: {result.experiment_duration/60:.1f} minutes")
    
    return result


if __name__ == "__main__":
    result = run_mnist_mlp_afl()