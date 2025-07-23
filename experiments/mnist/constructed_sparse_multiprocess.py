#!/usr/bin/env python3
"""
Multiprocessing version - bypasses GIL for true parallelism
"""

import torch
import torch.multiprocessing as mp
import numpy as np
import time
import logging
from pathlib import Path
import json
import sys

# IMPORTANT: Set start method before any CUDA operations
mp.set_start_method('spawn', force=True)

sys.path.append('/mnt/projects/afl')
from experiments.mnist.constructed_sparse_experiment import (
    ConstructedSparseMLP, calculate_sparse_architecture,
    train_constructed_sparse_model, evaluate_model, load_mnist_data
)
from shared.forge_config import get_afl_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_batch_of_models(args):
    """Train a batch of models in a separate process."""
    sparsity, trial_ids, sparse_sizes, epochs = args
    
    # Each process needs its own CUDA context
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data in this process
    train_loader, test_loader = load_mnist_data(batch_size=128)
    
    results = []
    for trial_id in trial_ids:
        try:
            # Set seed
            torch.manual_seed(42 + trial_id + int(sparsity * 1000))
            np.random.seed(42 + trial_id + int(sparsity * 1000))
            
            # Create and train model
            model = ConstructedSparseMLP(sparse_sizes).to(device)
            
            start_time = time.time()
            final_loss = train_constructed_sparse_model(model, train_loader, epochs=epochs)
            accuracy = evaluate_model(model, test_loader)
            training_time = time.time() - start_time
            
            # Get parameters before deleting model
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Cleanup
            del model
            torch.cuda.empty_cache()
            
            results.append({
                "sparsity": sparsity,
                "trial": trial_id,
                "accuracy": accuracy,
                "training_time": training_time,
                "final_loss": final_loss,
                "total_parameters": total_params,
                "architecture": sparse_sizes
            })
            
            print(f"[Process {mp.current_process().name}] S:{sparsity:.0%} T:{trial_id} "
                  f"Acc:{accuracy:.1f}% Time:{training_time:.0f}s")
            
        except Exception as e:
            print(f"Error in trial {trial_id}: {e}")
            
    return results


def run_multiprocess_experiment():
    """Run experiment using multiprocessing for true parallelism."""
    
    print("🚀 Multiprocessing Parallel Training")
    print("=" * 60)
    
    # Configuration
    original_architecture = [256, 128, 64]
    sparsity_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    trials_per_level = 30
    training_epochs = 50
    
    # Number of processes (usually = number of CPU cores)
    n_processes = min(mp.cpu_count(), 8)  # Cap at 8 to be reasonable
    print(f"Using {n_processes} processes")
    
    experiment_start = time.time()
    all_results = []
    
    # Process each sparsity level
    for sparsity in sparsity_levels:
        print(f"\n📊 Sparsity {sparsity:.0%}")
        print("-" * 40)
        
        sparse_sizes = calculate_sparse_architecture(original_architecture, sparsity)
        print(f"Architecture: {sparse_sizes}")
        
        # Split trials across processes
        trials_per_process = trials_per_level // n_processes
        extra_trials = trials_per_level % n_processes
        
        # Create tasks for each process
        tasks = []
        trial_start = 0
        for i in range(n_processes):
            n_trials = trials_per_process + (1 if i < extra_trials else 0)
            if n_trials > 0:
                trial_ids = list(range(trial_start, trial_start + n_trials))
                tasks.append((sparsity, trial_ids, sparse_sizes, training_epochs))
                trial_start += n_trials
        
        # Run processes
        print(f"Starting {len(tasks)} processes...")
        with mp.Pool(processes=n_processes) as pool:
            process_results = pool.map(train_batch_of_models, tasks)
        
        # Flatten results
        sparsity_results = []
        for batch_results in process_results:
            sparsity_results.extend(batch_results)
        
        all_results.extend(sparsity_results)
        
        # Print statistics
        accuracies = [r["accuracy"] for r in sparsity_results]
        print(f"Completed {len(accuracies)} trials")
        print(f"Mean accuracy: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
    
    experiment_duration = time.time() - experiment_start
    
    # Save results
    config = get_afl_config()
    experiment_name = "mnist_constructed_sparse_multiprocess_256_128_64"
    report_path = config.get_experiment_report_path(experiment_name)
    
    # Format results (same as original experiment)
    formatted_results = {
        "experiment": experiment_name,
        "dataset": "MNIST",
        "original_architecture": original_architecture,
        "sparsity_levels": sparsity_levels,
        "trials_per_level": trials_per_level,
        "training_epochs": training_epochs,
        "n_processes": n_processes,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_duration_hours": experiment_duration / 3600,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "sparsity_results": {},
        "detailed_results": all_results
    }
    
    # Calculate statistics for each sparsity
    for sparsity in sparsity_levels:
        sparsity_trials = [r for r in all_results if r["sparsity"] == sparsity]
        if sparsity_trials:
            accuracies = [r["accuracy"] for r in sparsity_trials]
            formatted_results["sparsity_results"][f"{sparsity:.1f}"] = {
                "architecture": sparsity_trials[0]["architecture"],
                "actual_sparsity": sparsity,
                "accuracies": accuracies,
                "mean_accuracy": np.mean(accuracies),
                "std_accuracy": np.std(accuracies),
                "min_accuracy": min(accuracies),
                "max_accuracy": max(accuracies),
                "mean_training_time": np.mean([r["training_time"] for r in sparsity_trials]),
                "total_parameters": sparsity_trials[0]["total_parameters"]
            }
    
    # Save
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(formatted_results, f, indent=2)
    
    print(f"\n✅ Complete!")
    print(f"⏱️  Total: {experiment_duration/60:.1f} minutes")
    print(f"📄 Results: {report_path}")
    
    # Summary
    print("\n📊 SUMMARY:")
    for sparsity in sparsity_levels:
        if f"{sparsity:.1f}" in formatted_results["sparsity_results"]:
            stats = formatted_results["sparsity_results"][f"{sparsity:.1f}"]
            print(f"Sparsity {sparsity:>4.0%}: {stats['mean_accuracy']:>6.2f}% ± {stats['std_accuracy']:>4.2f}%")


if __name__ == "__main__":
    # IMPORTANT: This must be inside if __name__ == "__main__" for multiprocessing
    run_multiprocess_experiment()