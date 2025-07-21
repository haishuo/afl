#!/usr/bin/env python3
"""
AFL Experiment Runner
====================

Location: /mnt/projects/afl/scripts/run_afl_experiment.py

Main script for running AFL experiments.
Can run individual experiments or all implemented combinations.

Usage:
    python3 run_afl_experiment.py                    # Run all implemented experiments  
    python3 run_afl_experiment.py --experiment mnist_mlp_256_128_64  # Run specific experiment
    python3 run_afl_experiment.py --list            # List available experiments
"""

import sys
import argparse
import logging
from pathlib import Path
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from shared.forge_config import get_afl_config, initialize_afl_directories

# Import experiment modules
from experiments.mnist.mnist_mlp_afl import run_mnist_mlp_afl
from experiments.wine_quality.wine_quality_mlp_afl import run_wine_quality_mlp_afl

# Available experiments registry
AVAILABLE_EXPERIMENTS = {
    "mnist_mlp_256_128_64": {
        "name": "MNIST MLP 256->128->64",
        "description": "MNIST dataset with 3-layer MLP architecture",
        "function": run_mnist_mlp_afl,
        "status": "implemented",
        "estimated_time": "30-60 minutes"
    },
    "wine_quality_mlp_256_128_64": {
        "name": "Wine Quality MLP 256->128->64",
        "description": "Wine Quality dataset with 3-layer MLP architecture",
        "function": run_wine_quality_mlp_afl,
        "status": "implemented",
        "estimated_time": "20-40 minutes"
    },
    # Future experiments (not yet implemented)
    "cifar10_resnet18": {
        "name": "CIFAR-10 ResNet-18", 
        "description": "CIFAR-10 dataset with ResNet-18 architecture",
        "function": None,
        "status": "planned",
        "estimated_time": "2-4 hours"
    },
    "cifar100_resnet18": {
        "name": "CIFAR-100 ResNet-18",
        "description": "CIFAR-100 dataset with ResNet-18 architecture",
        "function": None, 
        "status": "planned",
        "estimated_time": "3-5 hours"
    },
    "fashion_mnist_cnn": {
        "name": "Fashion-MNIST CNN",
        "description": "Fashion-MNIST dataset with CNN architecture",
        "function": None,
        "status": "planned", 
        "estimated_time": "1-2 hours"
    }
}


def setup_logging(experiment_name: str = None) -> None:
    """Setup logging configuration."""
    config = get_afl_config()
    
    if experiment_name:
        log_path = config.get_log_path(experiment_name)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_path = config.logs_dir / "afl_runner.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )


def list_experiments() -> None:
    """List all available experiments with their status."""
    print("\n📋 Available AFL Experiments")
    print("=" * 50)
    
    implemented = []
    planned = []
    
    for exp_id, exp_info in AVAILABLE_EXPERIMENTS.items():
        if exp_info["status"] == "implemented":
            implemented.append((exp_id, exp_info))
        else:
            planned.append((exp_id, exp_info))
    
    print(f"\n✅ Implemented ({len(implemented)}):")
    for exp_id, exp_info in implemented:
        print(f"  {exp_id}")
        print(f"    Name: {exp_info['name']}")
        print(f"    Description: {exp_info['description']}")
        print(f"    Estimated time: {exp_info['estimated_time']}")
        print()
    
    print(f"🚧 Planned ({len(planned)}):")
    for exp_id, exp_info in planned:
        print(f"  {exp_id}")
        print(f"    Name: {exp_info['name']}")
        print(f"    Description: {exp_info['description']}")
        print(f"    Estimated time: {exp_info['estimated_time']}")
        print()


def run_single_experiment(experiment_id: str) -> bool:
    """Run a single AFL experiment."""
    
    if experiment_id not in AVAILABLE_EXPERIMENTS:
        print(f"❌ Unknown experiment: {experiment_id}")
        print(f"Available experiments: {', '.join(AVAILABLE_EXPERIMENTS.keys())}")
        return False
    
    exp_info = AVAILABLE_EXPERIMENTS[experiment_id]
    
    if exp_info["status"] != "implemented":
        print(f"❌ Experiment not yet implemented: {experiment_id}")
        print(f"Status: {exp_info['status']}")
        return False
    
    print(f"\n🚀 Running AFL Experiment: {experiment_id}")
    print("=" * 60)
    print(f"Name: {exp_info['name']}")
    print(f"Description: {exp_info['description']}")
    print(f"Estimated time: {exp_info['estimated_time']}")
    print()
    
    # Setup logging for this experiment
    setup_logging(experiment_id)
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    
    try:
        # Run the experiment
        result = exp_info["function"]()
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ Experiment completed successfully!")
        print(f"⏱️  Total time: {elapsed_time/60:.1f} minutes")
        print(f"📊 AFL Result: {result.afl_value}")
        print(f"🎯 Recommendation: {result.recommendation}")
        
        logger.info(f"Experiment {experiment_id} completed successfully")
        logger.info(f"AFL Value: {result.afl_value}")
        logger.info(f"Recommendation: {result.recommendation}")
        
        return True
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        
        print(f"\n❌ Experiment failed after {elapsed_time/60:.1f} minutes")
        print(f"Error: {str(e)}")
        
        logger.error(f"Experiment {experiment_id} failed: {str(e)}")
        return False


def run_all_experiments() -> None:
    """Run all implemented AFL experiments."""
    # Get list of implemented experiments
    implemented_experiments = [
        exp_id for exp_id, exp_info in AVAILABLE_EXPERIMENTS.items() 
        if exp_info["status"] == "implemented"
    ]
    
    if not implemented_experiments:
        print("❌ No implemented experiments found!")
        return
    
    print(f"\n🚀 Running ALL AFL Experiments")
    print("=" * 50)
    print(f"Found {len(implemented_experiments)} implemented experiments:")
    for exp_id in implemented_experiments:
        exp_info = AVAILABLE_EXPERIMENTS[exp_id]
        print(f"  - {exp_id}: {exp_info['name']}")
    print()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Track results
    results = {}
    start_time = time.time()
    
    for i, experiment_id in enumerate(implemented_experiments, 1):
        print(f"\n📊 Running experiment {i}/{len(implemented_experiments)}: {experiment_id}")
        print("-" * 40)
        
        success = run_single_experiment(experiment_id)
        results[experiment_id] = success
        
        if success:
            print(f"✅ {experiment_id} completed")
        else:
            print(f"❌ {experiment_id} failed")
    
    # Summary
    total_time = time.time() - start_time
    successful = sum(results.values())
    failed = len(results) - successful
    
    print(f"\n📈 ALL EXPERIMENTS SUMMARY")
    print("=" * 40)
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"✅ Successful: {successful}/{len(results)} experiments")
    print(f"❌ Failed: {failed}/{len(results)} experiments")
    
    if successful > 0:
        print(f"\n✅ Successful experiments:")
        for exp_id, success in results.items():
            if success:
                print(f"  - {exp_id}")
    
    if failed > 0:
        print(f"\n❌ Failed experiments:")
        for exp_id, success in results.items():
            if not success:
                print(f"  - {exp_id}")
    
    logger.info(f"Batch run completed: {successful}/{len(results)} successful")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run AFL experiments")
    parser.add_argument("--experiment", type=str, help="Run specific experiment")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    parser.add_argument("--status", action="store_true", help="Show experiment status")
    
    args = parser.parse_args()
    
    print("🧪 AFL Experiment Runner")
    print("=" * 30)
    
    # Initialize AFL directories
    initialize_afl_directories()
    
    if args.list or args.status:
        list_experiments()
        return
    
    if args.experiment:
        # Run specific experiment
        success = run_single_experiment(args.experiment)
        sys.exit(0 if success else 1)
    else:
        # Run all implemented experiments
        run_all_experiments()


if __name__ == "__main__":
    main()