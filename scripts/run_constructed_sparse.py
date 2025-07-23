#!/usr/bin/env python3
"""
Run Constructed Sparse Experiments
==================================

Location: /mnt/projects/afl/scripts/run_constructed_sparse.py

Runner script for constructed sparse network experiments.
These experiments validate the AFL framework by comparing against
networks built at target sparsity from initialization.
"""

import sys
import argparse
import logging
from pathlib import Path
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Available experiments
EXPERIMENTS = {
    "mnist": {
        "name": "MNIST Constructed Sparse MLP",
        "script": "/mnt/projects/afl/experiments/mnist/constructed_sparse_experiment.py",
        "description": "MNIST with MLP 256->128->64 built at various sparsities"
    },
    "wine_quality": {
        "name": "Wine Quality Constructed Sparse MLP", 
        "script": "/mnt/projects/afl/experiments/wine_quality/constructed_sparse_experiment.py",
        "description": "Wine Quality with MLP 256->128->64 built at various sparsities"
    }
}


def list_experiments():
    """List all available constructed sparse experiments."""
    logger.info("📋 Available Constructed Sparse Experiments:")
    logger.info("=" * 60)
    
    for key, exp in EXPERIMENTS.items():
        logger.info(f"\n{key}:")
        logger.info(f"  Name: {exp['name']}")
        logger.info(f"  Description: {exp['description']}")
        logger.info(f"  Script: {exp['script']}")
        
        # Check if script exists
        if Path(exp['script']).exists():
            logger.info(f"  Status: ✅ Ready")
        else:
            logger.info(f"  Status: ❌ Not implemented")


def run_experiment(experiment_key: str):
    """Run a specific constructed sparse experiment."""
    if experiment_key not in EXPERIMENTS:
        logger.error(f"Unknown experiment: {experiment_key}")
        logger.error(f"Available experiments: {list(EXPERIMENTS.keys())}")
        return 1
    
    exp = EXPERIMENTS[experiment_key]
    script_path = Path(exp['script'])
    
    if not script_path.exists():
        logger.error(f"Experiment script not found: {script_path}")
        logger.error(f"Please implement the experiment first.")
        return 1
    
    logger.info(f"🚀 Running: {exp['name']}")
    logger.info(f"Script: {script_path}")
    logger.info("=" * 60)
    
    # Run the experiment
    try:
        result = subprocess.run([sys.executable, str(script_path)], 
                              capture_output=False, text=True)
        return result.returncode
    except Exception as e:
        logger.error(f"Error running experiment: {e}")
        return 1


def run_comparison(dataset: str):
    """Run comparison analysis after experiments complete."""
    logger.info(f"\n📊 Running comparison analysis for {dataset}...")
    
    comparison_script = Path("/mnt/projects/afl/scripts/analysis/compare_pruning_vs_constructed.py")
    
    if not comparison_script.exists():
        logger.error(f"Comparison script not found: {comparison_script}")
        return 1
    
    try:
        result = subprocess.run([
            sys.executable, 
            str(comparison_script),
            "--dataset", dataset
        ], capture_output=False, text=True)
        return result.returncode
    except Exception as e:
        logger.error(f"Error running comparison: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run constructed sparse network experiments to validate AFL framework"
    )
    parser.add_argument(
        "action",
        choices=["list", "run", "compare", "all"],
        help="Action to perform"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Specific experiment to run (for 'run' action)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset for comparison (for 'compare' action)"
    )
    
    args = parser.parse_args()
    
    if args.action == "list":
        list_experiments()
        
    elif args.action == "run":
        if not args.experiment:
            logger.error("Please specify --experiment for run action")
            return 1
        return run_experiment(args.experiment)
        
    elif args.action == "compare":
        if not args.dataset:
            logger.error("Please specify --dataset for compare action")
            return 1
        return run_comparison(args.dataset)
        
    elif args.action == "all":
        # Run all implemented experiments and comparisons
        logger.info("🚀 Running all constructed sparse experiments...")
        
        for key, exp in EXPERIMENTS.items():
            if Path(exp['script']).exists():
                logger.info(f"\n{'='*60}")
                logger.info(f"Running {key}...")
                logger.info(f"{'='*60}")
                
                ret = run_experiment(key)
                if ret != 0:
                    logger.error(f"Failed to run {key}")
                    continue
                
                # Run comparison if experiment succeeded
                logger.info(f"\nRunning comparison for {key}...")
                run_comparison(key)
        
        logger.info("\n✅ All experiments complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())