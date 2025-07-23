#!/usr/bin/env python3
"""
Run Constructed Sparse Experiments
==================================

Location: /mnt/projects/afl/scripts/run_constructed_sparse.py

Simple runner script for constructed sparse network experiments.
"""

import sys
import argparse
import logging
from pathlib import Path
import subprocess
import os

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


def check_partial_results(experiment_key: str) -> bool:
    """Check if partial results exist for an experiment."""
    # Simplified - just check if the experiment has been run before
    return False


def run_experiment(experiment_key: str, use_tmux: bool = True, max_parallel: int = 8):
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
    logger.info(f"Max parallel trials: {max_parallel}")
    logger.info("=" * 60)
    
    # Set environment variable for parallel execution
    env = os.environ.copy()
    env['AFL_MAX_PARALLEL'] = str(max_parallel)
    
    # Run the experiment
    try:
        if use_tmux and sys.stdin.isatty():
            # Use tmux for interactive sessions
            tmux_session = f"afl_{experiment_key}"
            logger.info(f"Starting in tmux session: {tmux_session}")
            
            # Create tmux session and run experiment
            commands = [
                f"tmux kill-session -t {tmux_session} 2>/dev/null || true",
                f"tmux new-session -d -s {tmux_session}",
                f"tmux send-keys -t {tmux_session} 'cd /mnt/projects/afl' C-m",
                f"tmux send-keys -t {tmux_session} 'export AFL_MAX_PARALLEL={max_parallel}' C-m",
                f"tmux send-keys -t {tmux_session} '{sys.executable} {script_path}' C-m",
            ]
            
            # Add GPU monitoring in split pane if using parallel
            if max_parallel > 1:
                commands.extend([
                    f"tmux split-window -t {tmux_session} -h -p 30",
                    f"tmux send-keys -t {tmux_session} 'watch -n 1 nvidia-smi' C-m",
                    f"tmux select-pane -t {tmux_session} -L"
                ])
            
            for cmd in commands:
                subprocess.run(cmd, shell=True)
            
            logger.info(f"\n📺 Experiment running in tmux session: {tmux_session}")
            logger.info(f"   View with: tmux attach -t {tmux_session}")
            logger.info(f"   Detach with: Ctrl+B, then D")
            
            # Ask if user wants to attach
            if input("\nAttach to tmux session now? [Y/n]: ").lower() != 'n':
                subprocess.run(f"tmux attach -t {tmux_session}", shell=True)
            
            return 0
        else:
            # Run directly
            result = subprocess.run([sys.executable, str(script_path)], 
                                  env=env, capture_output=False, text=True)
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
        choices=["list", "run", "compare", "all", "status"],
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
    parser.add_argument(
        "--no-tmux",
        action="store_true",
        help="Run without tmux (not recommended for long experiments)"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=8,
        help="Maximum parallel trials (default: 8)"
    )
    
    args = parser.parse_args()
    
    if args.action == "list":
        list_experiments()
        
    elif args.action == "status":
        # Check status of all experiments
        logger.info("📊 Experiment Status:")
        logger.info("=" * 60)
        
        for key, exp in EXPERIMENTS.items():
            logger.info(f"\n{key}:")
            
            # Check if complete results exist
            complete_path = Path(f"/mnt/artifacts/afl/reports/experiment_reports/{key}_constructed_sparse_256_128_64_report.json")
            if complete_path.exists():
                logger.info(f"  Status: ✅ Complete")
                logger.info(f"  Results: {complete_path}")
            elif Path(exp['script']).exists():
                logger.info(f"  Status: 🔄 Ready to run")
            else:
                logger.info(f"  Status: ❌ Not implemented")
        
    elif args.action == "run":
        if not args.experiment:
            logger.error("Please specify --experiment for run action")
            return 1
        return run_experiment(args.experiment, use_tmux=not args.no_tmux, 
                            max_parallel=args.max_parallel)
        
    elif args.action == "compare":
        if not args.dataset:
            logger.error("Please specify --dataset for compare action")
            return 1
        return run_comparison(args.dataset)
        
    elif args.action == "all":
        # Run all implemented experiments
        logger.info("🚀 Running all constructed sparse experiments...")
        logger.info(f"Max parallel trials: {args.max_parallel}")
        
        for key, exp in EXPERIMENTS.items():
            if Path(exp['script']).exists():
                # Check if already complete
                complete_path = Path(f"/mnt/artifacts/afl/reports/experiment_reports/{key}_constructed_sparse_256_128_64_report.json")
                if complete_path.exists():
                    logger.info(f"\n✅ {key} already complete, skipping...")
                    continue
                
                logger.info(f"\n{'='*60}")
                logger.info(f"Running {key}...")
                logger.info(f"{'='*60}")
                
                ret = run_experiment(key, use_tmux=not args.no_tmux, 
                                   max_parallel=args.max_parallel)
                if ret != 0:
                    logger.error(f"Failed to run {key}")
                    continue
                
                if args.no_tmux:
                    # Run comparison immediately if not using tmux
                    logger.info(f"\nRunning comparison for {key}...")
                    run_comparison(key)
        
        if not args.no_tmux:
            logger.info("\n📺 All experiments started in tmux sessions")
            logger.info("Check running sessions: tmux ls")
            logger.info("Monitor GPU usage: watch -n 1 nvidia-smi")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())