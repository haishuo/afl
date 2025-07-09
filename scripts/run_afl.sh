#!/bin/bash

# AFL Experiment Runner Script
# Location: /mnt/projects/afl/scripts/run_afl.sh
# 
# Simple wrapper script for running AFL experiments

# Change to scripts directory
cd "$(dirname "$0")"

echo "🧪 AFL Experiment Runner"
echo "========================"

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ python3 not found. Please ensure Python 3 is installed."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "run_afl_experiment.py" ]; then
    echo "❌ run_afl_experiment.py not found. Are you in the right directory?"
    echo "Expected location: /mnt/projects/afl/scripts/"
    exit 1
fi

# Parse command line arguments
case "${1:-}" in
    "list"|"--list"|"-l")
        echo "📋 Listing available experiments..."
        python3 run_afl_experiment.py --list
        ;;
    "status"|"--status"|"-s")
        echo "📊 Showing experiment status..."
        python3 run_afl_experiment.py --status
        ;;
    "help"|"--help"|"-h")
        echo "Usage: $0 [COMMAND|EXPERIMENT_NAME]"
        echo ""
        echo "Commands:"
        echo "  list, -l, --list       List available experiments"
        echo "  status, -s, --status   Show experiment status"
        echo "  help, -h, --help       Show this help message"
        echo ""
        echo "Experiments:"
        echo "  mnist_mlp_256_128_64   Run MNIST MLP experiment"
        echo "  (no argument)          Run all implemented experiments"
        echo ""
        echo "Examples:"
        echo "  $0                     # Run all experiments"
        echo "  $0 list                # List available experiments"
        echo "  $0 mnist_mlp_256_128_64 # Run specific experiment"
        ;;
    "")
        echo "🚀 Running all implemented AFL experiments..."
        python3 run_afl_experiment.py
        ;;
    *)
        echo "🎯 Running specific experiment: $1"
        python3 run_afl_experiment.py --experiment "$1"
        ;;
esac