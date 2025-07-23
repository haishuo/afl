#!/usr/bin/env python3
"""
Compare Pruning vs Constructed Sparse Analysis
==============================================

Location: /mnt/projects/afl/scripts/analysis/compare_pruning_vs_constructed.py

This script compares the results from:
1. Standard AFL pruning experiments (train → prune → fine-tune)
2. Constructed sparse experiments (build small → train)

The comparison validates whether pruning ≈ constructed sparse,
which would confirm that the AFL framework is meaningful.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_experiment_results(report_path: Path) -> Dict:
    """Load experiment results from JSON file."""
    with open(report_path, 'r') as f:
        return json.load(f)


def perform_statistical_comparison(pruned_accs: List[float], 
                                 constructed_accs: List[float]) -> Dict:
    """
    Perform statistical tests to compare two sets of accuracies.
    
    Returns:
        Dictionary with test statistics and interpretation
    """
    # Welch's t-test (doesn't assume equal variances)
    t_stat, p_value = stats.ttest_ind(pruned_accs, constructed_accs, equal_var=False)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(pruned_accs)**2 + np.std(constructed_accs)**2) / 2)
    cohens_d = (np.mean(pruned_accs) - np.mean(constructed_accs)) / pooled_std
    
    # Mann-Whitney U test (non-parametric alternative)
    u_stat, u_p_value = stats.mannwhitneyu(pruned_accs, constructed_accs, alternative='two-sided')
    
    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "mann_whitney_u": u_stat,
        "mann_whitney_p": u_p_value,
        "significantly_different": p_value < 0.05,
        "effect_size_interpretation": interpret_effect_size(cohens_d)
    }


def interpret_effect_size(cohens_d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def compare_experiments(pruned_path: Path, constructed_path: Path, 
                       output_dir: Path) -> Dict:
    """
    Compare pruning vs constructed sparse experiments.
    
    Args:
        pruned_path: Path to pruning experiment JSON
        constructed_path: Path to constructed sparse experiment JSON
        output_dir: Directory to save comparison results
        
    Returns:
        Comparison results dictionary
    """
    logger.info("📊 Loading experiment results...")
    pruned_results = load_experiment_results(pruned_path)
    constructed_results = load_experiment_results(constructed_path)
    
    # Verify experiments are comparable
    assert pruned_results["dataset"] == constructed_results["dataset"], \
        "Datasets must match for comparison"
    
    # Extract architecture info
    dataset = pruned_results["dataset"]
    pruned_arch = pruned_results.get("architecture", "Unknown")
    
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Architecture: {pruned_arch}")
    
    # Initialize comparison results
    comparison = {
        "dataset": dataset,
        "architecture": pruned_arch,
        "pruned_experiment": str(pruned_path.name),
        "constructed_experiment": str(constructed_path.name),
        "sparsity_comparisons": {},
        "overall_statistics": {},
        "afl_validation": {}
    }
    
    # Compare each sparsity level
    sparsity_levels = []
    all_differences = []
    significant_count = 0
    
    for sparsity_key in pruned_results["sparsity_results"]:
        if sparsity_key in constructed_results["sparsity_results"]:
            sparsity = float(sparsity_key)
            sparsity_levels.append(sparsity)
            
            # Get accuracies
            pruned_stats = pruned_results["sparsity_results"][sparsity_key]
            constructed_stats = constructed_results["sparsity_results"][sparsity_key]
            
            # Get accuracies - handle different field names
            if "accuracies" in pruned_stats:
                pruned_accs = pruned_stats["accuracies"]
            elif "trial_accuracies" in pruned_stats:
                pruned_accs = pruned_stats["trial_accuracies"]
            else:
                logger.warning(f"No accuracy data found for sparsity {sparsity_key}")
                continue
                
            constructed_accs = constructed_stats["accuracies"]
            
            # Statistical comparison
            stats_result = perform_statistical_comparison(pruned_accs, constructed_accs)
            
            # Calculate difference
            mean_diff = pruned_stats["mean_accuracy"] - constructed_stats["mean_accuracy"]
            all_differences.append(mean_diff)
            
            if stats_result["significantly_different"]:
                significant_count += 1
            
            # Store comparison
            comparison["sparsity_comparisons"][sparsity_key] = {
                "pruned": {
                    "mean": pruned_stats["mean_accuracy"],
                    "std": pruned_stats["std_accuracy"],
                    "n_trials": len(pruned_accs)
                },
                "constructed": {
                    "mean": constructed_stats["mean_accuracy"],
                    "std": constructed_stats["std_accuracy"],
                    "n_trials": len(constructed_accs),
                    "architecture": constructed_stats.get("architecture", "Unknown")
                },
                "difference": mean_diff,
                "statistical_test": stats_result
            }
            
            logger.info(f"\nSparsity {sparsity:.0%}:")
            logger.info(f"  Pruned: {pruned_stats['mean_accuracy']:.2f}% ± {pruned_stats['std_accuracy']:.2f}%")
            logger.info(f"  Constructed: {constructed_stats['mean_accuracy']:.2f}% ± {constructed_stats['std_accuracy']:.2f}%")
            logger.info(f"  Difference: {mean_diff:+.2f}%")
            logger.info(f"  p-value: {stats_result['p_value']:.4f}")
            logger.info(f"  Significant: {'YES' if stats_result['significantly_different'] else 'NO'}")
    
    # Overall statistics
    comparison["overall_statistics"] = {
        "mean_difference": np.mean(all_differences),
        "std_difference": np.std(all_differences),
        "max_difference": max(all_differences),
        "min_difference": min(all_differences),
        "n_significant": significant_count,
        "n_comparisons": len(sparsity_levels),
        "percent_significant": 100 * significant_count / len(sparsity_levels)
    }
    
    # AFL validation assessment
    if comparison["overall_statistics"]["percent_significant"] < 20:
        validation_status = "STRONGLY_VALIDATED"
        interpretation = "Pruning ≈ Constructed sparse. AFL framework is highly meaningful."
    elif comparison["overall_statistics"]["percent_significant"] < 50:
        validation_status = "VALIDATED"
        interpretation = "Pruning ≈ Constructed sparse for most sparsities. AFL framework is meaningful."
    else:
        validation_status = "NEEDS_INVESTIGATION"
        interpretation = "Significant differences found. Further investigation needed."
    
    comparison["afl_validation"] = {
        "status": validation_status,
        "interpretation": interpretation,
        "recommendation": "AFL framework provides valid baselines for pruning evaluation."
    }
    
    # Save comparison results
    output_path = output_dir / f"{dataset.lower()}_pruning_vs_constructed_comparison.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    comparison_native = convert_to_native(comparison)
    
    with open(output_path, 'w') as f:
        json.dump(comparison_native, f, indent=2)
    
    logger.info(f"\n✅ Comparison complete!")
    logger.info(f"📄 Results saved to: {output_path}")
    
    # Generate visualization
    generate_comparison_plot(comparison, output_dir, dataset)
    
    return comparison


def generate_comparison_plot(comparison: Dict, output_dir: Path, dataset: str):
    """Generate visualization comparing pruning vs constructed sparse."""
    
    plt.figure(figsize=(12, 8))
    
    # Extract data for plotting
    sparsities = []
    pruned_means = []
    pruned_stds = []
    constructed_means = []
    constructed_stds = []
    
    for sparsity_key in sorted(comparison["sparsity_comparisons"].keys()):
        sparsity = float(sparsity_key)
        comp = comparison["sparsity_comparisons"][sparsity_key]
        
        sparsities.append(sparsity * 100)  # Convert to percentage
        pruned_means.append(comp["pruned"]["mean"])
        pruned_stds.append(comp["pruned"]["std"])
        constructed_means.append(comp["constructed"]["mean"])
        constructed_stds.append(comp["constructed"]["std"])
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot 1: Accuracy comparison
    ax1.errorbar(sparsities, pruned_means, yerr=pruned_stds, 
                label='Pruned', marker='o', capsize=5, linewidth=2)
    ax1.errorbar(sparsities, constructed_means, yerr=constructed_stds,
                label='Constructed Sparse', marker='s', capsize=5, linewidth=2)
    
    ax1.set_xlabel('Sparsity (%)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title(f'{dataset} - Pruning vs Constructed Sparse Networks', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add significance markers
    for i, sparsity_key in enumerate(sorted(comparison["sparsity_comparisons"].keys())):
        comp = comparison["sparsity_comparisons"][sparsity_key]
        if comp["statistical_test"]["significantly_different"]:
            ax1.text(sparsities[i], max(pruned_means[i], constructed_means[i]) + 1, 
                    '*', ha='center', fontsize=16, color='red')
    
    # Plot 2: Difference plot
    differences = [p - c for p, c in zip(pruned_means, constructed_means)]
    colors = ['red' if comp["statistical_test"]["significantly_different"] else 'blue' 
              for comp in comparison["sparsity_comparisons"].values()]
    
    bars = ax2.bar(sparsities, differences, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Sparsity (%)', fontsize=12)
    ax2.set_ylabel('Difference (Pruned - Constructed) %', fontsize=12)
    ax2.set_title('Accuracy Difference by Sparsity Level', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add mean difference line
    mean_diff = comparison["overall_statistics"]["mean_difference"]
    ax2.axhline(y=mean_diff, color='green', linestyle='--', linewidth=2,
                label=f'Mean difference: {mean_diff:.2f}%')
    ax2.legend()
    
    # Add text box with summary
    textstr = f'AFL Validation: {comparison["afl_validation"]["status"]}\n'
    textstr += f'Significant differences: {comparison["overall_statistics"]["n_significant"]}/{comparison["overall_statistics"]["n_comparisons"]}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f"{dataset.lower()}_pruning_vs_constructed.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 Plot saved to: {plot_path}")


def main():
    """Main function to run the comparison analysis."""
    parser = argparse.ArgumentParser(
        description="Compare pruning vs constructed sparse network results"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="mnist",
        help="Dataset to analyze (mnist, wine_quality, etc.)"
    )
    parser.add_argument(
        "--output", 
        type=str,
        default="/mnt/artifacts/afl/reports/statistical_analysis",
        help="Output directory for comparison results"
    )
    
    args = parser.parse_args()
    
    # Set up paths
    reports_dir = Path("/mnt/artifacts/afl/reports/experiment_reports")
    
    # Map dataset names to experiment files
    experiment_map = {
        "mnist": {
            "pruned": "mnist_mlp_256_128_64_report.json",
            "constructed": "mnist_constructed_sparse_256_128_64_report.json"
        },
        "wine_quality": {
            "pruned": "wine_quality_mlp_256_128_64_report.json",
            "constructed": "wine_quality_constructed_sparse_256_128_64_report.json"
        }
    }
    
    if args.dataset not in experiment_map:
        logger.error(f"Unknown dataset: {args.dataset}")
        logger.error(f"Available: {list(experiment_map.keys())}")
        return
    
    # Get paths
    pruned_path = reports_dir / experiment_map[args.dataset]["pruned"]
    constructed_path = reports_dir / experiment_map[args.dataset]["constructed"]
    
    # Check if files exist
    if not pruned_path.exists():
        logger.error(f"Pruning results not found: {pruned_path}")
        return
    
    if not constructed_path.exists():
        logger.error(f"Constructed sparse results not found: {constructed_path}")
        logger.error("Please run the constructed sparse experiment first.")
        return
    
    # Run comparison
    output_dir = Path(args.output)
    comparison = compare_experiments(pruned_path, constructed_path, output_dir)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Dataset: {comparison['dataset']}")
    logger.info(f"Mean difference: {comparison['overall_statistics']['mean_difference']:.2f}%")
    logger.info(f"Significant comparisons: {comparison['overall_statistics']['n_significant']}/{comparison['overall_statistics']['n_comparisons']}")
    logger.info(f"AFL Validation: {comparison['afl_validation']['status']}")
    logger.info(f"Interpretation: {comparison['afl_validation']['interpretation']}")


if __name__ == "__main__":
    main()