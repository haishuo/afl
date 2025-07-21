#!/usr/bin/env python3
"""
AFL Results Visualizer
======================

Creates comprehensive visualizations for AFL experiment results.
Generates plots comparing different datasets and showing pruning degradation curves.

Usage:
    python visualize_afl_results.py [--experiment EXPERIMENT_NAME] [--compare]
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import argparse

# Style configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_experiment_results(experiment_name: str) -> Dict:
    """Load results from AFL experiment JSON file."""
    results_path = Path(f"/mnt/artifacts/afl/reports/experiment_reports/{experiment_name}_report.json")
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")
    
    with open(results_path, 'r') as f:
        return json.load(f)

def extract_accuracy_curve(results: Dict) -> Tuple[List[float], List[float], List[float]]:
    """Extract sparsity levels and corresponding accuracies from results."""
    sparsity_levels = []
    mean_accuracies = []
    std_accuracies = []
    
    for sparsity_str, data in results['sparsity_results'].items():
        sparsity = float(sparsity_str)
        sparsity_levels.append(sparsity * 100)  # Convert to percentage
        mean_accuracies.append(data['mean_accuracy'])
        std_accuracies.append(data['std_accuracy'])
    
    # Sort by sparsity level
    sorted_indices = np.argsort(sparsity_levels)
    sparsity_levels = [sparsity_levels[i] for i in sorted_indices]
    mean_accuracies = [mean_accuracies[i] for i in sorted_indices]
    std_accuracies = [std_accuracies[i] for i in sorted_indices]
    
    return sparsity_levels, mean_accuracies, std_accuracies

def plot_single_experiment(results: Dict, save_path: Optional[Path] = None):
    """Create comprehensive visualization for a single AFL experiment."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"AFL Analysis: {results['dataset']} - {results['architecture']}", fontsize=16)
    
    # Extract data
    sparsity_levels, mean_accuracies, std_accuracies = extract_accuracy_curve(results)
    baseline = results['baseline_accuracy']
    afl_value = results['afl_value'] * 100  # Convert to percentage
    
    # 1. Accuracy vs Sparsity Curve
    ax1 = axes[0, 0]
    ax1.plot(sparsity_levels, mean_accuracies, 'o-', linewidth=2, markersize=8, label='Mean Accuracy')
    ax1.fill_between(sparsity_levels, 
                     np.array(mean_accuracies) - np.array(std_accuracies),
                     np.array(mean_accuracies) + np.array(std_accuracies),
                     alpha=0.3, label='±1 std')
    ax1.axhline(y=baseline, color='green', linestyle='--', label=f'Baseline: {baseline:.1f}%')
    ax1.axvline(x=afl_value, color='red', linestyle='--', label=f'AFL: {afl_value:.0f}%')
    ax1.set_xlabel('Sparsity (%)')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Accuracy Degradation Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Retention Rate Plot
    ax2 = axes[0, 1]
    retention_rates = [acc/baseline * 100 for acc in mean_accuracies]
    ax2.plot(sparsity_levels, retention_rates, 's-', linewidth=2, markersize=8, color='orange')
    ax2.axhline(y=100, color='green', linestyle='--', label='100% Retention')
    ax2.axhline(y=95, color='yellow', linestyle='--', label='95% Retention')
    ax2.axvline(x=afl_value, color='red', linestyle='--', label=f'AFL: {afl_value:.0f}%')
    ax2.set_xlabel('Sparsity (%)')
    ax2.set_ylabel('Accuracy Retention (%)')
    ax2.set_title('Performance Retention vs Baseline')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Statistical Significance Heatmap
    ax3 = axes[1, 0]
    p_values = []
    cohens_d = []
    sparsity_labels = []
    
    for sparsity_str, data in sorted(results['sparsity_results'].items()):
        sparsity_labels.append(f"{float(sparsity_str)*100:.0f}%")
        p_values.append(data.get('p_value', 1.0))
        cohens_d.append(abs(data.get('cohens_d', 0)))
    
    # Create significance matrix
    sig_data = np.array([p_values, cohens_d])
    im = ax3.imshow(sig_data, aspect='auto', cmap='RdYlGn_r')
    ax3.set_xticks(range(len(sparsity_labels)))
    ax3.set_xticklabels(sparsity_labels)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['p-value', "Cohen's d"])
    ax3.set_title('Statistical Significance Across Sparsity Levels')
    
    # Add text annotations
    for i in range(len(sparsity_labels)):
        ax3.text(i, 0, f'{p_values[i]:.4f}', ha='center', va='center')
        ax3.text(i, 1, f'{cohens_d[i]:.2f}', ha='center', va='center')
    
    # 4. Summary Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    Experiment Summary
    ==================
    
    Dataset: {results['dataset']}
    Architecture: {results['architecture']}
    Parameters: {results['parameter_count']:,}
    
    Baseline Accuracy: {baseline:.2f}%
    AFL Value: {afl_value:.0f}%
    
    Recommendation: 
    {results['recommendation']}
    
    Duration: {results['experiment_duration']/60:.1f} minutes
    
    Key Findings:
    • Meaningful degradation at {afl_value:.0f}% sparsity
    • Accuracy drops to {mean_accuracies[1]:.1f}% at AFL
    • {retention_rates[1]:.1f}% performance retained at AFL
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()

def plot_comparison(experiment_names: List[str], save_path: Optional[Path] = None):
    """Create comparison plots for multiple AFL experiments."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('AFL Experiment Comparison', fontsize=18)
    
    # Load all results
    all_results = {}
    for exp_name in experiment_names:
        all_results[exp_name] = load_experiment_results(exp_name)
    
    # 1. Accuracy Curves Comparison
    ax1 = axes[0, 0]
    for exp_name, results in all_results.items():
        sparsity_levels, mean_accuracies, _ = extract_accuracy_curve(results)
        label = f"{results['dataset']} ({results['baseline_accuracy']:.1f}% baseline)"
        ax1.plot(sparsity_levels, mean_accuracies, 'o-', linewidth=2, markersize=6, label=label)
        
        # Mark AFL point
        afl_value = results['afl_value'] * 100
        afl_idx = sparsity_levels.index(afl_value)
        ax1.plot(afl_value, mean_accuracies[afl_idx], 'r*', markersize=15)
    
    ax1.set_xlabel('Sparsity (%)')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Accuracy Degradation Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. AFL Values Comparison
    ax2 = axes[0, 1]
    datasets = []
    afl_values = []
    colors = []
    
    for exp_name, results in all_results.items():
        datasets.append(results['dataset'])
        afl_values.append(results['afl_value'] * 100)
        
        # Color based on AFL value
        if results['afl_value'] <= 0.3:
            colors.append('red')
        elif results['afl_value'] <= 0.5:
            colors.append('orange')
        else:
            colors.append('green')
    
    bars = ax2.bar(datasets, afl_values, color=colors, edgecolor='black', linewidth=2)
    ax2.set_ylabel('AFL Value (%)')
    ax2.set_title('AFL Values Across Datasets')
    ax2.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, value in zip(bars, afl_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    # 3. Retention at 50% Sparsity
    ax3 = axes[1, 0]
    datasets = []
    retentions_50 = []
    
    for exp_name, results in all_results.items():
        datasets.append(results['dataset'])
        sparsity_results = results['sparsity_results']
        if '0.5' in sparsity_results:
            retention = sparsity_results['0.5']['mean_accuracy'] / results['baseline_accuracy'] * 100
            retentions_50.append(retention)
        else:
            retentions_50.append(0)
    
    ax3.bar(datasets, retentions_50, color='skyblue', edgecolor='black', linewidth=2)
    ax3.axhline(y=95, color='red', linestyle='--', label='95% retention threshold')
    ax3.set_ylabel('Accuracy Retention (%)')
    ax3.set_title('Performance Retention at 50% Sparsity')
    ax3.legend()
    
    # 4. Summary Table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create summary data
    summary_data = []
    for exp_name, results in all_results.items():
        summary_data.append([
            results['dataset'],
            f"{results['baseline_accuracy']:.1f}%",
            f"{results['afl_value']*100:.0f}%",
            f"{results['parameter_count']:,}",
            results['recommendation'].split('(')[0].strip()
        ])
    
    table = ax4.table(cellText=summary_data,
                     colLabels=['Dataset', 'Baseline', 'AFL', 'Parameters', 'Category'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(summary_data) + 1):
        for j in range(5):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to: {save_path}")
    else:
        plt.show()

def create_afl_spectrum_plot(save_path: Optional[Path] = None):
    """Create a visual spectrum of AFL values with interpretations."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define AFL spectrum
    afl_ranges = [
        (0, 20, 'HIGHLY_DISCRIMINATING', 'red', 'Extremely sensitive to pruning'),
        (20, 40, 'CHALLENGING', 'orange', 'Significant pruning resistance'), 
        (40, 60, 'MODERATE', 'yellow', 'Balanced pruning tolerance'),
        (60, 80, 'FORGIVING', 'lightgreen', 'High pruning tolerance'),
        (80, 100, 'EXTREMELY_FORGIVING', 'green', 'Minimal impact from pruning')
    ]
    
    # Draw spectrum bars
    for start, end, label, color, description in afl_ranges:
        ax.barh(0, end-start, left=start, height=0.5, color=color, 
                edgecolor='black', linewidth=2)
        ax.text((start+end)/2, 0, label.replace('_', '\n'), 
                ha='center', va='center', fontweight='bold', fontsize=10)
        ax.text((start+end)/2, -0.35, description, 
                ha='center', va='center', fontsize=8, style='italic')
    
    # Add example datasets
    examples = [
        ('Wine Quality', 20, -0.7),
        ('MNIST', 70, -0.7),
        ('CIFAR-10*', 45, -0.7),
        ('ImageNet*', 30, -0.7)
    ]
    
    for dataset, afl, y_pos in examples:
        ax.plot(afl, y_pos, 'ko', markersize=10)
        ax.text(afl, y_pos-0.1, dataset, ha='center', va='top', fontsize=9)
        ax.vlines(afl, -0.5, y_pos, colors='black', linestyles='dashed', alpha=0.5)
    
    ax.set_xlim(-5, 105)
    ax.set_ylim(-1.2, 0.8)
    ax.set_xlabel('AFL Value (%)', fontsize=12)
    ax.set_title('AFL (Approximate Forgiveness Level) Spectrum\n'
                 'Lower AFL = More discriminating benchmark for pruning methods',
                 fontsize=14)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add note
    ax.text(50, 0.6, '*Hypothetical values for illustration', 
            ha='center', style='italic', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved AFL spectrum plot to: {save_path}")
    else:
        plt.show()

def main():
    """Main function to run visualizations."""
    parser = argparse.ArgumentParser(description="Visualize AFL experiment results")
    parser.add_argument("--experiment", type=str, help="Specific experiment to visualize")
    parser.add_argument("--compare", action="store_true", help="Compare all available experiments")
    parser.add_argument("--spectrum", action="store_true", help="Create AFL spectrum visualization")
    parser.add_argument("--output", type=str, help="Output directory for plots", 
                       default="/mnt/artifacts/afl/visualizations/plots")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.spectrum:
        save_path = output_dir / "afl_spectrum.png"
        create_afl_spectrum_plot(save_path)
    
    elif args.compare:
        # Find all available experiment results
        reports_dir = Path("/mnt/artifacts/afl/reports/experiment_reports")
        available_experiments = []
        
        for report_file in reports_dir.glob("*_report.json"):
            exp_name = report_file.stem.replace("_report", "")
            available_experiments.append(exp_name)
        
        if len(available_experiments) >= 2:
            print(f"Found {len(available_experiments)} experiments to compare")
            save_path = output_dir / "afl_comparison.png"
            plot_comparison(available_experiments, save_path)
        else:
            print("Need at least 2 experiments to compare")
    
    elif args.experiment:
        save_path = output_dir / f"{args.experiment}_analysis.png"
        results = load_experiment_results(args.experiment)
        plot_single_experiment(results, save_path)
    
    else:
        # Default: visualize all available experiments
        reports_dir = Path("/mnt/artifacts/afl/reports/experiment_reports")
        
        for report_file in reports_dir.glob("*_report.json"):
            exp_name = report_file.stem.replace("_report", "")
            print(f"Visualizing {exp_name}...")
            
            save_path = output_dir / f"{exp_name}_analysis.png"
            results = load_experiment_results(exp_name)
            plot_single_experiment(results, save_path)

if __name__ == "__main__":
    main()