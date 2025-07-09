"""
AFL Framework Core Implementation
================================

Location: /mnt/projects/afl/core/afl_framework.py

Core AFL determination framework for neural network pruning evaluation.
Implements the statistical methodology for determining Approximate Forgiveness Levels.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import time
from dataclasses import dataclass
from scipy import stats

@dataclass
class AFLConfig:
    """Configuration for AFL determination experiments."""
    sparsity_levels: List[float] = None
    trials_per_level: int = 30
    confidence_level: float = 0.95
    fine_tune_epochs: int = 5
    random_seed_base: int = 42
    
    def __post_init__(self):
        if self.sparsity_levels is None:
            self.sparsity_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

@dataclass 
class AFLResult:
    """Results from AFL determination experiment."""
    dataset: str
    architecture: str
    parameter_count: int
    baseline_accuracy: float
    afl_value: Optional[float]
    sparsity_results: Dict[float, Dict]
    experiment_duration: float
    recommendation: str

class AFLFramework:
    """Core AFL determination framework."""
    
    def __init__(self, config: AFLConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def determine_afl(self, 
                     model: nn.Module,
                     train_loader,
                     test_loader,
                     dataset_name: str,
                     architecture_name: str) -> AFLResult:
        """
        Determine AFL for given model-dataset combination.
        
        Args:
            model: Trained neural network model
            train_loader: Training data loader
            test_loader: Test data loader  
            dataset_name: Name of dataset (for tracking)
            architecture_name: Name of architecture (for tracking)
            
        Returns:
            AFLResult containing complete experimental results
        """
        self.logger.info(f"Starting AFL determination for {dataset_name} + {architecture_name}")
        
        # Measure baseline performance
        baseline_accuracy = self._evaluate_model(model, test_loader)
        parameter_count = self._count_parameters(model)
        
        self.logger.info(f"Baseline accuracy: {baseline_accuracy:.2f}%")
        self.logger.info(f"Model parameters: {parameter_count:,}")
        
        # Test each sparsity level
        sparsity_results = {}
        afl_value = None
        
        for sparsity in self.config.sparsity_levels:
            self.logger.info(f"Testing sparsity level: {sparsity:.0%}")
            
            # Run multiple trials at this sparsity level
            trial_results = self._run_sparsity_trials(
                model, train_loader, test_loader, sparsity
            )
            
            # Statistical analysis
            sparsity_stats = self._analyze_sparsity_results(
                trial_results, baseline_accuracy
            )
            
            sparsity_results[sparsity] = sparsity_stats
            
            # Check if this is the AFL (first meaningful degradation)
            if afl_value is None and sparsity_stats['is_meaningfully_worse']:
                afl_value = sparsity
                self.logger.info(f"AFL determined: {afl_value:.0%} (first meaningful degradation)")
                if sparsity_stats['p_value'] is not None:
                    self.logger.info(f"  Statistical: p={sparsity_stats['p_value']:.4f}, "
                                   f"Practical: Cohen's d={sparsity_stats['cohens_d']:.3f}")
        
        # Generate recommendation
        recommendation = self._generate_recommendation(sparsity_results, afl_value)
        
        return AFLResult(
            dataset=dataset_name,
            architecture=architecture_name,
            parameter_count=parameter_count,
            baseline_accuracy=baseline_accuracy,
            afl_value=afl_value,
            sparsity_results=sparsity_results,
            experiment_duration=0.0,  # Will be updated by caller
            recommendation=recommendation
        )
    
    def _run_sparsity_trials(self, model, train_loader, test_loader, sparsity) -> List[float]:
        """Run multiple random pruning trials at given sparsity level."""
        trial_accuracies = []
        
        for trial in range(self.config.trials_per_level):
            trial_seed = self.config.random_seed_base + trial * 1000 + int(sparsity * 10000)
            
            try:
                # Create model copy
                model_copy = self._copy_model(model)
                
                # Apply random pruning
                pruned_model = self._apply_random_pruning(model_copy, sparsity, trial_seed)
                
                # Fine-tune if specified
                if self.config.fine_tune_epochs > 0:
                    pruned_model = self._fine_tune_model(
                        pruned_model, train_loader, self.config.fine_tune_epochs
                    )
                
                # Evaluate final performance
                accuracy = self._evaluate_model(pruned_model, test_loader)
                trial_accuracies.append(accuracy)
                
                # Progress logging
                if (trial + 1) % 10 == 0:
                    avg_acc = np.mean(trial_accuracies)
                    self.logger.info(f"  Completed {trial+1}/{self.config.trials_per_level} trials, "
                                   f"avg accuracy: {avg_acc:.2f}%")
                
            except Exception as e:
                self.logger.warning(f"Trial {trial} failed: {str(e)}")
                continue
        
        return trial_accuracies
    
    def _analyze_sparsity_results(self, trial_accuracies: List[float], 
                                 baseline_accuracy: float) -> Dict:
        """Perform statistical analysis of sparsity trial results."""
        if not trial_accuracies:
            return {
                'successful_trials': 0,
                'mean_accuracy': 0.0,
                'std_accuracy': 0.0,
                'mean_retention': 0.0,
                'is_significantly_worse': True,
                'is_practically_significant': True,
                'is_meaningfully_worse': True,
                'cohens_d': float('inf'),
                'p_value': None,
                'statistical_test': 'failed'
            }
        
        # Basic statistics
        mean_acc = np.mean(trial_accuracies)
        std_acc = np.std(trial_accuracies)
        mean_retention = mean_acc / baseline_accuracy
        
        # Calculate Cohen's d for effect size
        if len(trial_accuracies) > 1 and std_acc > 0:
            cohens_d = (baseline_accuracy - mean_acc) / std_acc
        else:
            cohens_d = 0.0
        
        # Determine practical significance (Cohen's d > 0.5 is "medium" effect)
        is_practically_significant = cohens_d > 0.5
        
        # Statistical significance test using Wilcoxon signed-rank test
        baseline_array = np.full(len(trial_accuracies), baseline_accuracy)
        
        try:
            if len(trial_accuracies) < 6:
                # Fall back to paired t-test for small samples
                t_stat, p_value = stats.ttest_rel(trial_accuracies, baseline_array)
                # One-tailed test: are pruned accuracies significantly LESS than baseline?
                p_value = p_value / 2 if t_stat < 0 else 1.0
                statistical_test = 'paired_t_test'
            else:
                # Use Wilcoxon signed-rank test (paired, one-tailed)
                w_stat, p_value = stats.wilcoxon(trial_accuracies, baseline_array, 
                                               alternative='less')
                statistical_test = 'wilcoxon_signed_rank'
            
            # Apply Bonferroni correction for multiple comparisons
            alpha = 1 - self.config.confidence_level
            corrected_alpha = alpha / len(self.config.sparsity_levels)
            
            # Statistical significance
            is_significantly_worse = (p_value < corrected_alpha)
            
        except Exception as e:
            self.logger.warning(f"Statistical test failed: {str(e)}")
            # Fallback: use practical significance only
            is_significantly_worse = mean_retention < 0.95
            p_value = None
            statistical_test = 'fallback_conservative'
            corrected_alpha = None
        
        # Combined significance: both statistical AND practical
        is_meaningfully_worse = is_significantly_worse and is_practically_significant
        
        return {
            'successful_trials': len(trial_accuracies),
            'mean_accuracy': float(mean_acc),
            'std_accuracy': float(std_acc),
            'mean_retention': float(mean_retention),
            'is_significantly_worse': bool(is_significantly_worse),
            'is_practically_significant': bool(is_practically_significant),
            'is_meaningfully_worse': bool(is_meaningfully_worse),
            'cohens_d': float(cohens_d),
            'p_value': float(p_value) if p_value is not None and not np.isnan(p_value) else None,
            'statistical_test': statistical_test,
            'corrected_alpha': float(corrected_alpha) if corrected_alpha is not None else None,
            'baseline_accuracy': float(baseline_accuracy),
            'trial_accuracies': [float(x) for x in trial_accuracies]
        }
    
    def _generate_recommendation(self, sparsity_results: Dict, afl_value: Optional[float]) -> str:
        """Generate dataset-architecture assessment based on AFL value."""
        if afl_value is None:
            max_tested = max(self.config.sparsity_levels)
            return f"VERY_FORGIVING (AFL > {max_tested:.0%})"
        elif afl_value >= 0.8:
            return f"FORGIVING (AFL = {afl_value:.0%})"
        elif afl_value >= 0.6:
            return f"MODERATELY_FORGIVING (AFL = {afl_value:.0%})"
        elif afl_value >= 0.4:
            return f"MODERATELY_DISCRIMINATING (AFL = {afl_value:.0%})"
        else:
            return f"HIGHLY_DISCRIMINATING (AFL = {afl_value:.0%})"
    
    # Abstract methods - must be implemented by subclasses
    def _evaluate_model(self, model: nn.Module, test_loader) -> float:
        """Evaluate model accuracy on test set."""
        raise NotImplementedError("Subclasses must implement model evaluation")
    
    def _count_parameters(self, model: nn.Module) -> int:
        """Count total trainable parameters in model."""
        raise NotImplementedError("Subclasses must implement parameter counting")
    
    def _copy_model(self, model: nn.Module) -> nn.Module:
        """Create deep copy of model."""
        raise NotImplementedError("Subclasses must implement model copying")
    
    def _apply_random_pruning(self, model: nn.Module, sparsity: float, seed: int) -> nn.Module:
        """Apply random pruning at specified sparsity level."""
        raise NotImplementedError("Subclasses must implement random pruning")
    
    def _fine_tune_model(self, model: nn.Module, train_loader, epochs: int) -> nn.Module:
        """Fine-tune pruned model."""
        raise NotImplementedError("Subclasses must implement fine-tuning")