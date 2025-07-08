# AFL-TFL Framework: Approximate Forgiveness Level Tables for Neural Network Pruning

## Overview

This project establishes **falsifiable evaluation standards** for neural network pruning research by creating reference tables of Approximate Forgiveness Levels (AFL) across common dataset-architecture combinations.

## The Problem: Lack of Falsifiability in Pruning Research

Currently, the neural network pruning field operates without proper statistical controls. Researchers claim superior performance without demonstrating that their methods outperform appropriate baselines. This is equivalent to clinical trials without placebo controls - scientifically invalid.

**Example of Current Practice:**
- Paper A: "Our method achieves 85% accuracy at 90% sparsity on CIFAR-10"
- Paper B: "Our method achieves 87% accuracy at 90% sparsity on ImageNet" 
- Conclusion: "Method B is superior"

This comparison is meaningless without knowing the baseline performance of random pruning on each dataset-architecture pair.

## Theoretical Foundation

### True Forgiveness Level (TFL)

The **True Forgiveness Level (TFL)** is the theoretical sparsity threshold above which random pruning causes statistically significant performance degradation. Mathematically:

```
TFL = sup{s ∈ [0,1] : E[Accuracy(RandomPrune(s))] ≥ E[Accuracy(Baseline)] - ε}
```

Where:
- `s` = sparsity level (fraction of parameters removed)
- `ε` = acceptable performance degradation threshold
- `RandomPrune(s)` = uniform random pruning at sparsity `s`

**Why TFL is Unattainable:**
1. **Infinite precision required**: TFL exists at infinitesimal sparsity increments
2. **Infinite trials needed**: True expectation requires infinite sampling
3. **Computational impossibility**: Exhaustive testing is intractable

### Approximate Forgiveness Level (AFL)

The **Approximate Forgiveness Level (AFL)** is the empirically measurable threshold where random pruning first shows statistically significant degradation. It represents our best discrete approximation to the continuous TFL.

**AFL Determination Protocol:**
1. Test random pruning at discrete sparsity levels: {10%, 20%, ..., 90%, 95%}
2. Run ≥30 independent trials per sparsity level
3. Apply statistical significance testing (Wilcoxon signed-rank, p < 0.05)
4. AFL = first sparsity level showing significant degradation

**Mathematical Relationship:**
```
AFL ≈ TFL + δ
```
Where `δ` represents discretization error bounded by our sampling granularity.

### The Student's t-Table Analogy

Just as **Student (William Gosset)** created discrete tables for the continuous t-distribution, we create discrete AFL tables for the continuous forgiveness landscape:

**Student's t-Distribution:**
- **Continuous reality**: t-distribution exists at all real-valued degrees of freedom
- **Discrete approximation**: t-tables provided values at integer degrees of freedom
- **Practical utility**: Enabled statistical inference despite discretization

**AFL Tables:**
- **Continuous reality**: TFL exists at infinitesimal sparsity increments  
- **Discrete approximation**: AFL tables provide values at tested sparsity levels
- **Practical utility**: Enable falsifiable pruning evaluation despite discretization

## Falsifiability Framework

### The Bonsai Criterion

For a pruning method to claim scientific validity, it must satisfy:

1. **Trial Rigor**: ≥30 independent runs with randomized seeds
2. **Comparative Baseline**: Performance compared against random pruning at same sparsity
3. **AFL Awareness**: Evaluation conducted above the established AFL threshold
4. **Statistical Significance**: Demonstrate significant improvement over random baseline (p < 0.05)

### Why AFL-Based Evaluation is Necessary

**Without AFL:**
- Cannot distinguish method effectiveness from dataset forgiveness
- Results are scientifically unfalsifiable  
- Field operates on untested assumptions

**With AFL:**
- Clear null hypothesis: method performs no better than random pruning
- Falsifiable claims: can test and potentially refute method superiority
- Scientific rigor: proper statistical controls and significance testing

## AFL Table Structure

Each AFL entry specifies:

```yaml
Dataset: CIFAR-10
Architecture: ResNet-18  
Parameters: ~11M
Training: SGD, lr=0.1, 200 epochs
AFL: 75% ± 3%
Confidence: 95%
Trials: 30 per sparsity level
```

**Critical Dependencies:**
- **Dataset**: Different datasets have vastly different forgiveness
- **Architecture**: Model structure affects redundancy patterns  
- **Size**: Parameter count influences sparsity interpretation
- **Training**: Optimization affects learned representations

## Statistical Methodology

### Hypothesis Testing Framework

For each sparsity level `s`:

**Null Hypothesis (H₀):** Random pruning at sparsity `s` maintains baseline performance  
**Alternative Hypothesis (H₁):** Random pruning at sparsity `s` degrades performance

**Test Procedure:**
1. Baseline accuracy: `μ₀ = E[Accuracy(UnprunedModel)]`
2. Random pruning trials: `{a₁, a₂, ..., a₃₀} = Accuracy(RandomPrune(s))`
3. Statistical test: Wilcoxon signed-rank test comparing samples to `μ₀`
4. Decision: Reject H₀ if p < 0.05

**AFL Determination:** First sparsity level where H₀ is rejected.

### Multiple Comparisons Correction

When testing multiple sparsity levels, apply Bonferroni correction:
```
α_corrected = α / k
```
Where `k` = number of sparsity levels tested.

## Project Structure

```
/mnt/data/afl/          # Experimental data and configurations
/mnt/projects/afl/      # Implementation code  
/mnt/artifacts/afl/     # Final reference tables and publications
```

## Usage Example

**Before AFL Framework:**
```python
# Scientifically invalid
accuracy = prune_with_method(model, sparsity=0.9)
print(f"Achieved {accuracy}% at 90% sparsity")  # Meaningless without baseline
```

**After AFL Framework:**
```python
# Scientifically valid
afl = get_afl("CIFAR-10", "ResNet-18")  # afl = 0.75
if sparsity > afl:
    accuracy = prune_with_method(model, sparsity)
    random_baseline = get_random_pruning_performance("CIFAR-10", "ResNet-18", sparsity)
    p_value = statistical_test(accuracy, random_baseline)
    
    if p_value < 0.05:
        print(f"Method significantly outperforms random pruning (p={p_value})")
    else:
        print(f"Method shows no significant improvement over random pruning")
else:
    print(f"Cannot evaluate: sparsity {sparsity} below AFL {afl}")
```

## Scientific Impact

This framework introduces **falsifiability** to pruning research, transforming it from optimization theater into genuine science. Key contributions:

1. **Establishes null hypotheses** for pruning method evaluation
2. **Provides reference standards** for statistical comparison  
3. **Enables method ranking** through controlled comparison
4. **Forces rigorous evaluation** of existing methods

## Expected Outcomes

**Optimistic Scenario:** Many existing pruning methods prove genuinely superior to random baselines when properly tested.

**Realistic Scenario:** Some popular methods show no significant improvement over random pruning, requiring field-wide reevaluation.

**Either Way:** Science advances through truth rather than assumption.

## Citation

```bibtex
@misc{afl_framework_2025,
  title={AFL-TFL Framework: Approximate Forgiveness Level Tables for Falsifiable Neural Network Pruning Evaluation},
  author={[Author Name]},
  year={2025},
  note={Available at: [Repository URL]}
}
```

## Acknowledgments

This work builds on principles established in biostatistics and evidence-based medicine, applying rigorous statistical methodology to machine learning evaluation.

---

*"In science, there is only physics; all the rest is stamp collecting."* - Ernest Rutherford

*"In machine learning, there is only statistics; all the rest is parameter tuning."* - AFL Framework Philosophy