# Noise Robustness Ablation Study for LLMAT

## Overview

This ablation study evaluates the robustness of LLMAT to label noise on toy benchmarks. The study injects different types and scales of noise into the objective function labels to understand how the algorithm performs under noisy observations.

## Motivation

In real-world Bayesian Optimization scenarios, observed labels often contain noise due to:
- Measurement errors in experimental setups
- Stochastic processes in the objective function
- Approximation errors in simulators
- Human annotation errors

Understanding the robustness of LLMAT to such noise is crucial for practical applications.

## Study Design

### Noise Types

Three types of noise are evaluated:

1. **Gaussian Noise** (Additive)
   - Formula: `y_noisy = y + N(0, σ * std(y))`
   - Most common noise model in practice
   - Represents measurement errors with normal distribution

2. **Uniform Noise** (Additive)
   - Formula: `y_noisy = y + U(-σ * std(y), +σ * std(y))`
   - Bounded noise range
   - Represents systematic errors with known bounds

3. **Multiplicative Noise**
   - Formula: `y_noisy = y * (1 + N(0, σ))`
   - Noise proportional to signal magnitude
   - Common in physical systems where errors scale with measurement

### Noise Scales

The following noise scales (σ) are tested:
- **0.0**: Baseline (no noise)
- **0.05**: 5% noise (low noise)
- **0.1**: 10% noise (moderate noise)
- **0.2**: 20% noise (high noise)
- **0.5**: 50% noise (very high noise)
- **1.0**: 100% noise (extreme noise)

The noise scale is relative to the standard deviation of the target values.

### Test Functions

Two toy functions are evaluated:
- **Levy Function**: Complex multimodal landscape (10D)
- **Ackley Function**: Challenging optimization benchmark (10D)

### Experimental Setup

- **Number of seeds**: 5 independent runs per configuration
- **BO iterations**: 100
- **Initial data points**: 10
- **Dataset size**: 1000 samples per function

## File Structure

```
LLMAT/
├── benchmarks/TOY/
│   ├── toy_benchmarks.py          # Original toy benchmark
│   └── toy_benchmarks_noisy.py    # Extended with noise injection
├── configs/ablation/
│   └── noise_ablation.yaml        # Ablation study configuration
├── src/
│   ├── main/
│   │   └── run_noise_ablation.py  # Main ablation runner
│   └── utils/
│       └── visualize_ablation.py  # Visualization utilities
└── docs/
    └── noise_ablation_study.md    # This file
```

## Usage

### 1. Running the Full Ablation Study

Run all configurations specified in the config file:

```bash
cd <repo_root>

python src/main/run_noise_ablation.py \
    --config configs/ablation/noise_ablation.yaml
```

Results will be saved to `results/ablation/noise_YYYYMMDD_HHMMSS/`

### 2. Running a Single Test (for debugging)

Test with a specific configuration:

```bash
python src/main/run_noise_ablation.py \
    --config configs/ablation/noise_ablation.yaml \
    --single \
    --function levy \
    --noise_type gaussian \
    --noise_scale 0.1 \
    --seed 42
```

### 3. Customizing the Configuration

Edit `configs/ablation/noise_ablation.yaml` to modify:

```yaml
ablation:
  functions:
    - levy
    - ackley

  noise_types:
    - gaussian
    - uniform
    - multiplicative

  noise_scales:
    - 0.0
    - 0.1
    - 0.2
    - 0.5

  n_seeds: 5
  seeds: [42, 123, 456, 789, 1024]

  dims: [10]
  n_samples: 1000
```

### 4. Visualizing Results

Generate all visualizations from completed ablation study:

```bash
python src/utils/visualize_ablation.py \
    results/ablation/noise_YYYYMMDD_HHMMSS/
```

This creates:
- `noise_robustness_gap.png`: Performance vs noise scale
- `noise_robustness_best_y.png`: Best objective vs noise scale
- `noise_type_comparison.png`: Box plots comparing noise types
- `convergence_curves.png`: BO convergence for different noise levels
- `summary_table.csv`: Quantitative results table

## Output Files

### Individual Results

Each run produces a pickle file:
```
{function}_{noise_type}_{noise_scale}_seed{seed}.pkl
```

Contents:
- `function`: Test function name
- `noise_type`: Type of noise
- `noise_scale`: Scale of noise
- `seed`: Random seed
- `final_best_y`: Best objective found
- `ground_truth_opt`: True optimum
- `final_gap`: Gap to optimum
- `trace_y`: Trajectory of best values over BO iterations
- `metrics`: Additional performance metrics

### Aggregated Results

- `all_results.csv`: All runs in tabular format
- `summary_statistics.csv`: Mean ± std grouped by configuration
- `config.yaml`: Copy of the configuration used

## Key Metrics

1. **Final Gap**: `final_gap = best_y - ground_truth_opt`
   - Lower is better
   - Measures optimization performance

2. **Final Best Y**: `final_best_y`
   - Higher is better (for maximization)
   - Direct measure of solution quality

3. **Signal-to-Noise Ratio (SNR)**: `SNR = std(y_clean) / std(noise)`
   - Printed during data generation
   - Characterizes noise level

## Expected Outcomes

### Hypotheses

1. **Noise Scale Effect**: Performance should degrade as noise scale increases
2. **Noise Type Effect**: Different noise types may affect performance differently
3. **Tree Robustness**: MCTS tree structure may provide some robustness to noise through aggregation
4. **UCT Selection**: UCT values (v_bar, v_var) should naturally account for noise variance

### Analysis Questions

- At what noise level does performance significantly degrade?
- Which noise type is most challenging for LLMAT?
- How does noise affect tree construction (depth, split quality)?
- Does the hierarchical partition help with noise robustness?

## Implementation Details

### NoisyToyBench Class

Key features:
- Inherits from `ToyBench`
- Injects noise during data generation
- Preserves noise seed for reproducibility
- Adds noise info to dataset name (prevents cache collision)

### Noise Injection

```python
def _inject_noise(self, targets):
    targets = np.array(targets)
    target_std = np.std(targets)

    if self.noise_type == "gaussian":
        noise = np.random.normal(0, self.noise_scale * target_std, len(targets))
        noisy_targets = targets + noise

    # ... (similar for uniform and multiplicative)

    return noisy_targets
```

### Reproducibility

- Data generation seed: `randseed` parameter
- Noise injection seed: `noise_seed` parameter (default: `randseed`)
- BO algorithm seed: `seed` parameter

All seeds are saved with results for exact reproduction.

## Tips and Best Practices

### For Running Experiments

1. **Start with a single run** (`--single` flag) to verify setup
2. **Use smaller configurations** for initial testing (fewer seeds/scales)
3. **Monitor disk space** - results can grow large with many configurations
4. **Check intermediate results** - inspect individual pickle files during long runs

### For Analysis

1. **Look at convergence curves** first to understand dynamics
2. **Compare noise types** at the same scale
3. **Identify noise scale threshold** where performance drops significantly
4. **Check variance across seeds** to assess stability

### For Debugging

1. **Check SNR values** printed during data generation
2. **Verify noise injection** by comparing `targets` with clean function values
3. **Inspect tree construction** - noise may affect split quality
4. **Review UCT values** - variance should increase with noise

## Future Extensions

Possible extensions to this ablation study:

1. **Heterogeneous Noise**: Different noise levels for different regions
2. **Outliers**: Add rare but large errors (e.g., heavy-tailed noise)
3. **Input Noise**: Add noise to features, not just labels
4. **Noise Estimation**: Implement GP with noise variance estimation
5. **Adaptive Methods**: Adjust hyperparameters based on estimated noise
6. **Compare Baselines**: Include GP-UCB, Random Search, etc.
7. **Higher Dimensions**: Test on 20D, 50D problems
8. **Real Benchmarks**: Apply to material science benchmarks with synthetic noise

## References

- LLMAT paper (if published)
- MCTS-based Bayesian Optimization literature
- Noisy optimization benchmarks

## Contact

For questions or issues, please contact:
- qichen@cs.toronto.edu
- Open an issue on the GitHub repository
