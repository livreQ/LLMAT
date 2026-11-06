#!/bin/bash
# Run baseline BO comparison on toy benchmarks

echo "Running Baseline BO Comparison on Toy Benchmarks"
echo "================================================="
echo ""

# Default settings
CONFIG="configs/ablation/noise_ablation.yaml"
RESULTS_DIR="results/baselines_toy"
LLMAT_DIR="results/ablation"
COMPARISON_DIR="results/comparisons"

# Parse command line arguments
BASELINE="all"  # Default: run all baselines
while [[ $# -gt 0 ]]; do
    case $1 in
        --baseline)
            BASELINE="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --results_dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --baseline BASELINE    Which baseline to run: gp, laplace, or all (default: all)"
            echo "  --config CONFIG        Path to config file (default: configs/ablation/noise_ablation.yaml)"
            echo "  --results_dir DIR      Directory to save results (default: results/baselines_toy)"
            echo "  --help                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run all baselines"
            echo "  $0 --baseline gp                      # Run only GP baseline"
            echo "  $0 --baseline laplace                 # Run only Laplace baseline"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Config file: $CONFIG"
echo "  Baseline: $BASELINE"
echo "  Results directory: $RESULTS_DIR"
echo ""

# Run baseline experiments
echo "Step 1: Running baseline BO experiments..."
echo "==========================================="
python src/main/run_baseline_toy.py \
    --config "$CONFIG" \
    --baseline "$BASELINE" \
    --results_dir "$RESULTS_DIR"

if [ $? -ne 0 ]; then
    echo "Error: Baseline experiments failed"
    exit 1
fi

echo ""
echo "Step 2: Generating comparison visualizations..."
echo "==============================================="

# Check if LLMAT results exist
if [ ! -d "$LLMAT_DIR" ]; then
    echo "Warning: LLMAT results not found at $LLMAT_DIR"
    echo "Skipping comparison plots. Run the ablation study first with:"
    echo "  python src/main/run_noise_ablation.py --config $CONFIG"
    exit 0
fi

# Generate comparison plots
python src/utils/visualize_baseline_comparison.py \
    --baseline_dir "$RESULTS_DIR" \
    --llmat_dir "$LLMAT_DIR" \
    --output_dir "$COMPARISON_DIR"

if [ $? -ne 0 ]; then
    echo "Error: Visualization failed"
    exit 1
fi

echo ""
echo "================================================="
echo "Baseline comparison completed successfully!"
echo "Results saved to: $RESULTS_DIR"
echo "Comparison plots saved to: $COMPARISON_DIR"
echo "================================================="
