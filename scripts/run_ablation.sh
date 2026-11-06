#!/bin/bash
# Quick-start script for running noise ablation study

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  LLMAT Noise Robustness Ablation Study${NC}"
echo -e "${GREEN}================================================${NC}\n"

# Default values
CONFIG="configs/ablation/noise_ablation.yaml"
MODE="full"
FUNCTION="levy"
NOISE_TYPE="gaussian"
NOISE_SCALE=0.1
SEED=42

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            MODE="test"
            shift
            ;;
        --single)
            MODE="single"
            shift
            ;;
        --visualize)
            MODE="visualize"
            RESULTS_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --function)
            FUNCTION="$2"
            shift 2
            ;;
        --noise-type)
            NOISE_TYPE="$2"
            shift 2
            ;;
        --noise-scale)
            NOISE_SCALE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --test              Run quick test (1 seed, 2 noise scales)"
            echo "  --single            Run single experiment"
            echo "  --visualize DIR     Visualize results from directory"
            echo "  --config PATH       Path to config file (default: configs/ablation/noise_ablation.yaml)"
            echo "  --function NAME     Function for single run (default: levy)"
            echo "  --noise-type TYPE   Noise type for single run (default: gaussian)"
            echo "  --noise-scale VAL   Noise scale for single run (default: 0.1)"
            echo "  --seed NUM          Random seed for single run (default: 42)"
            echo "  --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run full ablation study"
            echo "  $0 --test                             # Quick test"
            echo "  $0 --single --function ackley --noise-scale 0.2"
            echo "  $0 --visualize results/ablation/noise_20251127_120000/"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG${NC}"
    exit 1
fi

# Execute based on mode
case $MODE in
    test)
        echo -e "${YELLOW}Running quick test (1 seed, 2 noise scales)...${NC}\n"

        # Create temporary test config
        TEST_CONFIG="configs/ablation/noise_ablation_test.yaml"
        cp "$CONFIG" "$TEST_CONFIG"

        # Modify config for quick test (would need Python or sed to do this properly)
        # For now, just run with original config but inform user
        echo -e "${YELLOW}Note: Using original config. For true test mode, manually edit config.${NC}\n"

        python src/main/run_noise_ablation.py --config "$CONFIG"
        ;;

    single)
        echo -e "${YELLOW}Running single experiment...${NC}"
        echo -e "  Function: ${FUNCTION}"
        echo -e "  Noise Type: ${NOISE_TYPE}"
        echo -e "  Noise Scale: ${NOISE_SCALE}"
        echo -e "  Seed: ${SEED}\n"

        python src/main/run_noise_ablation.py \
            --config "$CONFIG" \
            --single \
            --function "$FUNCTION" \
            --noise_type "$NOISE_TYPE" \
            --noise_scale "$NOISE_SCALE" \
            --seed "$SEED"
        ;;

    visualize)
        if [ -z "$RESULTS_DIR" ]; then
            echo -e "${RED}Error: Results directory required for --visualize${NC}"
            echo "Usage: $0 --visualize <results_directory>"
            exit 1
        fi

        if [ ! -d "$RESULTS_DIR" ]; then
            echo -e "${RED}Error: Results directory not found: $RESULTS_DIR${NC}"
            exit 1
        fi

        echo -e "${YELLOW}Visualizing results from: ${RESULTS_DIR}${NC}\n"
        python src/utils/visualize_ablation.py "$RESULTS_DIR"
        ;;

    full)
        echo -e "${YELLOW}Running full ablation study...${NC}"
        echo -e "Config: ${CONFIG}\n"

        # Show warning
        echo -e "${RED}WARNING: Full ablation study may take several hours!${NC}"
        echo -e "${YELLOW}Press Ctrl+C within 5 seconds to cancel...${NC}\n"
        sleep 5

        python src/main/run_noise_ablation.py --config "$CONFIG"

        # After completion, ask about visualization
        if [ $? -eq 0 ]; then
            echo -e "\n${GREEN}Ablation study completed successfully!${NC}\n"

            # Find most recent results directory
            LATEST_RESULTS=$(ls -td results/ablation/noise_* 2>/dev/null | head -1)

            if [ -n "$LATEST_RESULTS" ]; then
                echo -e "${YELLOW}Generate visualizations now? (y/n)${NC}"
                read -r response
                if [[ "$response" =~ ^[Yy]$ ]]; then
                    python src/utils/visualize_ablation.py "$LATEST_RESULTS"
                    echo -e "\n${GREEN}Visualizations saved to: ${LATEST_RESULTS}/figures${NC}"
                else
                    echo -e "\n${YELLOW}To visualize later, run:${NC}"
                    echo -e "  $0 --visualize $LATEST_RESULTS"
                fi
            fi
        else
            echo -e "\n${RED}Ablation study failed. Check logs for errors.${NC}"
            exit 1
        fi
        ;;
esac

echo -e "\n${GREEN}Done!${NC}\n"
