#!/bin/bash
# =============================================================================
# Master script to generate Target Drift Figure (Figure 5)
# Compares SODA reservoir stability vs ERM batch instability
#
# This script:
# 1. Trains ERM on GPU 0 and SODA on GPU 1 in parallel
# 2. Logs target drift during training
# 3. Generates the comparison plot
#
# Usage:
#   chmod +x run_drift_experiment.sh
#   ./run_drift_experiment.sh
# =============================================================================

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$BASE_DIR")"

DATASET="OfficeHome"
TEST_ENV=0
NUM_ITERATIONS=5000
LOG_EVERY=10
OUTPUT_DIR="${REPO_DIR}/drift_results"
FIG_DIR="${REPO_DIR}/fig"

echo "=============================================="
echo "  Target Drift Experiment"
echo "=============================================="
echo "Script directory: $SCRIPT_DIR"
echo "Base directory: $BASE_DIR"
echo "Repo directory: $REPO_DIR"
echo "Dataset: $DATASET"
echo "Test environment: $TEST_ENV"
echo "Iterations: $NUM_ITERATIONS"
echo "Output directory: $OUTPUT_DIR"
echo "Figure directory: $FIG_DIR"
echo "=============================================="

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$FIG_DIR"

# Check for required packages
echo ""
echo "Checking dependencies..."
python -c "import geomloss; print('  ✓ geomloss installed')" || {
    echo "  ✗ geomloss not found. Installing..."
    pip install geomloss
}
python -c "import scipy; print('  ✓ scipy installed')" || {
    echo "  ✗ scipy not found. Installing..."
    pip install scipy
}
python -c "import matplotlib; print('  ✓ matplotlib installed')" || {
    echo "  ✗ matplotlib not found. Installing..."
    pip install matplotlib
}
python -c "import tqdm; print('  ✓ tqdm installed')" || {
    echo "  ✗ tqdm not found. Installing..."
    pip install tqdm
}

# Check GPUs
echo ""
echo "Checking GPUs..."
python -c "import torch; print(f'  Available GPUs: {torch.cuda.device_count()}')"
python -c "import torch; [print(f'    GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# Navigate to script directory
cd "$SCRIPT_DIR"

# Option 1: Run sequentially (safer, more verbose)
echo ""
echo "=============================================="
echo "  Phase 1: Training ERM (GPU 0)"
echo "=============================================="
CUDA_VISIBLE_DEVICES=0 python train_drift_comparison.py \
    --algorithm ERM \
    --dataset "$DATASET" \
    --test_env $TEST_ENV \
    --iterations $NUM_ITERATIONS \
    --log_every $LOG_EVERY \
    --output_dir "$OUTPUT_DIR" \
    --data_dir "$BASE_DIR/data" \
    --gpu 0

echo ""
echo "=============================================="
echo "  Phase 2: Training SODA (GPU 1)"
echo "=============================================="
CUDA_VISIBLE_DEVICES=1 python train_drift_comparison.py \
    --algorithm SODA \
    --dataset "$DATASET" \
    --test_env $TEST_ENV \
    --iterations $NUM_ITERATIONS \
    --log_every $LOG_EVERY \
    --output_dir "$OUTPUT_DIR" \
    --data_dir "$BASE_DIR/data" \
    --gpu 0

# Option 2: Run in parallel (uncomment to use)
# echo ""
# echo "=============================================="
# echo "  Training ERM and SODA in parallel"
# echo "=============================================="
# python train_drift_comparison.py \
#     --run_both \
#     --dataset "$DATASET" \
#     --test_env $TEST_ENV \
#     --iterations $NUM_ITERATIONS \
#     --log_every $LOG_EVERY \
#     --output_dir "$OUTPUT_DIR" \
#     --data_dir "$BASE_DIR/data"

# Generate plot
echo ""
echo "=============================================="
echo "  Phase 3: Generating Figure"
echo "=============================================="

DATASET_LOWER=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')

python plot_target_drift.py \
    --erm_file "$OUTPUT_DIR/drift_erm_${DATASET_LOWER}.json" \
    --soda_file "$OUTPUT_DIR/drift_soda_${DATASET_LOWER}.json" \
    --output "$FIG_DIR/target_drift.pdf" \
    --smooth 5.0

echo ""
echo "=============================================="
echo "  DONE!"
echo "=============================================="
echo "Results saved to:"
echo "  - ERM drift data:  $OUTPUT_DIR/drift_erm_${DATASET_LOWER}.json"
echo "  - SODA drift data: $OUTPUT_DIR/drift_soda_${DATASET_LOWER}.json"
echo "  - Figure (PDF):    $FIG_DIR/target_drift.pdf"
echo "  - Figure (PNG):    $FIG_DIR/target_drift.png"
echo "=============================================="
