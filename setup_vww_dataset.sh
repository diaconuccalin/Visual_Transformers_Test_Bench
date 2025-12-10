#!/bin/bash
# Setup script for Visual Wake Words Dataset (MLPerf Tiny compatible)
#
# This script downloads COCO 2014 and generates Visual Wake Words annotations
# for use with MLPerf benchmarking.

set -e

# Configuration
DATA_DIR="${1:-./data}"
COCO_DIR="${DATA_DIR}/coco2014"
VWW_DIR="${DATA_DIR}/vww"
YEAR="2014"

echo "========================================="
echo "Visual Wake Words Dataset Setup"
echo "========================================="
echo "Data directory: ${DATA_DIR}"
echo "COCO directory: ${COCO_DIR}"
echo "VWW directory: ${VWW_DIR}"
echo ""

# Check if pyvww is installed
if ! python -c "import pyvww" 2>/dev/null; then
    echo "Installing pyvww library..."
    pip install pyvww
fi

# Create directories
mkdir -p "${COCO_DIR}"
mkdir -p "${VWW_DIR}"

# Clone visualwakewords repository for scripts
if [ ! -d "${DATA_DIR}/visualwakewords_repo" ]; then
    echo "Cloning visualwakewords repository..."
    git clone https://github.com/Mxbonn/visualwakewords.git "${DATA_DIR}/visualwakewords_repo"
fi

cd "${DATA_DIR}/visualwakewords_repo"

# Download COCO 2014 dataset
echo ""
echo "Step 1: Downloading COCO 2014 dataset..."
echo "This will download ~40 GB of data (images + annotations)"
echo ""
bash scripts/download_mscoco.sh "${COCO_DIR}" "${YEAR}"

# Create train/minival split
echo ""
echo "Step 2: Creating train/minival split..."
python scripts/create_coco_train_minival_split.py \
    --train_annotations_file="${COCO_DIR}/annotations/instances_train${YEAR}.json" \
    --val_annotations_file="${COCO_DIR}/annotations/instances_val${YEAR}.json" \
    --output_dir="${COCO_DIR}/annotations"

# Generate Visual Wake Words annotations
echo ""
echo "Step 3: Generating Visual Wake Words annotations..."
python scripts/create_visualwakewords_annotations.py \
    --train_annotations_file="${COCO_DIR}/annotations/instances_maxitrain.json" \
    --val_annotations_file="${COCO_DIR}/annotations/instances_minival.json" \
    --output_dir="${VWW_DIR}" \
    --threshold=0.005 \
    --foreground_class='person'

cd -

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Your Visual Wake Words dataset is ready at: ${VWW_DIR}"
echo ""
echo "To use it with the benchmark script, run:"
echo ""
echo "python benchmark.py --model mobilenet_v2 --dataset visual_wake_words \\"
echo "  --vww-root ${COCO_DIR}/all \\"
echo "  --vww-ann ${VWW_DIR}/instances_train.json"
echo ""
echo "Dataset statistics:"
echo "- Training images: ~82,783"
echo "- Validation images: ~40,504"
echo "- Test images: Same as validation (minival split)"
echo ""
