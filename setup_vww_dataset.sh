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
mkdir -p "${COCO_DIR}/annotations"
mkdir -p "${COCO_DIR}/train2014"
mkdir -p "${COCO_DIR}/val2014"
mkdir -p "${VWW_DIR}"

# Download COCO 2014 dataset manually
echo ""
echo "Step 1: Downloading COCO 2014 dataset..."
echo "This will download ~40 GB of data (images + annotations)"
echo ""

# Download training images
if [ ! -f "${COCO_DIR}/train2014.zip" ] && [ "$(ls -A ${COCO_DIR}/train2014 2>/dev/null)" = "" ]; then
    echo "Downloading COCO 2014 training images (~13 GB)..."
    wget -P "${COCO_DIR}" http://images.cocodataset.org/zips/train2014.zip
    echo "Extracting training images..."
    unzip -q "${COCO_DIR}/train2014.zip" -d "${COCO_DIR}"
    rm "${COCO_DIR}/train2014.zip"
else
    echo "Training images already exist, skipping download..."
fi

# Download validation images
if [ ! -f "${COCO_DIR}/val2014.zip" ] && [ "$(ls -A ${COCO_DIR}/val2014 2>/dev/null)" = "" ]; then
    echo "Downloading COCO 2014 validation images (~6 GB)..."
    wget -P "${COCO_DIR}" http://images.cocodataset.org/zips/val2014.zip
    echo "Extracting validation images..."
    unzip -q "${COCO_DIR}/val2014.zip" -d "${COCO_DIR}"
    rm "${COCO_DIR}/val2014.zip"
else
    echo "Validation images already exist, skipping download..."
fi

# Download annotations
if [ ! -f "${COCO_DIR}/annotations/instances_train2014.json" ]; then
    echo "Downloading COCO 2014 annotations..."
    wget -P "${COCO_DIR}" http://images.cocodataset.org/annotations/annotations_trainval2014.zip
    echo "Extracting annotations..."
    unzip -q "${COCO_DIR}/annotations_trainval2014.zip" -d "${COCO_DIR}"
    rm "${COCO_DIR}/annotations_trainval2014.zip"
else
    echo "Annotations already exist, skipping download..."
fi

# Clone visualwakewords repository for scripts
if [ ! -d "${DATA_DIR}/visualwakewords_repo" ]; then
    echo ""
    echo "Cloning visualwakewords repository for annotation scripts..."
    git clone https://github.com/Mxbonn/visualwakewords.git "${DATA_DIR}/visualwakewords_repo"
fi

cd "${DATA_DIR}/visualwakewords_repo"

# Create train/minival split
echo ""
echo "Step 2: Creating train/minival split..."
if [ ! -f "${COCO_DIR}/annotations/instances_maxitrain.json" ]; then
    python scripts/create_coco_train_minival_split.py \
        --train_annotations_file="${COCO_DIR}/annotations/instances_train${YEAR}.json" \
        --val_annotations_file="${COCO_DIR}/annotations/instances_val${YEAR}.json" \
        --output_dir="${COCO_DIR}/annotations"
else
    echo "Train/minival split already exists, skipping..."
fi

# Create 'all' directory with all images (required by pyvww)
echo ""
echo "Creating 'all' directory with all images..."
mkdir -p "${COCO_DIR}/all"
if [ "$(ls -A ${COCO_DIR}/all 2>/dev/null)" = "" ]; then
    echo "Copying train images to all/..."
    cp -r "${COCO_DIR}/train2014/"* "${COCO_DIR}/all/" 2>/dev/null || true
    echo "Copying val images to all/..."
    cp -r "${COCO_DIR}/val2014/"* "${COCO_DIR}/all/" 2>/dev/null || true
else
    echo "'all' directory already populated, skipping..."
fi

# Generate Visual Wake Words annotations
echo ""
echo "Step 3: Generating Visual Wake Words annotations..."
if [ ! -f "${VWW_DIR}/instances_train.json" ]; then
    python scripts/create_visualwakewords_annotations.py \
        --train_annotations_file="${COCO_DIR}/annotations/instances_maxitrain.json" \
        --val_annotations_file="${COCO_DIR}/annotations/instances_minival.json" \
        --output_dir="${VWW_DIR}" \
        --threshold=0.005 \
        --foreground_class='person'
else
    echo "VWW annotations already exist, skipping..."
fi

cd -

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Your Visual Wake Words dataset is ready at: ${VWW_DIR}"
echo ""
echo "Directory structure:"
echo "  ${COCO_DIR}/all/          - All COCO images (train + val)"
echo "  ${COCO_DIR}/annotations/  - COCO annotations"
echo "  ${VWW_DIR}/                - VWW annotations"
echo ""
echo "To use it with the benchmark script, run:"
echo ""
echo "python benchmark.py --model mobilenet_v2 --dataset visual_wake_words \\"
echo "  --vww-root ${COCO_DIR}/all \\"
echo "  --vww-ann ${VWW_DIR}/instances_train.json"
echo ""
echo "For validation/test set, use:"
echo "python benchmark.py --model mobilenet_v2 --dataset visual_wake_words \\"
echo "  --vww-root ${COCO_DIR}/all \\"
echo "  --vww-ann ${VWW_DIR}/instances_val.json"
echo ""
echo "Dataset statistics:"
echo "- Training images: ~82,783"
echo "- Validation images: ~40,504"
echo ""
