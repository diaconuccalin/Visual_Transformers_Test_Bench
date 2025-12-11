#!/bin/bash
# Setup script for Visual Wake Words Dataset (MLPerf Tiny compatible)
#
# This script downloads COCO 2014 and generates Visual Wake Words annotations
# for use with MLPerf benchmarking.

set -e

# Configuration
DATA_DIR="${1:-./data}"
# Convert to absolute path
mkdir -p "${DATA_DIR}"
DATA_DIR="$(cd "${DATA_DIR}" && pwd)"
COCO_DIR="${DATA_DIR}/coco2014"
VWW_DIR="${COCO_DIR}/annotations/vww"
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

# Download COCO 2014 dataset manually
echo ""
echo "Step 1: Downloading COCO 2014 dataset..."
echo "This will download ~40 GB of data (images + annotations)"
echo ""

# Download training images
if [ "$(ls -A ${COCO_DIR}/train2014 2>/dev/null | head -n 1)" != "" ]; then
    echo "Training images already exist (found in ${COCO_DIR}/train2014), skipping download..."
elif [ -f "${COCO_DIR}/train2014.zip" ]; then
    echo "Found existing train2014.zip, extracting..."
    unzip -q "${COCO_DIR}/train2014.zip" -d "${COCO_DIR}"
    rm "${COCO_DIR}/train2014.zip"
else
    echo "Downloading COCO 2014 training images (~13 GB)..."
    wget -P "${COCO_DIR}" http://images.cocodataset.org/zips/train2014.zip
    echo "Extracting training images..."
    unzip -q "${COCO_DIR}/train2014.zip" -d "${COCO_DIR}"
    rm "${COCO_DIR}/train2014.zip"
fi

# Download validation images
if [ "$(ls -A ${COCO_DIR}/val2014 2>/dev/null | head -n 1)" != "" ]; then
    echo "Validation images already exist (found in ${COCO_DIR}/val2014), skipping download..."
elif [ -f "${COCO_DIR}/val2014.zip" ]; then
    echo "Found existing val2014.zip, extracting..."
    unzip -q "${COCO_DIR}/val2014.zip" -d "${COCO_DIR}"
    rm "${COCO_DIR}/val2014.zip"
else
    echo "Downloading COCO 2014 validation images (~6 GB)..."
    wget -P "${COCO_DIR}" http://images.cocodataset.org/zips/val2014.zip
    echo "Extracting validation images..."
    unzip -q "${COCO_DIR}/val2014.zip" -d "${COCO_DIR}"
    rm "${COCO_DIR}/val2014.zip"
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

# Check if VWW annotations already exist
echo ""
echo "Step 2: Checking for existing Visual Wake Words annotations..."
mkdir -p "${VWW_DIR}"

if [ -f "${VWW_DIR}/instances_train.json" ] && [ -f "${VWW_DIR}/instances_val.json" ]; then
    echo "VWW annotations already exist, skipping annotation generation..."
else
    echo "VWW annotations not found, generating them now..."

    # Clone visualwakewords repository for scripts
    if [ ! -d "${DATA_DIR}/visualwakewords_repo" ]; then
        echo ""
        echo "Cloning visualwakewords repository for annotation scripts..."
        git clone https://github.com/diaconuccalin/visualwakewords.git "${DATA_DIR}/visualwakewords_repo"
    fi

    cd "${DATA_DIR}/visualwakewords_repo"

    # Create train/minival split
    echo ""
    echo "Creating train/minival split..."
    if [ ! -f "${COCO_DIR}/annotations/instances_maxitrain.json" ]; then
        python scripts/create_coco_train_minival_split.py \
            --train_annotations_file="${COCO_DIR}/annotations/instances_train${YEAR}.json" \
            --val_annotations_file="${COCO_DIR}/annotations/instances_val${YEAR}.json" \
            --output_dir="${COCO_DIR}/annotations"
    else
        echo "Train/minival split already exists, skipping..."
    fi

    # Create 'all' directory with symlinks to all images (required by pyvww)
    echo ""
    echo "Creating 'all' directory with symlinks to all images..."
    mkdir -p "${COCO_DIR}/all"
    if [ "$(ls -A ${COCO_DIR}/all 2>/dev/null | head -n 1)" = "" ]; then
        echo "Creating symlinks to train images in all/..."
        for img in "${COCO_DIR}/train2014/"*; do
            [ -f "$img" ] && ln -sf "$img" "${COCO_DIR}/all/$(basename "$img")"
        done
        echo "Creating symlinks to val images in all/..."
        for img in "${COCO_DIR}/val2014/"*; do
            [ -f "$img" ] && ln -sf "$img" "${COCO_DIR}/all/$(basename "$img")"
        done
        echo "Symlinks created successfully."
    else
        echo "'all' directory already populated, skipping..."
    fi

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

    # Cleanup: Remove cloned repository
    echo ""
    echo "Step 4: Cleaning up..."
    if [ -d "${DATA_DIR}/visualwakewords_repo" ]; then
        echo "Removing visualwakewords repository (no longer needed)..."
        rm -rf "${DATA_DIR}/visualwakewords_repo"
        echo "Cleanup complete."
    else
        echo "Repository directory not found, skipping cleanup..."
    fi
fi

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Your COCO 2014 dataset with Visual Wake Words annotations is ready!"
echo ""
echo "Directory structure:"
echo "  ${COCO_DIR}/all/                    - All COCO images (symlinked)"
echo "  ${COCO_DIR}/train2014/              - Training images"
echo "  ${COCO_DIR}/val2014/                - Validation images"
echo "  ${COCO_DIR}/annotations/            - COCO annotations"
echo "  ${COCO_DIR}/annotations/vww/        - VWW annotations"
echo ""
echo "To use it with the benchmark script, run:"
echo ""
echo "python benchmark.py --model mobilenet_v1 --dataset visual_wake_words \\"
echo "  --vww-root ${COCO_DIR}/all \\"
echo "  --vww-ann ${VWW_DIR}/instances_val.json"
echo ""
echo "For training set, use:"
echo "python benchmark.py --model mobilenet_v1 --dataset visual_wake_words \\"
echo "  --vww-root ${COCO_DIR}/all \\"
echo "  --vww-ann ${VWW_DIR}/instances_train.json"
echo ""
echo "Dataset statistics:"
echo "- Training images: ~82,783"
echo "- Validation images: ~40,504"
echo ""
