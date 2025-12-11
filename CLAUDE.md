# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Visual Transformers Test Bench is a benchmarking framework for evaluating visual models on low-power devices with MLPerf Tiny compatibility. The architecture uses a modular pipeline: CLI → Model Loading → Dataset Loading → Evaluation → Metrics Reporting.

## Core Architecture

The codebase has four main components that work together:

1. **benchmark.py**: Main orchestrator that coordinates the evaluation pipeline
2. **utils/model_loader.py**: Multi-source model loading with automatic fallback (file checkpoints → torchvision → timm)
3. **utils/evaluation.py**: Model evaluation and comprehensive metrics calculation (binary and multi-class)
4. **datasets/visual_wake_words.py**: Dataset loader wrapping pyvww with PyTorch DataLoader interface
5. **datasets/imagenet.py**: ImageNet 1k dataset loader using torchvision ImageFolder

### Model Loading Strategy

The `load_model()` function implements a fallback chain:

**Priority 1 - File Checkpoints:**
- Detects checkpoint format (full model vs state_dict)
- Reconstructs architecture from metadata if state_dict-only
- Adapts classifier head for target num_classes

**Priority 2 - Torchvision:**
- Loads pretrained models (e.g., `resnet18`, `vit_b_16`)
- Automatically modifies output layers (handles `fc`, `classifier`, `head` naming patterns)

**Priority 3 - Timm (Fallback):**
- Uses `timm.create_model()` for extended model library

When modifying model loading, maintain this fallback chain and handle all three classifier head patterns.

### Dataset Architecture

Visual Wake Words uses pyvww library wrapped in PyTorch Dataset interface:

**Preprocessing Pipeline:**
- Training: Resize → RandomHorizontalFlip → ToTensor → Normalize(ImageNet stats)
- Val/Test: Resize → ToTensor → Normalize(ImageNet stats)

**Key Configuration:**
- ImageNet normalization hardcoded: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Binary classification: 0=no person, 1=person present
- MLPerf spec uses 96×96 images (default is 224×224)

When adding new datasets, follow the pattern in `datasets/visual_wake_words.py`: create a factory function that returns a DataLoader with appropriate transforms.

### ImageNet Dataset

ImageNet 1k uses torchvision's ImageFolder with standard preprocessing:

**Directory Structure:**
```
imagenet_root/
    train/
        n01440764/
            *.JPEG
        n01443537/
            *.JPEG
        ...
    val/
        n01440764/
            *.JPEG
        ...
```

**Preprocessing Pipeline:**
- Training: RandomResizedCrop(224) → RandomHorizontalFlip → ToTensor → Normalize(ImageNet stats)
- Val: Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet stats)

**Key Configuration:**
- 1000 classes (automatically set when using `--dataset imagenet`)
- Default image size: 224×224
- Uses top-1 and top-5 accuracy metrics

### Evaluation Metrics

**Binary Classification (VWW):**
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix (TP/TN/FP/FN)
- Performance: inference time per sample, throughput, total time

**Multi-class Classification (ImageNet):**
- Top-1 Accuracy (standard accuracy)
- Top-5 Accuracy (correct class in top 5 predictions)
- Performance: inference time per sample, throughput, total time

**Prediction Logic:**
- Binary (single output): sigmoid threshold at 0.5
- Binary (two outputs): softmax + argmax
- Multi-class: top-k predictions using argmax

## Common Commands

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download and prepare Visual Wake Words dataset (~40GB COCO 2014)
bash utils/datasets/vww/setup_vww_dataset.sh ./data

# The trained VWW model downloads automatically when using mobilenet_v1_vww
# Or manually download:
python utils/datasets/vww/download_vww_model.py --output-dir ./models
```

### Running Benchmarks

```bash
# Recommended: Evaluate trained MobileNetV1 (auto-downloads, ~84% accuracy)
# Note: mobilenet_v1 automatically uses mobilenet_v1_vww for VWW dataset
python benchmark.py \
  --model mobilenet_v1 \
  --dataset visual_wake_words \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/coco2014/annotations/vww/instances_val.json

# Custom checkpoint evaluation
python benchmark.py \
  --model ./models/my_model.pth \
  --dataset visual_wake_words \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/coco2014/annotations/vww/instances_val.json

# Direct TFLite model evaluation
python benchmark.py \
  --model ./models/vww_96_float.tflite \
  --dataset visual_wake_words \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/coco2014/annotations/vww/instances_val.json

# Example: Untrained ResNet18 (ImageNet weights, poor VWW accuracy ~50%)
python benchmark.py \
  --model resnet18 \
  --dataset visual_wake_words \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/coco2014/annotations/vww/instances_val.json

# CPU-only evaluation
python benchmark.py \
  --model mobilenet_v1 \
  --dataset visual_wake_words \
  --device cpu \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/coco2014/annotations/vww/instances_val.json
```

### ImageNet Evaluation

```bash
# Evaluate MobileNetV2 on ImageNet validation set (pretrained weights)
python benchmark.py \
  --model mobilenet_v2 \
  --dataset imagenet \
  --imagenet-root ./data/imagenet

# Evaluate ResNet50 on ImageNet with custom batch size
python benchmark.py \
  --model resnet50 \
  --dataset imagenet \
  --imagenet-root ./data/imagenet \
  --batch-size 64

# Evaluate Vision Transformer on ImageNet
python benchmark.py \
  --model vit_b_16 \
  --dataset imagenet \
  --imagenet-root ./data/imagenet

# Evaluate MobileNetV3 variants
python benchmark.py \
  --model mobilenet_v3_small \
  --dataset imagenet \
  --imagenet-root ./data/imagenet

# CPU-only ImageNet evaluation
python benchmark.py \
  --model mobilenet_v2 \
  --dataset imagenet \
  --imagenet-root ./data/imagenet \
  --device cpu
```

## Important Implementation Details

### Checkpoint Format

When saving model checkpoints, use this metadata structure for compatibility with the loader:

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'architecture': 'resnet18',  # torchvision model name
    'num_classes': 2,
    'dataset': 'visual_wake_words'
}
```

The loader checks for `model` key (full model), then `state_dict` or `model_state_dict` keys.

### Dataset Requirements

**Visual Wake Words:**
- `--vww-root`: Directory containing COCO images (expects `coco2014/all/` structure from setup script)
- `--vww-ann`: JSON annotation file with VWW binary labels

The setup script (`utils/datasets/vww/setup_vww_dataset.sh`) creates proper directory structure and generates annotations from COCO using the visualwakewords repository scripts.

**ImageNet:**
- `--imagenet-root`: Root directory containing `train/` and `val/` subdirectories
- Each subdirectory contains class folders named by WordNet IDs (e.g., `n01440764/`)
- Download from https://www.image-net.org/ (requires registration)
- Standard structure from ILSVRC2012 challenge is supported

### Adding New Datasets

1. Create `datasets/your_dataset.py` with a factory function returning a DataLoader
2. Add export in `datasets/__init__.py`
3. Update `SUPPORTED_DATASETS` list in `benchmark.py`
4. Add dataset-specific argument parsing in `parse_args()`
5. Add loading logic in the dataset selection block

Follow the `get_visual_wake_words_dataset()` pattern for consistency.

## MLPerf Tiny Compatibility

This framework supports MLPerf Tiny visual wake words benchmark:
- Dataset: Visual Wake Words from COCO 2014
- Image size: 96×96 (use `--image-size 96`)
- Model: MobileNetV1/V2 or equivalent
- Target accuracy: 80% for MLPerf submissions
- Task: Binary person detection

## Testing Strategy

The codebase uses manual validation rather than automated tests. When making changes:

1. Test with trained VWW model: `python benchmark.py --model mobilenet_v1_vww --dataset visual_wake_words ...`
2. Test with ImageNet pretrained model: `python benchmark.py --model mobilenet_v2 --dataset imagenet ...`
3. Verify metrics output matches expected format (binary vs multi-class)
4. Test model loading fallback chain (TFLite → checkpoint → torchvision → timm)
5. Validate dataset loading with different splits (train/val/test)
6. Verify preprocessing selection (TFLite vs ImageNet normalization)

## Project Structure Notes

- `benchmark.py` - Single entry point, keep CLI interface stable
- `utils/model_loader.py` - Model loading logic, maintain fallback chain
- `utils/evaluation.py` - Evaluation metrics (binary and multi-class), extend for new metric types
- `datasets/` - Dataset loaders, each returns configured DataLoader
  - `visual_wake_words.py` - VWW dataset loader (binary classification)
  - `imagenet.py` - ImageNet 1k dataset loader (1000-class classification)
- `models/` - Custom model implementations (currently empty, for future use)
- `utils/datasets/vww/setup_vww_dataset.sh` - Bash script for dataset automation
- `utils/datasets/vww/download_vww_model.py` - Utility for VWW-trained model preparation

## Design Patterns Used

- **Factory Pattern**: `load_model()` and dataset getters create appropriate objects
- **Strategy Pattern**: Multiple model loading strategies with automatic selection
- **Adapter Pattern**: `OriginalVWWDataset` wraps pyvww for PyTorch compatibility
- **Composition**: Transforms pipeline, metrics calculation

Maintain these patterns when extending functionality.
