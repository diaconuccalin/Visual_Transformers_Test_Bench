# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Visual Transformers Test Bench is a benchmarking framework for evaluating visual models on low-power devices with MLPerf Tiny compatibility. The architecture uses a modular pipeline: CLI → Model Loading → Dataset Loading → Evaluation → Metrics Reporting.

## Core Architecture

The codebase has four main components that work together:

1. **benchmark.py**: Main orchestrator that coordinates the evaluation pipeline
2. **utils/model_loader.py**: Multi-source model loading with automatic fallback (file checkpoints → torchvision → timm)
3. **utils/evaluation.py**: Model evaluation and comprehensive metrics calculation
4. **datasets/visual_wake_words.py**: Dataset loader wrapping pyvww with PyTorch DataLoader interface

### Model Loading Strategy

The `load_model()` function implements a fallback chain:

**Priority 1 - File Checkpoints:**
- Detects checkpoint format (full model vs state_dict)
- Reconstructs architecture from metadata if state_dict-only
- Adapts classifier head for target num_classes

**Priority 2 - Torchvision:**
- Loads pretrained models (e.g., `mobilenet_v2`, `resnet18`)
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

### Evaluation Metrics

The evaluation pipeline calculates binary classification metrics:
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix (TP/TN/FP/FN)
- Performance: inference time per sample, throughput, total time

**Prediction Logic:**
- Single output: sigmoid threshold at 0.5
- Multiple outputs: softmax + argmax

## Common Commands

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download and prepare Visual Wake Words dataset (~40GB COCO 2014)
bash setup_vww_dataset.sh ./data

# Optional: Download VWW-trained MobileNet model
python download_vww_model.py --output-dir ./models
```

### Running Benchmarks

```bash
# Basic evaluation with pretrained model
python benchmark.py \
  --model mobilenet_v2 \
  --dataset visual_wake_words \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/coco2014/annotations/vww/instances_val.json

# Custom checkpoint evaluation
python benchmark.py \
  --model ./models/my_model.pth \
  --dataset visual_wake_words \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/coco2014/annotations/vww/instances_val.json

# MLPerf Tiny compliant (96×96 images)
python benchmark.py \
  --model mobilenet_v2 \
  --dataset visual_wake_words \
  --image-size 96 \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/coco2014/annotations/vww/instances_train.json

# CPU-only evaluation
python benchmark.py \
  --model mobilenet_v2 \
  --dataset visual_wake_words \
  --device cpu \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/coco2014/annotations/vww/instances_val.json
```

## Important Implementation Details

### Checkpoint Format

When saving model checkpoints, use this metadata structure for compatibility with the loader:

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'architecture': 'mobilenet_v2',  # torchvision model name
    'num_classes': 2,
    'dataset': 'visual_wake_words'
}
```

The loader checks for `model` key (full model), then `state_dict` or `model_state_dict` keys.

### Dataset Requirements

Visual Wake Words expects:
- `vww_root`: Directory containing COCO images (expects `coco2014/all/` structure from setup script)
- `vww_ann`: JSON annotation file with VWW binary labels

The setup script (`setup_vww_dataset.sh`) creates proper directory structure and generates annotations from COCO using the visualwakewords repository scripts.

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

1. Test with pretrained model: `python benchmark.py --model mobilenet_v2 --dataset visual_wake_words ...`
2. Verify metrics output matches expected format
3. Test model loading fallback chain (checkpoint → torchvision → timm)
4. Validate dataset loading with different splits (train/val/test)

## Project Structure Notes

- `benchmark.py` - Single entry point, keep CLI interface stable
- `utils/model_loader.py` - Model loading logic, maintain fallback chain
- `utils/evaluation.py` - Evaluation metrics, extend for new metric types
- `datasets/` - Dataset loaders, each returns configured DataLoader
- `models/` - Custom model implementations (currently empty, for future use)
- `setup_vww_dataset.sh` - Bash script for dataset automation
- `download_vww_model.py` - Utility for VWW-trained model preparation

## Design Patterns Used

- **Factory Pattern**: `load_model()` and dataset getters create appropriate objects
- **Strategy Pattern**: Multiple model loading strategies with automatic selection
- **Adapter Pattern**: `OriginalVWWDataset` wraps pyvww for PyTorch compatibility
- **Composition**: Transforms pipeline, metrics calculation

Maintain these patterns when extending functionality.
