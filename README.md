# Visual Transformers Test Bench

A benchmarking framework for evaluating visual transformer and hybrid models on various datasets, with a focus on models targeting low-power devices and **MLPerf Tiny compatibility**.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/diaconuccalin/Visual_Transformers_Test_Bench.git
cd Visual_Transformers_Test_Bench
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Setup

### Quick Setup (Automated)

Run the automated setup script to download COCO 2014 and generate Visual Wake Words annotations:

```bash
bash setup_vww_dataset.sh ./data
```

This will:
1. Download COCO 2014 dataset (~40 GB)
2. Generate Visual Wake Words annotations
3. Create train/minival splits

### Manual Setup

For manual setup instructions or if you need more control over the installation process, see [MANUAL_SETUP.md](datasets/MANUAL_SETUP.md).

## Usage

### Basic Usage

Evaluate the trained MobileNetV1 model on Visual Wake Words (automatically downloads if needed):

```bash
python benchmark.py --model mobilenet_v1 --dataset visual_wake_words \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/coco2014/annotations/vww/instances_val.json
```

This will:
- Automatically use the trained VWW model (you can also use `--model mobilenet_v1_vww` explicitly)
- Download the MLCommons Tiny trained model (~1MB) if not already present
- Use the correct 96x96 input size
- Achieve ~84% accuracy on the validation set

Note: For the VWW dataset, `mobilenet_v1` automatically loads the trained `mobilenet_v1_vww` model. Using other models from torchvision or timm will result in poor accuracy (~50%, random guessing) unless you train them first on the VWW dataset.

### Advanced Usage

For advanced usage scenarios including custom models, batch size configuration, and device selection, see [ADVANCED_USAGE.md](ADVANCED_USAGE.md).


### Command Line Arguments

**Core Arguments:**
- `--model`: Path to model checkpoint or name of pretrained model (required)
- `--dataset`: Dataset to evaluate on (required, currently supports: `visual_wake_words`)
- `--split`: Dataset split to use (`train`, `val`, or `test`, default: `test`)
- `--batch-size`: Batch size for evaluation (default: 32)
- `--num-workers`: Number of data loading workers (default: 4)
- `--image-size`: Input image size (default: 224)
- `--device`: Device to run evaluation on (`cuda` or `cpu`, default: auto-detect)
- `--num-classes`: Number of output classes (default: 2)
- `--quiet`: Suppress progress bars

**Dataset Arguments:**
- `--vww-root`: Root directory for VWW COCO images (required)
- `--vww-ann`: Path to VWW annotation JSON file (required)

## Supported Datasets

### Visual Wake Words (MLPerf Compatible)

The **Visual Wake Words dataset** is a binary classification task derived from the COCO dataset, used in the MLPerf Tiny benchmark suite.

- **Classes**: 2 (person present, no person)
- **Train samples**: ~82,783
- **Validation samples**: ~40,504
- **Test samples**: ~40,775 (same as validation/minival split)
- **Setup**: Requires COCO 2014 download and annotation generation
- **Source**: [TensorFlow Models - Visual Wake Words](https://github.com/tensorflow/models/blob/master/research/slim/datasets/download_and_convert_visualwakewords.py)
- **MLPerf**: Official dataset for MLPerf Tiny visual wake words benchmark

**MLPerf Tiny Specifications:**
- Image size: 96×96 pixels (resized from COCO)
- Model: MobileNetV1 or equivalent
- Accuracy target: 80% for MLPerf submissions
- Task: Binary classification (person/not-person detection)

## Supported Models

The benchmark script supports multiple model formats:

1. **Trained VWW Models**: Pre-trained models that auto-download from MLCommons
   - `mobilenet_v1_vww`: MobileNetV1 (alpha=0.25) trained on VWW, ~84% accuracy (Recommended)

2. **TFLite Models**: TensorFlow Lite models (`.tflite` files)
   - Automatically detected and loaded with correct preprocessing

3. **PyTorch Checkpoints**: Load custom trained models from `.pth` or `.pt` files

4. **Torchvision Models**: Use any pretrained model from torchvision (e.g., `resnet18`, `vit_b_16`)
   - Note: These are ImageNet-pretrained, not VWW-trained. Expect poor accuracy without fine-tuning on VWW.

5. **Timm Models**: Access thousands of models from the timm library (e.g., `mobilevit_s`, `efficientnet_b0`)

### Example Models for Low-Power Devices

- `mobilenet_v1_vww`: **Recommended** - Pre-trained on VWW, 96x96 input, ~84% accuracy
- `resnet18`: Lightweight ResNet variant (ImageNet weights, requires VWW fine-tuning)
- `mobilevit_s`: Mobile Vision Transformer (requires timm library and VWW training)
- `efficientnet_b0`: Efficient CNN with excellent accuracy/efficiency trade-off (requires VWW training)

## Output Metrics

The benchmark script provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: Proportion of positive identifications that were correct
- **Recall**: Proportion of actual positives that were identified correctly
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: True/False Positives/Negatives
- **Inference Time**: Average time per sample (in milliseconds)
- **Throughput**: Samples processed per second
- **Total Time**: Total evaluation time

## Project Structure

```
Visual_Transformers_Test_Bench/
├── benchmark.py              # Main benchmark script
├── setup_vww_dataset.sh      # Automated dataset setup script
├── datasets/
│   ├── __init__.py
│   └── visual_wake_words.py  # VWW and Wake Vision dataset loaders
├── models/
│   └── __init__.py           # Custom model implementations
├── utils/
│   ├── __init__.py
│   ├── evaluation.py         # Evaluation metrics and utilities
│   └── model_loader.py       # Model loading utilities
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## MLPerf Tiny Benchmark

This repository is designed to work with the MLPerf Tiny benchmark suite. For official MLPerf submissions:

1. Use the original Visual Wake Words dataset (default)
2. Resize images to 96×96 pixels (`--image-size 96`)
3. Target 80% accuracy for the quality metric
4. Follow MLPerf Tiny rules and guidelines

For more information, see:
- [MLPerf Tiny Benchmark](https://mlcommons.org/en/inference-tiny/)
- [Visual Wake Words Challenge](https://arxiv.org/abs/1906.05721)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- pyvww
- NumPy
- Pillow
- tqdm

## References

- [Visual Wake Words Dataset Paper](https://arxiv.org/abs/1906.05721)
- [MLCommons MLPerf Tiny v1.3](https://mlcommons.org/2025/09/mlperf-tiny-v1-3-tech/)
- [pyvww Library](https://github.com/Mxbonn/visualwakewords)
- [TensorFlow Visual Wake Words](https://github.com/tensorflow/models/blob/master/research/slim/datasets/download_and_convert_visualwakewords.py)

## License

This project is provided as-is for research and benchmarking purposes.
