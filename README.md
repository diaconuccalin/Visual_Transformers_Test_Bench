# Visual Transformers Test Bench

A benchmarking framework for evaluating visual transformer and hybrid models on various datasets, with a focus on models targeting low-power devices.

## Features

- **Easy-to-use CLI interface** for model evaluation
- **Visual Wake Words dataset** support (binary classification: person vs no person)
- **Comprehensive metrics**: accuracy, precision, recall, F1 score, inference time, throughput
- **Flexible model loading**: supports PyTorch checkpoints, torchvision models, and timm models
- **GPU and CPU support** for inference

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

## Usage

### Basic Usage

Evaluate a pretrained model on the Visual Wake Words dataset:

```bash
python benchmark.py --model mobilenet_v2 --dataset visual_wake_words
```

### Advanced Usage

```bash
# Use a custom model checkpoint
python benchmark.py --model ./models/my_model.pth --dataset visual_wake_words

# Adjust batch size and image size
python benchmark.py --model mobilenet_v2 --dataset visual_wake_words --batch-size 64 --image-size 224

# Use CPU instead of GPU
python benchmark.py --model mobilenet_v2 --dataset visual_wake_words --device cpu

# Evaluate on validation set
python benchmark.py --model mobilenet_v2 --dataset visual_wake_words --split val

# Specify custom data directory
python benchmark.py --model mobilenet_v2 --dataset visual_wake_words --data-dir ./data
```

### Command Line Arguments

- `--model`: Path to model checkpoint or name of pretrained model (required)
- `--dataset`: Dataset to evaluate on (required, currently supports: `visual_wake_words`)
- `--split`: Dataset split to use (`train`, `val`, or `test`, default: `test`)
- `--batch-size`: Batch size for evaluation (default: 32)
- `--num-workers`: Number of data loading workers (default: 4)
- `--image-size`: Input image size (default: 224)
- `--device`: Device to run evaluation on (`cuda` or `cpu`, default: auto-detect)
- `--data-dir`: Directory to store/load dataset
- `--num-classes`: Number of output classes (default: 2)
- `--quiet`: Suppress progress bars

## Supported Datasets

### Visual Wake Words

The Visual Wake Words dataset is a binary classification task derived from the COCO dataset. The task is to detect whether an image contains a person or not.

- **Classes**: 2 (person present, no person)
- **Train samples**: ~82,783
- **Validation samples**: ~40,504
- **Test samples**: ~40,775
- **Source**: [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/visual_wake_words)

## Supported Models

The benchmark script supports multiple model formats:

1. **PyTorch Checkpoints**: Load custom trained models from `.pth` or `.pt` files
2. **Torchvision Models**: Use any pretrained model from torchvision (e.g., `mobilenet_v2`, `resnet18`, `vit_b_16`)
3. **Timm Models**: Access thousands of models from the timm library (e.g., `mobilevit_s`, `efficientnet_b0`)

### Example Models for Low-Power Devices

- `mobilenet_v2`: Efficient CNN for mobile devices
- `mobilenet_v3_small`: Lightweight MobileNet variant
- `mobilevit_s`: Mobile Vision Transformer (requires timm)
- `efficientnet_b0`: Efficient CNN with excellent accuracy/efficiency trade-off
- `resnet18`: Lightweight ResNet variant

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
├── datasets/
│   ├── __init__.py
│   └── visual_wake_words.py  # Visual Wake Words dataset loader
├── models/
│   └── __init__.py           # Custom model implementations
├── utils/
│   ├── __init__.py
│   ├── evaluation.py         # Evaluation metrics and utilities
│   └── model_loader.py       # Model loading utilities
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Adding New Datasets

To add a new dataset:

1. Create a new dataset loader in `datasets/` (e.g., `datasets/my_dataset.py`)
2. Implement a function that returns a PyTorch DataLoader
3. Update `datasets/__init__.py` to export the new function
4. Add the dataset name to `SUPPORTED_DATASETS` in `benchmark.py`
5. Add dataset loading logic in the `get_dataset()` function

## Adding Custom Models

To add custom model architectures:

1. Implement your model in `models/` directory
2. Save trained checkpoints with the full model or state_dict
3. Load the model using the `--model` argument pointing to the checkpoint file

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- TensorFlow 2.12+ (for dataset loading)
- tensorflow-datasets
- NumPy
- Pillow
- tqdm

## Future Enhancements

- [ ] Add more datasets (ImageNet, CIFAR-10, etc.)
- [ ] Support for ONNX and TFLite models
- [ ] Energy consumption measurement
- [ ] Model compression techniques (quantization, pruning)
- [ ] Batch inference optimization
- [ ] Export results to CSV/JSON
- [ ] Comparative analysis across multiple models

## License

This project is provided as-is for research and benchmarking purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
