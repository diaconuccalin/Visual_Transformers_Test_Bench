# Advanced Usage

This guide covers advanced usage scenarios for the Visual Transformers Test Bench.

## Custom Model Checkpoint

Load and evaluate a custom trained model from a checkpoint file:

**Visual Wake Words:**
```bash
python benchmark.py --model ./models/my_model.pth --dataset visual_wake_words \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/coco2014/annotations/vww/instances_train.json
```

**CIFAR-10:**
```bash
python benchmark.py --model ./models/my_cifar_model.pth --dataset cifar10 \
  --num-classes 10
```

## Adjusting Batch Size and Image Size

Configure batch size and input image dimensions for your evaluation:

**Visual Wake Words:**
```bash
python benchmark.py --model mobilenet_v1_vww --dataset visual_wake_words \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/coco2014/annotations/vww/instances_val.json \
  --batch-size 64
```

**CIFAR-10:**
```bash
python benchmark.py --model mobilenet_v1 --dataset cifar10 \
  --batch-size 128
```

**Note**: The `mobilenet_v1_vww` model automatically uses 96×96 images (MLPerf Tiny standard), and `mobilenet_v1_cifar10` uses 32×32 (CIFAR-10 standard). For custom models, specify `--image-size` to match your model's input requirements.

## Using CPU Instead of GPU

Force evaluation on CPU even when GPU is available:

```bash
python benchmark.py --model mobilenet_v1_vww --dataset visual_wake_words \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/coco2014/annotations/vww/instances_val.json \
  --device cpu

# Or for CIFAR-10
python benchmark.py --model mobilenet_v1 --dataset cifar10 --device cpu
```

## Evaluating on Different Splits

Evaluate on the training or validation split instead of the test split:

**Visual Wake Words:**
```bash
python benchmark.py --model mobilenet_v1_vww --dataset visual_wake_words \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/coco2014/annotations/vww/instances_train.json \
  --split train
```

**CIFAR-10:**
```bash
# Evaluate on validation split (10% of training data)
python benchmark.py --model mobilenet_v1 --dataset cifar10 --split val

# Evaluate on training split
python benchmark.py --model mobilenet_v1 --dataset cifar10 --split train
```

## Using Custom Data Paths

Specify custom paths for dataset storage:

**CIFAR-10:**
```bash
python benchmark.py --model mobilenet_v1 --dataset cifar10 \
  --cifar10-root /my/custom/path/data
```

## Combining Options

You can combine multiple options for fine-grained control:

```bash
# MLPerf Tiny benchmark configuration with custom batch size
python benchmark.py --model mobilenet_v1_vww --dataset visual_wake_words \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/coco2014/annotations/vww/instances_val.json \
  --batch-size 32 --device cuda

# Large batch evaluation on custom model for CIFAR-10
python benchmark.py --model ./models/custom_resnet.pth --dataset cifar10 \
  --batch-size 256 --num-workers 8 --device cuda

# Custom VIT on VWW with specific settings
python benchmark.py --model ./models/custom_vit.pth --dataset visual_wake_words \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/coco2014/annotations/vww/instances_val.json \
  --batch-size 128 --num-workers 8 --image-size 224
```

## Evaluating Pretrained Models from Torchvision

You can evaluate any torchvision model, though they need fine-tuning for best results:

**VWW with ResNet18 (ImageNet weights - expect ~50% accuracy):**
```bash
python benchmark.py --model resnet18 --dataset visual_wake_words \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/coco2014/annotations/vww/instances_val.json
```

**CIFAR-10 with ResNet18 (ImageNet weights - expect ~60% accuracy):**
```bash
python benchmark.py --model resnet18 --dataset cifar10 --image-size 32
```

For a complete list of command line arguments, see the main [README.md](README.md#command-line-arguments).
