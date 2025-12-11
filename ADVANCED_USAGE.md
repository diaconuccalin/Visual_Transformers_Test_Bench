# Advanced Usage

This guide covers advanced usage scenarios for the Visual Transformers Test Bench.

## Custom Model Checkpoint

Load and evaluate a custom trained model from a checkpoint file:

```bash
python benchmark.py --model ./models/my_model.pth --dataset visual_wake_words \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/coco2014/annotations/vww/instances_train.json
```

## Adjusting Batch Size and Image Size

Configure batch size and input image dimensions for your evaluation:

```bash
python benchmark.py --model mobilenet_v1_vww --dataset visual_wake_words \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/coco2014/annotations/vww/instances_val.json \
  --batch-size 64
```

**Note**: The `mobilenet_v1_vww` model automatically uses 96×96 images (MLPerf Tiny standard). For custom models, specify `--image-size` to match your model's input requirements.

## Using CPU Instead of GPU

Force evaluation on CPU even when GPU is available:

```bash
python benchmark.py --model mobilenet_v1_vww --dataset visual_wake_words \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/coco2014/annotations/vww/instances_val.json \
  --device cpu
```

## Evaluating on Training Set

Evaluate on the training split instead of the validation split:

```bash
python benchmark.py --model mobilenet_v1_vww --dataset visual_wake_words \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/coco2014/annotations/vww/instances_train.json \
  --split train
```

## Combining Options

You can combine multiple options for fine-grained control:

```bash
# MLPerf Tiny benchmark configuration with custom batch size
python benchmark.py --model mobilenet_v1_vww --dataset visual_wake_words \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/coco2014/annotations/vww/instances_val.json \
  --batch-size 32 --device cuda

# Large batch evaluation on custom model
python benchmark.py --model ./models/custom_vit.pth --dataset visual_wake_words \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/coco2014/annotations/vww/instances_val.json \
  --batch-size 128 --num-workers 8 --image-size 224
```

For a complete list of command line arguments, see the main [README.md](README.md#command-line-arguments).
