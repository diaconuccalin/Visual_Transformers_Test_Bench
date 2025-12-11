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
python benchmark.py --model mobilenet_v2 --dataset visual_wake_words \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/coco2014/annotations/vww/instances_train.json \
  --batch-size 64 --image-size 224
```

**Note**: For MLPerf Tiny compliance, use `--image-size 96`.

## Using CPU Instead of GPU

Force evaluation on CPU even when GPU is available:

```bash
python benchmark.py --model mobilenet_v2 --dataset visual_wake_words \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/coco2014/annotations/vww/instances_train.json \
  --device cpu
```

## Evaluating on Validation Set

Evaluate on the validation split instead of the test split:

```bash
python benchmark.py --model mobilenet_v2 --dataset visual_wake_words \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/coco2014/annotations/vww/instances_minival.json \
  --split val
```

## Combining Options

You can combine multiple options for fine-grained control:

```bash
# MLPerf Tiny benchmark configuration
python benchmark.py --model mobilenet_v2 --dataset visual_wake_words \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/coco2014/annotations/vww/instances_train.json \
  --image-size 96 --batch-size 32 --device cuda

# Large batch evaluation on custom model
python benchmark.py --model ./models/custom_vit.pth --dataset visual_wake_words \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/coco2014/annotations/vww/instances_train.json \
  --batch-size 128 --num-workers 8
```

For a complete list of command line arguments, see the main [README.md](README.md#command-line-arguments).
