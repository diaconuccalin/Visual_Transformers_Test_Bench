# Visual Wake Words Model Setup

This guide explains how to set up and use MobileNet models trained for the Visual Wake Words task.

## Quick Start

### Option 1: Use ImageNet-pretrained MobileNet with VWW-adapted head

```bash
python download_vww_model.py
```

This creates a MobileNetV2 with:
- ImageNet pretrained backbone
- 2-class classification head (person/not-person)
- Saved to `./models/mobilenet_v2_vww.pth`

### Option 2: Use the model name directly

The benchmark script automatically adapts MobileNet models for binary classification:

```bash
python benchmark.py --model mobilenet_v2 --dataset visual_wake_words \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/vww/instances_val.json
```

## Available Pre-trained Models

### Official MLPerf Tiny Model (TFLite)

The official MLPerf Tiny benchmark provides a quantized TFLite model:
- **File**: `vww_96_int8.tflite`
- **URL**: https://github.com/mlcommons/tiny/blob/master/benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite
- **Format**: TensorFlow Lite (int8 quantized)
- **Input size**: 96×96 pixels
- **Accuracy**: ~85% on VWW test set

**Note**: This is a TFLite model and requires conversion to PyTorch format for use with this benchmark.

### PyTorch Conversion

For PyTorch users, you can convert the winning MIT VWW solution:

1. Download the TensorFlow model from [MIT HAN Lab](https://github.com/mit-han-lab/VWW)
2. Use the conversion script from [mitvww-pytorch](https://github.com/mzemlyanikin/mitvww-pytorch):
   ```bash
   git clone https://github.com/mzemlyanikin/mitvww-pytorch.git
   cd mitvww-pytorch
   python load_weights_from_pb.py -m /path/to/model_fp32.pb
   ```
3. This creates `mitvww_pytorch.pth` with trained weights

### EdgeImpulse Weights

EdgeImpulse provides MobileNet transfer learning weights:
- **URL**: https://cdn.edgeimpulse.com/transfer-learning-weights/keras/mobilenet_2_5_128_tf.h5
- **Format**: Keras/TensorFlow
- **Note**: Requires conversion to PyTorch

## Model Architecture for VWW

The Visual Wake Words task uses:
- **Input**: 96×96 RGB images (resized from COCO)
- **Architecture**: MobileNetV1 or MobileNetV2
- **Output**: 2 classes (person present: 1, no person: 0)
- **Loss**: Binary cross-entropy
- **Target accuracy**: ≥80% for MLPerf submissions

### MobileNetV2 Configuration

```python
import torchvision.models as models
import torch.nn as nn

# Load pretrained MobileNetV2
model = models.mobilenet_v2(pretrained=True)

# Adapt for VWW (2 classes)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(1280, 2)  # 1280 is MobileNetV2's last_channel
)
```

## Fine-tuning on VWW

To fine-tune a model on the Visual Wake Words dataset:

```bash
# First, set up the VWW dataset
bash setup_vww_dataset.sh ./data

# Create a model with VWW-adapted head
python download_vww_model.py

# Fine-tune (you'll need to implement training script)
python train_vww.py --model ./models/mobilenet_v2_vww.pth \
  --data-root ./data/coco2014/all \
  --train-ann ./data/vww/instances_train.json \
  --val-ann ./data/vww/instances_val.json \
  --epochs 10 \
  --lr 0.001
```

## Evaluation

Once you have a trained model, evaluate it:

```bash
python benchmark.py --model ./models/mobilenet_v2_vww.pth \
  --dataset visual_wake_words \
  --vww-root ./data/coco2014/all \
  --vww-ann ./data/vww/instances_val.json \
  --image-size 96  # MLPerf standard
```

## Model Performance

Expected performance on VWW test set:

| Model | Accuracy | Params | FLOPs | Notes |
|-------|----------|--------|-------|-------|
| MobileNetV1 | ~85% | 4.2M | 569M | MLPerf baseline |
| MobileNetV2 | ~87% | 3.5M | 300M | Better efficiency |
| MobileNetV2 (ImageNet init) | ~75-80% | 3.5M | 300M | Without VWW fine-tuning |

**Note**: Models fine-tuned on VWW significantly outperform ImageNet-only models.

## References

- [MLPerf Tiny Benchmark](https://github.com/mlcommons/tiny)
- [MIT VWW PyTorch](https://github.com/mzemlyanikin/mitvww-pytorch)
- [Visual Wake Words Dataset Paper](https://arxiv.org/abs/1906.05721)
- [TensorFlow VWW Blog Post](https://blog.tensorflow.org/2019/10/visual-wake-words-with-tensorflow-lite_30.html)

## Next Steps

1. **For quick testing**: Use ImageNet-pretrained MobileNet with adapted head
2. **For MLPerf benchmarking**: Fine-tune on VWW dataset or convert official TFLite model
3. **For best results**: Train from scratch or fine-tune with data augmentation
