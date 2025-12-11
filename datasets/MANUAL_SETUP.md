# Manual Dataset Setup

If you prefer manual setup or need more control over the Visual Wake Words dataset installation:

## Prerequisites

Install pyvww:
```bash
pip install pyvww
```

## Step-by-Step Setup

### 1. Clone the visualwakewords repository

```bash
git clone https://github.com/Mxbonn/visualwakewords.git
cd visualwakewords
```

### 2. Download COCO 2014

```bash
bash scripts/download_mscoco.sh /path/to/coco2014 2014
```

This will download the COCO 2014 dataset (~40 GB).

### 3. Create train/minival split

```bash
python scripts/create_coco_train_minival_split.py \
  --train_annotations_file="/path/to/coco2014/annotations/instances_train2014.json" \
  --val_annotations_file="/path/to/coco2014/annotations/instances_val2014.json" \
  --output_dir="/path/to/coco2014/annotations"
```

### 4. Generate Visual Wake Words annotations

```bash
python scripts/create_visualwakewords_annotations.py \
  --train_annotations_file="/path/to/coco2014/annotations/instances_maxitrain.json" \
  --val_annotations_file="/path/to/coco2014/annotations/instances_minival.json" \
  --output_dir="/path/to/coco2014/annotations/vww" \
  --threshold=0.005 \
  --foreground_class='person'
```

## Next Steps

After completing the manual setup, you can use the dataset with the benchmark script:

```bash
python benchmark.py --model mobilenet_v1 --dataset visual_wake_words \
  --vww-root /path/to/coco2014/all \
  --vww-ann /path/to/coco2014/annotations/vww/instances_val.json
```

Note: `mobilenet_v1` automatically uses the trained VWW model for this dataset.

See the main [README.md](../README.md) for more usage examples.
