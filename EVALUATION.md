# Evaluation protocols

This document separates reusable W2Rep evaluation code from the comparison
methods used in the paper. All commands consume a W2Rep checkpoint and never
instantiate an external baseline.

## Frozen linear probes

The common protocol is implemented by `w2rep.eval.linear_probe`:

1. load the EMA `target` encoder;
2. freeze every encoder parameter;
3. mean-pool patch features;
4. standardize features using training-set mean and standard deviation;
5. train an AdamW linear head for 50 epochs at learning rate `1e-3` and weight
   decay `1e-4`;
6. report the fixed final epoch for seeds 42, 43, and 44.

The script records top-1, top-5, macro accuracy, per-seed curves, checkpoint
SHA-256, preprocessing, temporal sampling, and feature-cache identity.

### ImageNet-1K

Use a torchvision `ImageFolder` layout with `train/` and `val/` directories.
The paper protocol directly resizes RGB images to 224 x 224.

```bash
python -m w2rep.eval.linear_probe image \
  --checkpoint checkpoints/w2rep_vitb16.pt \
  --data-root /datasets/imagenet \
  --output-dir outputs/imagenet
```

### SSv2, UCF101, and Diving48

Use a CSV with the columns below. Additional columns are ignored.

```csv
path,label,split,id
videos/1.mp4,0,train,1
videos/2.mp4,0,val,2
```

Labels must be contiguous integers beginning at zero. The paper protocol
uniformly samples 8 frames over each full video and directly resizes them to
224 x 224. Two encoding modes are reported:

- `independent`: run the encoder on each frame separately and average the
  resulting frame descriptors;
- `joint`: run the encoder once on all 8 frames and average over space and
  time.

```bash
python -m w2rep.eval.linear_probe video \
  --checkpoint checkpoints/w2rep_vitb16.pt \
  --manifest data/ssv2.csv \
  --data-root /datasets/ssv2 \
  --encoding joint \
  --output-dir outputs/ssv2_joint
```

## Full video fine-tuning

The default fine-tuning recipe matches the paper's SSv2 experiment: joint
8-frame encoding, 50 epochs, effective global batch 256 on 8 GPUs, five warmup
epochs, AdamW, layer-wise learning-rate decay 0.75, label smoothing 0.1, and no
horizontal flip. The fixed epoch-50 model is reported; intermediate validation
does not select a checkpoint.

```bash
NPROC_PER_NODE=8 scripts/finetune_video.sh \
  --checkpoint checkpoints/w2rep_vitb16.pt \
  --manifest data/ssv2.csv \
  --data-root /datasets/ssv2 \
  --output-dir outputs/ssv2_full_finetune
```

The base learning rate is automatically scaled by the effective global batch.
Resume is supported at saved epoch boundaries.

## ADE20K

ADE20K uses a frozen W2Rep backbone with Feature2Pyramid and UPerNet for 160k
iterations. See `w2rep/segmentation/README.md`. This environment is optional
and intentionally isolated from the core PyTorch dependencies.

## Protocol changes

Any change to frame sampling, image resizing, checkpoint key, encoding mode,
probe seed, or training duration is a different protocol. The evaluation code
records these choices so caches and results cannot be silently mixed.

