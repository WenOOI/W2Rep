# Release implementation audit

This file records checks performed while separating W2Rep from the research
workspace. It is not a substitute for rerunning the public artifact on a clean
machine.

## Method identity

The default ViT-B/16 configuration was matched against the final paper run:

- mask-position supervision only;
- ordinary bidirectional attention in the masked-video pass;
- one same-image and three cross-frame targets;
- four spatial prediction blocks sampled at scale `[0.15, 0.20]`;
- signed temporal offsets;
- 16 auxiliary clip latents;
- Smooth L1 prediction loss plus `1e-4 * mean(z^2)`;
- 8 frames with training stride 3;
- 100k optimizer steps.

The historical context-region loss and its `0.75/0.25` weights are not active
in the final method and are absent from the default release configuration.

## Checkpoint compatibility

The final ViT-B/16 step-100k research checkpoint has SHA-256:

```text
652d55cda680e98d100fcee2df372f6541e56444c23d9a4e3a6e7be2a2a61e95
```

The following checks passed on 2026-09-20:

1. EMA encoder state loaded with `strict=True` (151 tensors).
2. Predictor state loaded with `strict=True` (147 tensors).
3. A single-image forward pass matched the research implementation exactly:
   maximum and mean absolute differences were both `0.0`.
4. A masked two-frame pass, including auxiliary clip latents, matched exactly:
   maximum absolute differences for both patch features and latents were `0.0`.
5. A predictor forward pass with nonzero signed offsets matched exactly:
   maximum and mean absolute differences were both `0.0`.
6. With the same PyTorch seed, freshly initialized release encoder and
   predictor state dicts matched the research implementation exactly.

The loader also handles the legacy serialized configuration safely without
requiring the old `utils.config.DotDict` module.

The release preserves the spatial-mask distribution but gives its generator a
deterministic batch-indexed stream. The development loader generated mask
locations inside data-loader workers, so its exact random sequence depended on
worker scheduling. Consequently, a new run reproduces the method and recipe,
not the historical run's mask sequence bit for bit.

## Automated checks

The lightweight tests cover:

- image, joint-video, and intermediate-layer feature shapes;
- predictor gradients through image context and clip latents;
- rectangular inputs for dense evaluation;
- non-overlap and regular batching of spatial masks;
- inherited ablation configuration resolution;
- portable encoder-checkpoint round trips.

The available server environment did not include `pytest`, so these functions
were executed directly after `python -m compileall`. All checks passed. The
same files are compatible with `pytest -q` once the test extra is installed.

## Evaluation protocol separation

The release makes two previously implicit choices explicit:

- pretraining uses a stride-3 temporal window and shared spatial augmentation;
- frozen video evaluation defaults to 8 uniformly sampled frames with direct
  224 x 224 resizing, matching the paper's probe scripts.

ImageNet frozen evaluation likewise defaults to direct 224 x 224 resizing.
Center-crop and stride-based alternatives are available only through explicit
CLI flags and are recorded in the feature-cache identity.
