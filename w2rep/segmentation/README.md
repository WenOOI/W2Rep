# ADE20K frozen-backbone evaluation

The adapter exposes normalized intermediate W2Rep patch features to
MMSegmentation's Feature2Pyramid + UPerNet stack. It loads the EMA `target`
encoder by default and does not use the predictor or auxiliary clip latents.

The paper protocol uses:

- ADE20K at 512 x 512;
- a frozen W2Rep backbone;
- block outputs `(2, 5, 8, 11)` for ViT-B/16 and `(5, 11, 17, 23)` for
  ViT-L/16;
- Feature2Pyramid with scales `[4, 2, 1, 0.5]`;
- UPerNet trained for 160k iterations with AdamW.

Install a compatible MMSegmentation environment separately. The template uses
the package-qualified `mmseg::` base-config syntax available in modern
MMEngine/MMSegmentation releases.

```bash
PYTHONPATH=$PWD mim train mmseg \
  w2rep/segmentation/upernet_ade20k_160k.py \
  --work-dir outputs/ade20k \
  --cfg-options \
    model.backbone.checkpoint_path=/path/to/ckpt_final.pt \
    train_dataloader.dataset.data_root=/path/to/ADEChallengeData2016 \
    val_dataloader.dataset.data_root=/path/to/ADEChallengeData2016 \
    test_dataloader.dataset.data_root=/path/to/ADEChallengeData2016
```

For ViT-L/16, additionally set the four `embed_dim`/`in_channels` fields shown
in the comments of the configuration template.

