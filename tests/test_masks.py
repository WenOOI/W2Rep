import torch

from w2rep.data.masks import BlockMaskCollator


def _sample():
    return {
        "clip_px": torch.zeros(3, 3, 32, 32, dtype=torch.uint8),
        "source_index": 0,
        "target_indices": [1, 2],
        "label": 0,
        "video_id": "sample",
        "path": "sample.mp4",
    }


def test_block_masks_have_regular_batch_shapes_and_do_not_overlap():
    collator = BlockMaskCollator(
        grid_size=8,
        encoder_scale=(0.6, 0.8),
        predictor_scale=(0.1, 0.2),
        aspect_ratio=(0.75, 1.5),
        predictor_blocks=3,
        min_keep=2,
        seed=9,
    )
    batch = collator([_sample(), _sample()])
    encoder = batch["mask_encoder"]
    predictor = batch["mask_predictor"]
    assert encoder.ndim == 2 and encoder.shape[0] == 2
    assert predictor.ndim == 3 and predictor.shape[:2] == (2, 3)
    for sample in range(2):
        predicted = set(predictor[sample].flatten().tolist())
        visible = set(encoder[sample].tolist())
        assert not predicted.intersection(visible)

