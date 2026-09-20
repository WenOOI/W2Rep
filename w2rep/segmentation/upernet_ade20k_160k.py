"""Frozen W2Rep-B/16 + UPerNet, 160k-iteration ADE20K template."""

_base_ = [
    "mmseg::_base_/datasets/ade20k.py",
    "mmseg::_base_/default_runtime.py",
    "mmseg::_base_/schedules/schedule_160k.py",
]

custom_imports = dict(
    imports=["w2rep.segmentation.mmseg_backbone"], allow_failed_imports=False
)

crop_size = (512, 512)
norm_cfg = dict(type="SyncBN", requires_grad=True)
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)

model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="W2RepMMSegBackbone",
        checkpoint_path=None,
        checkpoint_key="target",
        image_size=224,
        out_indices=(2, 5, 8, 11),
        frozen=True,
    ),
    neck=dict(type="Feature2Pyramid", embed_dim=768, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type="UPerHead",
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
        ),
    ),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="slide", crop_size=crop_size, stride=(336, 336)),
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW", lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05
    ),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type="PolyLR",
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    ),
]

train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader

# ViT-L/16 override:
# model.backbone.out_indices=(5,11,17,23)
# model.neck.embed_dim=1024
# model.decode_head.in_channels=[1024,1024,1024,1024]
# model.auxiliary_head.in_channels=1024

