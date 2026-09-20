# Third-party boundary

W2Rep is evaluated against established image- and video-representation
methods, but their repositories are not included here.

The architecture uses standard Vision Transformer building blocks and a
teacher-student joint-embedding training pattern. The public release contains
a small, independently organized implementation needed to instantiate W2Rep
and to load the paper checkpoints. It does not import or copy an I-JEPA,
V-JEPA, VideoMAE, or TDV source tree at runtime.

Datasets and optional evaluation frameworks retain their own licenses and must
be obtained separately:

- Something-Something V2, ImageNet-1K, UCF101, Diving48, and ADE20K;
- PyTorch and torchvision;
- OpenCV;
- MMSegmentation and its dependencies for ADE20K evaluation.

Before publication, add exact versions and links used in the camera-ready
artifact documentation.

