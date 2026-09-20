#!/usr/bin/env bash
set -euo pipefail

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" train.py \
  --config configs/pretrain/w2rep_vitl16.yaml "$@"

