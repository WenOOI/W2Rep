#!/usr/bin/env bash
set -euo pipefail

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" \
  -m w2rep.eval.finetune_video "$@"

