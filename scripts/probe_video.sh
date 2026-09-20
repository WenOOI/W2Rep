#!/usr/bin/env bash
set -euo pipefail

python -m w2rep.eval.linear_probe video "$@"

