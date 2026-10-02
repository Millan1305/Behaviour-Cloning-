#!/usr/bin/env bash
# Preprocess demonstrations (yours + bundled sample) and train the Behavior Cloning model.
#   ./train.sh                    # sample + your demos
#   ./train.sh --sources user     # only your own demos
#   ./train.sh --epochs 300
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/common.sh"
check_dir; venv_activate
python "$ROOT/scripts/train_behavior_cloning.py" "$@"
