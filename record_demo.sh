#!/usr/bin/env bash
# ./record_demo.sh <stand|raise_hand|wave|bend> [--duration 4] [--episodes 3] [--video clip.mp4]
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/common.sh"
check_dir; venv_activate
[ $# -ge 1 ] || die "usage: ./record_demo.sh <stand|raise_hand|wave|bend> [options]"
python "$ROOT/scripts/record_demo.py" "$@"
