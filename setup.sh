#!/usr/bin/env bash
# One-command setup: dependencies -> H1 model -> build -> first model.
set -Eeuo pipefail
trap 'echo -e "\n\033[1;31mSETUP FAILED\033[0m at line $LINENO: $BASH_COMMAND\nFix the error above and re-run ./setup.sh (it is safe to re-run)." >&2' ERR
source "$(dirname "${BASH_SOURCE[0]}")/scripts/common.sh"
check_dir
cd "$ROOT"
. /etc/os-release
[ "${VERSION_ID:-}" = "24.04" ] || die "Ubuntu 24.04 required (found ${PRETTY_NAME:-?})"
info "1/5 Dependencies"; ./install_dependencies.sh
ros_source; command -v gz >/dev/null && ok "Gazebo: $(gz sim --versions 2>/dev/null | head -1)"
info "2/5 Unitree H1 description"; ./scripts/fetch_h1.sh
info "3/5 Build workspace"; ./build.sh
info "4/5 Sample dataset"; [ -f data/sample/sample_wave.csv ] || python3 scripts/generate_sample_data.py
info "5/5 Train initial Behavior Cloning model on sample data"
if [ ! -f models/bc_model.pt ]; then ./train.sh; else ok "model already exists (models/bc_model.pt)"; fi
echo
echo "SETUP COMPLETE"
echo "Next:   cd $ROOT && ./run.sh"
