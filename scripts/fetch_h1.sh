#!/usr/bin/env bash
# Fetch ONLY robots/h1_description from the official Unitree repo (sparse checkout, ~tens of MB not hundreds).
#   repo  : https://github.com/unitreerobotics/unitree_ros   (provides the Unitree H1 URDF + meshes)
#   pin   : first run records the checked-out commit in third_party/H1_COMMIT.txt; later runs check out
#           exactly that commit.  Override with H1_REF=<tag|branch|sha>.  Commit H1_COMMIT.txt to your fork.
set -Eeuo pipefail
trap 'echo -e "\n\033[1;31mERROR:\033[0m fetch_h1.sh failed at line $LINENO: $BASH_COMMAND" >&2' ERR
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
REPO="https://github.com/unitreerobotics/unitree_ros.git"
DEST="$ROOT/third_party/unitree_ros"
LOCK="$ROOT/third_party/H1_COMMIT.txt"
command -v git >/dev/null || die "git not installed"
if [ ! -d "$DEST/.git" ]; then
  info "Cloning $REPO (sparse: robots/h1_description)"
  git clone --filter=blob:none --no-checkout --sparse "$REPO" "$DEST" || die "git clone failed (internet? proxy?)"
  git -C "$DEST" sparse-checkout set robots/h1_description
  if [ -n "${H1_REF:-}" ]; then git -C "$DEST" checkout "$H1_REF"
  elif [ -s "$LOCK" ]; then git -C "$DEST" checkout "$(tr -d '[:space:]' < "$LOCK")"
  else git -C "$DEST" checkout; fi
fi
git -C "$DEST" rev-parse HEAD > "$LOCK.new"
if [ ! -s "$LOCK" ]; then mv "$LOCK.new" "$LOCK"; info "Pinned H1 description commit: $(cat "$LOCK")"; else rm -f "$LOCK.new"; fi
H1="$DEST/robots/h1_description"
[ -f "$H1/urdf/h1.urdf" ] || { find "$DEST" -name '*.urdf' | head; die "h1.urdf not found at $H1/urdf/h1.urdf - upstream layout changed? Look at the list above."; }
ok "H1 description at $H1 (commit $(git -C "$DEST" rev-parse --short HEAD))"
python3 "$ROOT/scripts/make_h1_gz_urdf.py"
