#!/usr/bin/env python3
"""CLI wrapper: build generated/h1_gz.urdf from the cloned Unitree URDF (also done automatically at launch)."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "robot_cloning"))
from robot_cloning.h1_logic import load_yaml  # noqa: E402
from robot_cloning.urdf_tools import build_gz_urdf  # noqa: E402

H1 = ROOT / "third_party" / "unitree_ros" / "robots" / "h1_description"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=str(H1 / "urdf" / "h1.urdf"))
    ap.add_argument("--mesh-root", default=str(H1))
    ap.add_argument("--out", default=str(ROOT / "generated"))
    a = ap.parse_args()
    cfg = load_yaml(ROOT / "config" / "h1_joints.yaml")
    r = build_gz_urdf(a.urdf, a.out, a.mesh_root, cfg.get("pelvis_height", 1.05))
    print("Generated:", r["urdf"])
    print("Base link:", r["base_link"], "| movable joints:", len(r["joint_order"]))
    expected = set(cfg["joint_order"])
    missing, extra = expected - set(r["joint_order"]), set(r["joint_order"]) - expected
    if missing or extra:
        print(f"WARNING: joint list differs from config/h1_joints.yaml: missing={sorted(missing)} extra={sorted(extra)}")
