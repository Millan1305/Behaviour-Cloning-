#!/usr/bin/env python3
"""ROS-free inference: run the trained network on an action's reference sequence (or live webcam) and print joints.

  python3 scripts/run_inference.py --action wave
  python3 scripts/run_inference.py --camera           # live pose -> joints (needs webcam + mediapipe)
  python3 scripts/run_inference.py --check            # software-only self test (exit code != 0 on failure)
"""
import argparse
import sys
import time

import numpy as np
from _bootstrap import ROOT, need

need("torch")
from robot_cloning.common import ACTIONS, BC_JOINTS, landmarks_to_features  # noqa: E402
from robot_cloning.h1_logic import load_mapper  # noqa: E402
from robot_cloning.model import BehaviorCloningPolicy  # noqa: E402

MODEL, REF = ROOT / "models" / "bc_model.pt", ROOT / "models" / "reference_trajectories.npz"


def load():
    if not MODEL.exists() or not REF.exists():
        sys.exit("ERROR: model or reference trajectories missing. Run ./train.sh first.")
    return BehaviorCloningPolicy(MODEL), np.load(REF)


def run_action(policy, ref, mapper, action, verbose=True):
    seq = ref[f"ref_{action}"]
    outs, preds = [], []
    for f in seq:
        j, a, _ = policy.predict(f)
        mapper.set_canonical_targets(policy.joint_names, j)
        for _ in range(3):
            cmd = mapper.step(0.02)
        outs.append((j, cmd.copy()))
        preds.append(a)
    J = np.stack([o[0] for o in outs])
    if verbose:
        print(f"{action:11s} frames={len(seq):3d}  predicted-action(mode)={max(set(preds), key=preds.count):10s} "
              f"joint max (rad): " + " ".join(f"{n.split('_')[0][0]}{n.split('_')[1][:3]}={J[:, i].max():.2f}" for i, n in enumerate(BC_JOINTS)))
    return J, preds, [o[1] for o in outs]


def check():
    policy, ref = load()
    mapper = load_mapper(ROOT)
    ok = True
    expect = {  # canonical joint index & minimum peak (rad)
        "raise_hand": (BC_JOINTS.index("right_shoulder_abduction"), 2.0),
        "wave": (BC_JOINTS.index("right_shoulder_abduction"), 1.4),
        "bend": (BC_JOINTS.index("left_hip_flexion"), 0.6),
    }
    for a in ACTIONS:
        J, preds, _ = run_action(policy, ref, mapper, a)
        mode = max(set(preds), key=preds.count)
        good = mode == a
        if a in expect:
            i, thr = expect[a]
            good &= J[:, i].max() > thr
        if a == "wave":
            good &= np.ptp(J[:, BC_JOINTS.index("right_elbow_flexion")]) > 0.5
        print(("PASS" if good else "FAIL"), a)
        ok &= bool(good)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--action", choices=ACTIONS)
    ap.add_argument("--camera", action="store_true")
    ap.add_argument("--video", default="")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        check()
    policy, ref = load()
    mapper = load_mapper(ROOT)
    if a.action:
        run_action(policy, ref, mapper, a.action)
    elif a.camera or a.video:
        need("cv2")
        from robot_cloning.vision import PoseSource, PoseSourceError
        try:
            src = PoseSource("video" if a.video else "camera", 0, a.video)
        except PoseSourceError as e:
            sys.exit(f"ERROR: {e}")
        print("Live inference, Ctrl+C to stop")
        try:
            while True:
                lm, _, ok = src.read()
                if lm is not None:
                    j, act, _ = policy.predict(landmarks_to_features(lm))
                    print(f"{act:10s} " + " ".join(f"{v:5.2f}" for v in j), end="\r")
        except KeyboardInterrupt:
            pass
    else:
        ap.print_help()
