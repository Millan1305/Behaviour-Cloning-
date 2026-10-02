#!/usr/bin/env python3
"""Check webcam + MediaPipe and print live pose features.  Sanity values: arms hanging -> *_shoulder_angle ~0.1-0.3;
arms overhead -> ~2.8-3.1; straight legs/arms -> hip/knee/elbow angle ~3.0-3.14."""
import argparse
import sys
import time

from _bootstrap import need

need("cv2")
need("mediapipe")
from robot_cloning.common import FEATURE_NAMES, landmarks_to_features  # noqa: E402
from robot_cloning.vision import PoseSource, PoseSourceError  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--video", default="")
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--no-preview", action="store_true")
    a = ap.parse_args()
    try:
        src = PoseSource("video" if a.video else "camera", a.camera_index, a.video)
    except PoseSourceError as e:
        sys.exit(f"ERROR: {e}")
    import cv2
    t0, n, seen = time.time(), 0, 0
    while time.time() - t0 < a.seconds:
        lm, frame, ok = src.read()
        if not ok:
            break
        n += 1
        if lm is not None:
            seen += 1
            f = landmarks_to_features(lm)
            if n % 10 == 0:
                print("  ".join(f"{k}={v:.2f}" for k, v in zip(FEATURE_NAMES, f)))
        if not a.no_preview and frame is not None:
            cv2.imshow("test_camera (press q)", src.draw(cv2.flip(frame, 1), "person detected" if lm is not None else "no person"))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    src.close()
    print(f"\nframes: {n}, person detected in {seen}. {'OK' if seen else 'FAILED: no person detected'}")
    sys.exit(0 if seen else 1)
