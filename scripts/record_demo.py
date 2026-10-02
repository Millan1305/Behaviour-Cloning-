#!/usr/bin/env python3
"""Record a human demonstration (webcam or video file) to data/demonstrations/<action>_<timestamp>.csv.

  python3 scripts/record_demo.py wave                 # 4 s, webcam 0, 3 s countdown
  python3 scripts/record_demo.py wave --episodes 5    # five takes in one file
  python3 scripts/record_demo.py wave --video my.mp4  # no webcam: use a pre-recorded clip
Face the camera, full body in view (at least shoulders to ankles). Start and end in a relaxed standing pose.
"""
import argparse
import sys
import time

import numpy as np
from _bootstrap import ROOT, need

need("cv2")
need("mediapipe")
from robot_cloning.common import ACTIONS  # noqa: E402
from robot_cloning.dataset import write_demo_csv  # noqa: E402
from robot_cloning.vision import PoseSource, PoseSourceError  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=ACTIONS)
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--countdown", type=float, default=3.0)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--video", default="")
    ap.add_argument("--no-preview", action="store_true")
    a = ap.parse_args()
    try:
        src = PoseSource("video" if a.video else "camera", a.camera_index, a.video, loop_video=False)
    except PoseSourceError as e:
        sys.exit(f"ERROR: {e}")
    import cv2
    preview = not a.no_preview
    episodes = []
    for ep in range(a.episodes):
        if not a.video:
            t_end = time.time() + a.countdown
            while time.time() < t_end:
                _, frame, _ = src.read()
                if preview and frame is not None:
                    cv2.imshow("record_demo", src.draw(cv2.flip(frame, 1), f"{a.action} take {ep + 1}: starts in {t_end - time.time():.0f}"))
                    cv2.waitKey(1)
        ts, lms, t0 = [], [], time.time()
        while True:
            lm, frame, ok = src.read()
            if not ok or (not a.video and time.time() - t0 >= a.duration):
                break
            if lm is not None and lm[:, 3].min() >= 0.5:
                ts.append(time.time() - t0)
                lms.append(lm)
            if preview and frame is not None:
                cv2.imshow("record_demo", src.draw(cv2.flip(frame, 1), f"REC {a.action} {time.time() - t0:.1f}s"))
                cv2.waitKey(1)
        print(f"take {ep + 1}: {len(ts)} frames with a fully visible person")
        if len(ts) >= 10:
            episodes.append(dict(episode=ep, action=a.action, t=np.array(ts), lm=np.stack(lms)))
        else:
            print("  too few valid frames (is your whole body visible? good lighting?) - take discarded")
    src.close()
    if not episodes:
        sys.exit("ERROR: nothing recorded.")
    out = ROOT / "data" / "demonstrations" / f"{a.action}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    write_demo_csv(out, episodes)
    print(f"Saved {out}\nNext: ./train.sh")
