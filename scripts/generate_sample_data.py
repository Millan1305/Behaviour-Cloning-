#!/usr/bin/env python3
"""Generate the bundled SAMPLE demonstration dataset.

IMPORTANT / HONEST NOTE: the sample data is SYNTHETIC. It is produced by a parametric
stick-figure human (robot_cloning.common.synth_landmarks) so that the repository works
before you own any webcam recordings. It uses exactly the same CSV format and the same
feature-extraction pipeline as real MediaPipe recordings. Record your own demos with
./record_demo.sh <action> to train on real human motion.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "robot_cloning"))
from robot_cloning.common import ACTIONS, action_params, synth_landmarks  # noqa: E402
from robot_cloning.dataset import write_demo_csv  # noqa: E402

FPS, DURATION, EPISODES = 20, 4.0, 6


def main():
    out = ROOT / "data" / "sample"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)
    for action in ACTIONS:
        eps = []
        for k in range(EPISODES):
            n = int(FPS * DURATION * rng.uniform(0.85, 1.15))
            amp = rng.uniform(0.85, 1.0 if action != "bend" else 1.1)
            phase = rng.uniform(0, 2 * np.pi)
            ts = np.arange(n) / FPS
            lm = np.stack([synth_landmarks(action_params(action, i / (n - 1), amp, phase), rng) for i in range(n)])
            eps.append(dict(episode=k, action=action, t=ts, lm=lm))
        write_demo_csv(out / f"sample_{action}.csv", eps)
        print(f"wrote {out / f'sample_{action}.csv'}")


if __name__ == "__main__":
    main()
