#!/usr/bin/env python3
"""Landmarks CSVs -> features + robot target joints (data/processed/dataset.npz) + reference trajectories."""
import argparse

from _bootstrap import ROOT
from robot_cloning.common import ACTIONS
from robot_cloning.dataset import preprocess


def run(sources="both", quiet=False):
    arrs, episodes, ref = preprocess(sources, ROOT / "data" / "demonstrations", ROOT / "data" / "sample",
                                     ROOT / "data" / "processed" / "dataset.npz",
                                     ROOT / "models" / "reference_trajectories.npz")
    if not quiet:
        print(f"Processed {len(episodes)} episodes, {len(arrs['X'])} samples "
              f"({sum(e['source'] == 'user' for e in episodes)} user, {sum(e['source'] == 'sample' for e in episodes)} sample)")
        for a in ACTIONS:
            n = sum(e['action'] == a for e in episodes)
            print(f"  {a:11s} episodes: {n}")
        print("Saved data/processed/dataset.npz and models/reference_trajectories.npz")
    return arrs, episodes, ref


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", choices=["user", "sample", "both"], default="both",
                    help="which demonstrations to include (default: both)")
    try:
        run(ap.parse_args().sources)
    except FileNotFoundError as e:
        raise SystemExit(f"ERROR: {e}")
