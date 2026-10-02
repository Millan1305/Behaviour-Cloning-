"""Demonstration CSV I/O and dataset preprocessing (numpy only).

CSV format (one row per video frame), one file per recording session:
  episode,timestamp,action,lm0_x,lm0_y,lm0_z,lm0_v, ... lm11_x,lm11_y,lm11_z,lm11_v
Landmarks are the 12 MediaPipe landmarks listed in common.LM_NAMES.
Robot target joint values are computed during preprocessing via common.retarget().
"""
import csv
from pathlib import Path

import numpy as np

from .common import (ACTIONS, BC_JOINTS, FEATURE_NAMES, LM_NAMES, N_FEATURES, N_JOINTS, N_LM,
                     landmarks_to_features, retarget)

HEADER = ["episode", "timestamp", "action"] + [f"lm{i}_{c}" for i in range(N_LM) for c in "xyzv"]


def write_demo_csv(path, episodes):
    """episodes: list of dicts {episode:int, action:str, t:(T,), lm:(T,12,4)}"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for ep in episodes:
            for t, lm in zip(ep["t"], ep["lm"]):
                w.writerow([ep["episode"], f"{t:.4f}", ep["action"]] + [f"{v:.5f}" for v in np.asarray(lm).reshape(-1)])


def read_demo_csv(path):
    """Return list of episode dicts from one CSV file."""
    path = Path(path)
    eps = {}
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        header = next(r, None)
        if header is None or header[:3] != HEADER[:3] or len(header) != len(HEADER):
            raise ValueError(f"{path}: unexpected CSV header (expected {len(HEADER)} columns starting with {HEADER[:3]})")
        for row in r:
            if not row:
                continue
            e = eps.setdefault(row[0], dict(episode=row[0], action=row[2], t=[], lm=[]))
            if row[2] not in ACTIONS:
                raise ValueError(f"{path}: unknown action '{row[2]}' (valid: {ACTIONS})")
            e["t"].append(float(row[1]))
            e["lm"].append(np.array(row[3:], dtype=float).reshape(N_LM, 4))
    out = []
    for e in eps.values():
        e["t"] = np.array(e["t"])
        e["lm"] = np.stack(e["lm"])
        e["file"] = path.name
        out.append(e)
    return out


def load_episodes(dirs, min_visibility=0.5):
    """Load all *.csv from the given [(source_name, dir)] list. Drops low-visibility frames."""
    episodes = []
    for source, d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.csv")):
            for e in read_demo_csv(f):
                keep = e["lm"][:, :, 3].min(axis=1) >= min_visibility
                if keep.sum() < 5:
                    continue
                e["t"], e["lm"] = e["t"][keep], e["lm"][keep]
                e["source"] = source
                e["uid"] = f"{source}:{f.stem}:{e['episode']}"
                episodes.append(e)
    return episodes


def episodes_to_arrays(episodes):
    X, Y, A, EP, T = [], [], [], [], []
    for i, e in enumerate(episodes):
        feats = np.stack([landmarks_to_features(lm) for lm in e["lm"]])
        joints = np.stack([retarget(f) for f in feats])
        X.append(feats)
        Y.append(joints)
        A.append(np.full(len(feats), ACTIONS.index(e["action"])))
        EP.append(np.full(len(feats), i))
        T.append(e["t"] - e["t"][0])
    return dict(X=np.concatenate(X).astype(np.float32), Y=np.concatenate(Y).astype(np.float32),
                A=np.concatenate(A).astype(np.int64), EP=np.concatenate(EP).astype(np.int64),
                T=np.concatenate(T).astype(np.float32))


def build_reference_trajectories(episodes):
    """For each action pick one representative episode (user demos take priority over sample).

    Used by the keyboard "action mode": the recorded pose-feature sequence is fed through the
    trained network, which then produces the joint commands.
    """
    ref = {}
    for a in ACTIONS:
        cands = [e for e in episodes if e["action"] == a]
        user = [e for e in cands if e["source"] == "user"]
        cands = user or cands
        if not cands:
            continue
        cands = sorted(cands, key=lambda e: len(e["t"]))
        e = cands[len(cands) // 2]
        feats = np.stack([landmarks_to_features(lm) for lm in e["lm"]]).astype(np.float32)
        dt = float(np.median(np.diff(e["t"]))) if len(e["t"]) > 1 else 0.05
        ref[f"ref_{a}"] = feats
        ref[f"dt_{a}"] = np.float32(max(dt, 1e-3))
    return ref


def preprocess(sources, demo_dir, sample_dir, out_npz, out_ref, min_visibility=0.5):
    dirs = []
    if sources in ("user", "both"):
        dirs.append(("user", demo_dir))
    if sources in ("sample", "both"):
        dirs.append(("sample", sample_dir))
    episodes = load_episodes(dirs, min_visibility)
    if not episodes:
        raise FileNotFoundError(
            f"No demonstrations found in {[str(d) for _, d in dirs]}. Record some with ./record_demo.sh wave "
            f"or regenerate the sample set with: python3 scripts/generate_sample_data.py")
    arrs = episodes_to_arrays(episodes)
    meta_actions = np.array([e["action"] for e in episodes])
    meta_uid = np.array([e["uid"] for e in episodes])
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **arrs, episode_action=meta_actions, episode_uid=meta_uid,
             feature_names=np.array(FEATURE_NAMES), joint_names=np.array(BC_JOINTS), actions=np.array(ACTIONS))
    ref = build_reference_trajectories(episodes)
    Path(out_ref).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_ref, **ref)
    return arrs, episodes, ref


def load_processed(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}
