import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "robot_cloning"))
try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def _data():
    from robot_cloning.dataset import preprocess
    with tempfile.TemporaryDirectory() as d:
        arrs, _, _ = preprocess("sample", ROOT / "data" / "demonstrations", ROOT / "data" / "sample",
                                Path(d) / "a.npz", Path(d) / "b.npz")
    return arrs


def test_train_save_load_predict():
    if not HAVE_TORCH:
        print("SKIPPED (torch not installed)")
        return
    from robot_cloning.model import BehaviorCloningPolicy, save_checkpoint, split_by_episode, train_model
    a = _data()
    tr, va = split_by_episode(a["EP"], a["A"])
    assert not (set(a["EP"][tr]) & set(a["EP"][va])), "episodes must not leak between train/val"
    model, meta, hist = train_model(a["X"], a["Y"], a["A"], a["EP"], epochs=40, log=lambda *_: None)
    assert hist[-1][1] < hist[0][1], "training loss should decrease"
    assert meta["val_joint_rmse_rad"] < 0.35 and meta["val_action_acc"] > 0.6
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.pt"
        save_checkpoint(p, model, meta)
        pol = BehaviorCloningPolicy(p)
    j, act, probs = pol.predict(a["X"][0])
    assert j.shape == (8,) and abs(probs.sum() - 1) < 1e-4
