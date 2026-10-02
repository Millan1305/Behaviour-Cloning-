import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "robot_cloning"))
from robot_cloning.common import ACTIONS, BC_JOINTS, N_FEATURES, N_JOINTS, action_params, landmarks_to_features, retarget, synth_landmarks  # noqa: E402
from robot_cloning.dataset import preprocess, read_demo_csv, write_demo_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_features_shape_and_rest_pose():
    lm = synth_landmarks(action_params("stand", 0.0), np.random.default_rng(0), noise=0.0)
    f = landmarks_to_features(lm)
    assert f.shape == (N_FEATURES,)
    assert f[0] < 0.3 and f[1] < 0.3          # arms hanging
    assert f[4] > 3.0 and f[6] > 3.0          # legs straight


def test_retarget_matches_action_semantics():
    rng = np.random.default_rng(0)
    up = retarget(landmarks_to_features(synth_landmarks(action_params("raise_hand", 0.6), rng, 0.0)))
    assert up[BC_JOINTS.index("right_shoulder_abduction")] > 2.5
    assert up[BC_JOINTS.index("left_shoulder_abduction")] < 0.3
    bend = retarget(landmarks_to_features(synth_landmarks(action_params("bend", 0.5), rng, 0.0)))
    assert bend[BC_JOINTS.index("left_hip_flexion")] > 0.7


def test_csv_roundtrip():
    lm = np.stack([synth_landmarks(action_params("wave", i / 29), np.random.default_rng(i)) for i in range(30)])
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.csv"
        write_demo_csv(p, [dict(episode=0, action="wave", t=np.arange(30) / 20, lm=lm)])
        eps = read_demo_csv(p)
    assert len(eps) == 1 and eps[0]["lm"].shape == (30, 12, 4) and eps[0]["action"] == "wave"
    assert np.allclose(eps[0]["lm"], lm, atol=1e-4)


def test_sample_dataset_preprocess():
    with tempfile.TemporaryDirectory() as d:
        arrs, eps, ref = preprocess("sample", ROOT / "data" / "demonstrations", ROOT / "data" / "sample",
                                    Path(d) / "ds.npz", Path(d) / "ref.npz")
    assert arrs["X"].shape[1] == N_FEATURES and arrs["Y"].shape[1] == N_JOINTS
    assert set(np.unique(arrs["A"])) == set(range(len(ACTIONS)))
    assert all(f"ref_{a}" in ref for a in ACTIONS)
    assert len(np.unique(arrs["EP"])) >= 8
