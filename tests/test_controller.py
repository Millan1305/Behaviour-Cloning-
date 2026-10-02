import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "robot_cloning"))
from robot_cloning.h1_logic import H1JointCommandMapper, load_yaml  # noqa: E402


def mapper():
    return H1JointCommandMapper(load_yaml(ROOT / "config" / "h1_joints.yaml"))


def test_joint_count_and_mapping_targets_exist():
    m = mapper()
    assert len(m.joint_order) == 19
    for v in m.mapping.values():
        assert v["joint"] in m.joint_order


def test_limits_and_signs():
    m = mapper()
    m.set_canonical_targets(["right_shoulder_abduction", "left_hip_flexion", "left_knee_flexion"], [10.0, 0.5, 9.0])
    for _ in range(500):
        c = m.step(0.02)
    assert abs(c[m.idx["right_shoulder_roll_joint"]] - m.lo[m.idx["right_shoulder_roll_joint"]]) < 1e-3   # clamped, negative sign
    assert abs(c[m.idx["left_hip_pitch_joint"]] + 0.5) < 1e-3
    assert abs(c[m.idx["left_knee_joint"]] - m.hi[m.idx["left_knee_joint"]]) < 1e-3


def test_rate_limit():
    m = mapper()
    m.set_canonical_targets(["left_elbow_flexion"], [2.0])
    c = m.step(0.02)
    assert abs(c[m.idx["left_elbow_joint"]]) <= m.max_velocity * 0.02 + 1e-9


def test_unknown_joint_reported():
    assert mapper().set_canonical_targets(["nope"], [1.0]) == ["nope"]
