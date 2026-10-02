"""ROS-free H1 command logic: canonical BC joints -> clamped, rate-limited H1 joint vector."""
from pathlib import Path

import numpy as np
import yaml


def load_yaml(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


class H1JointCommandMapper:
    def __init__(self, cfg, joint_info=None):
        """cfg: parsed config/h1_joints.yaml.  joint_info: parsed generated/h1_joint_info.yaml (optional)."""
        if joint_info and joint_info.get("joint_order"):
            self.joint_order = list(joint_info["joint_order"])
            limits = {k: tuple(v) for k, v in joint_info["limits"].items()}
        else:
            self.joint_order = list(cfg["joint_order"])
            limits = {k: tuple(v) for k, v in cfg["default_limits"].items()}
        self.limits = {j: limits.get(j, (-3.14, 3.14)) for j in self.joint_order}
        self.mapping = cfg["mapping"]
        for canon, m in self.mapping.items():
            if m["joint"] not in self.joint_order:
                raise ValueError(f"Mapping '{canon}' refers to joint '{m['joint']}' which is not in the controlled "
                                 f"joint list {self.joint_order}")
        c = cfg.get("controller", {})
        self.max_velocity = float(c.get("max_velocity", 2.5))
        self.alpha = float(c.get("smoothing", 0.35))
        neutral = float(cfg.get("neutral", {}).get("default", 0.0))
        n = len(self.joint_order)
        self.idx = {j: i for i, j in enumerate(self.joint_order)}
        self.lo = np.array([self.limits[j][0] for j in self.joint_order])
        self.hi = np.array([self.limits[j][1] for j in self.joint_order])
        self.target = np.clip(np.full(n, neutral), self.lo, self.hi)
        self.filtered = self.target.copy()
        self.current = self.target.copy()

    def set_canonical_targets(self, names, values):
        """names/values: canonical BC joint names and their predicted values (rad)."""
        unknown = []
        for n, v in zip(names, values):
            m = self.mapping.get(n)
            if m is None:
                unknown.append(n)
                continue
            i = self.idx[m["joint"]]
            self.target[i] = np.clip(m["sign"] * float(v) + m.get("offset", 0.0), self.lo[i], self.hi[i])
        return unknown

    def step(self, dt):
        """Advance one control tick and return the full joint command vector (rad), in joint_order."""
        self.filtered += self.alpha * (self.target - self.filtered)
        max_step = self.max_velocity * dt
        self.current += np.clip(self.filtered - self.current, -max_step, max_step)
        self.current = np.clip(self.current, self.lo, self.hi)
        return self.current.copy()


def load_mapper(root):
    root = Path(root)
    cfg = load_yaml(root / "config" / "h1_joints.yaml")
    info_path = root / "generated" / "h1_joint_info.yaml"
    info = load_yaml(info_path) if info_path.exists() else None
    return H1JointCommandMapper(cfg, info)
