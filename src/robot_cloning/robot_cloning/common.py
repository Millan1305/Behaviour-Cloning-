"""Shared constants, pose-feature extraction, retargeting and a synthetic skeleton generator.

Only numpy is required here, so this module works without ROS or PyTorch.

Landmark coordinate convention (same as MediaPipe *world* landmarks):
  x -> right in the image, y -> DOWN, z -> away from camera (negative = closer to camera).
The person's LEFT limbs appear at +x when the person faces the (unmirrored) camera.
"""
import numpy as np

ACTIONS = ["stand", "raise_hand", "wave", "bend"]
ACTION_KEYS = {"1": "stand", "2": "raise_hand", "3": "wave", "4": "bend"}

# Subset of the 33 MediaPipe Pose landmarks that we use.
LM_NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
LM_MP_IDS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
N_LM = len(LM_NAMES)
LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA = range(12)

FEATURE_NAMES = [
    "l_shoulder_angle", "r_shoulder_angle", "l_elbow_angle", "r_elbow_angle",
    "l_hip_angle", "r_hip_angle", "l_knee_angle", "r_knee_angle",
    "l_wrist_height", "r_wrist_height", "torso_lean", "torso_foreshortening",
]
N_FEATURES = len(FEATURE_NAMES)

# Canonical robot joints predicted by the Behavior Cloning network.
# All values are POSITIVE for abduction / flexion.
# config/h1_joints.yaml maps them (with sign + offset) onto real H1 joints.
BC_JOINTS = [
    "left_shoulder_abduction", "right_shoulder_abduction",
    "left_elbow_flexion", "right_elbow_flexion",
    "left_hip_flexion", "right_hip_flexion",
    "left_knee_flexion", "right_knee_flexion",
]
N_JOINTS = len(BC_JOINTS)
BC_JOINT_RANGE = {
    "shoulder_abduction": (0.0, 3.0), "elbow_flexion": (0.0, 2.5),
    "hip_flexion": (0.0, 2.2), "knee_flexion": (0.0, 2.0),
}


def _joint_range(name):
    for k, v in BC_JOINT_RANGE.items():
        if name.endswith(k):
            return v
    return (-3.2, 3.2)


def _angle(a, b, c):
    """Angle at b (rad) between vectors b->a and b->c."""
    return _vec_angle(np.asarray(a, float) - b, np.asarray(c, float) - b)


def _vec_angle(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-9:
        return np.pi
    return float(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def landmarks_to_features(lm):
    """lm: (12,3) or (12,4) array -> (12,) feature vector (scale/translation invariant)."""
    lm = np.asarray(lm, dtype=float)[:, :3]
    mid_sh = (lm[LS] + lm[RS]) / 2.0
    mid_hip = (lm[LH] + lm[RH]) / 2.0
    torso_down = mid_hip - mid_sh
    torso_up = -torso_down
    torso_len = max(np.linalg.norm(torso_down), 1e-6)
    sh_width = max(np.linalg.norm(lm[LS] - lm[RS]), 1e-6)
    f = np.zeros(N_FEATURES)
    f[0] = _vec_angle(torso_down, lm[LE] - lm[LS])   # 0 = arm hanging, pi/2 = horizontal, pi = overhead
    f[1] = _vec_angle(torso_down, lm[RE] - lm[RS])
    f[2] = _angle(lm[LS], lm[LE], lm[LW])            # pi = straight arm
    f[3] = _angle(lm[RS], lm[RE], lm[RW])
    f[4] = _vec_angle(torso_up, lm[LK] - lm[LH])     # pi = standing straight
    f[5] = _vec_angle(torso_up, lm[RK] - lm[RH])
    f[6] = _angle(lm[LH], lm[LK], lm[LA])
    f[7] = _angle(lm[RH], lm[RK], lm[RA])
    f[8] = (lm[LS][1] - lm[LW][1]) / torso_len       # > 0 : wrist above shoulder (y is down)
    f[9] = (lm[RS][1] - lm[RW][1]) / torso_len
    f[10] = -(mid_sh[2] - mid_hip[2]) / torso_len    # ~ sin(forward lean)
    f[11] = np.linalg.norm((mid_sh - mid_hip)[:2]) / sh_width
    return f


def retarget(f):
    """Kinematic retargeting: human pose features -> canonical robot joint targets (rad).

    These are the *robot target joint values* stored with every demonstration, i.e. the
    supervised labels that the Behavior Cloning network learns to predict from pose.
    """
    pi = np.pi
    j = np.array([f[0], f[1], pi - f[2], pi - f[3], pi - f[4], pi - f[5], pi - f[6], pi - f[7]], dtype=float)
    for i, n in enumerate(BC_JOINTS):
        lo, hi = _joint_range(n)
        j[i] = np.clip(j[i], lo, hi)
    return j


# ----------------------------------------------------------------------------------
# Synthetic human skeleton: used ONLY to generate the bundled sample dataset + tests.
# ----------------------------------------------------------------------------------
def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _pulse(t, up_end, down_start):
    return _smooth(t / up_end) * (1 - _smooth((t - down_start) / (1 - down_start)))


REST = dict(l_sh=0.10, r_sh=0.10, l_el=3.00, r_el=3.00, lean=0.0, knee=0.05)


def action_params(action, t, amp=1.0, phase=0.0):
    """Joint-angle description of a human performing `action` at normalised time t in [0,1]."""
    p = dict(REST)
    if action == "stand":
        p["lean"] = 0.02 * np.sin(2 * np.pi * (t + phase))
    elif action == "raise_hand":
        p["r_sh"] = REST["r_sh"] + (2.9 * amp - REST["r_sh"]) * _pulse(t, 0.4, 0.8)
    elif action == "wave":
        up = _pulse(t, 0.25, 0.85)
        p["r_sh"] = REST["r_sh"] + (1.9 * amp - REST["r_sh"]) * up
        osc = 0.5 * amp * np.sin(2 * np.pi * 3.0 * (t - 0.25) / 0.6 + phase)
        p["r_el"] = REST["r_el"] + (2.0 - REST["r_el"]) * up + osc * up
    elif action == "bend":
        b = _pulse(t, 0.35, 0.65)
        p["lean"] = 0.9 * amp * b
        p["knee"] = REST["knee"] + 0.25 * b
    else:
        raise ValueError(f"unknown action {action}")
    return p


def synth_landmarks(p, rng=None, noise=0.004):
    """Build a (12,4) [x,y,z,visibility] skeleton from joint-angle dict `p`."""
    rng = rng or np.random.default_rng(0)
    sw, hw, ua, fa, th, sh, torso = 0.20, 0.125, 0.30, 0.27, 0.45, 0.45, 0.50
    lm = np.zeros((N_LM, 3))

    def arm(side, a, e):
        s = 1.0 if side == "l" else -1.0
        shoulder = np.array([s * sw, -torso, 0.0])
        d = np.array([s * np.sin(a), np.cos(a), 0.0])
        elbow = shoulder + ua * d
        delta = s * (np.pi - e)
        c, sn = np.cos(delta), np.sin(delta)
        f = np.array([d[0] * c - d[1] * sn, d[0] * sn + d[1] * c, 0.0])
        return shoulder, elbow, elbow + fa * f

    lm[LS], lm[LE], lm[LW] = arm("l", p["l_sh"], p["l_el"])
    lm[RS], lm[RE], lm[RW] = arm("r", p["r_sh"], p["r_el"])
    cs, sn = np.cos(p["lean"]), np.sin(p["lean"])
    for i in (LS, RS, LE, RE, LW, RW):
        y, z = lm[i][1], lm[i][2]
        lm[i][1] = y * cs - z * sn
        lm[i][2] = y * sn + z * cs
    m = p["knee"]
    for hx, ki, ai, hi in ((hw, LK, LA, LH), (-hw, RK, RA, RH)):
        lm[hi] = [hx, 0.0, 0.0]
        lm[ki] = [hx, th, 0.0]
        lm[ai] = lm[ki] + sh * np.array([0.0, np.cos(m), np.sin(m)])
    lm += rng.normal(0.0, noise, lm.shape)
    return np.hstack([lm, np.full((N_LM, 1), 0.99)])
