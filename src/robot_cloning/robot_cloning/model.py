"""PyTorch Behavior Cloning network, training loop, checkpoint I/O and inference wrapper (CPU friendly)."""
import numpy as np
import torch
import torch.nn as nn

from .common import ACTIONS, BC_JOINTS, FEATURE_NAMES, N_FEATURES, N_JOINTS


class BCNet(nn.Module):
    """Pose features -> (robot joint targets, action logits). Supervised behaviour cloning."""

    def __init__(self, n_in=N_FEATURES, n_out=N_JOINTS, n_actions=len(ACTIONS), hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.joint_head = nn.Linear(hidden, n_out)
        self.action_head = nn.Linear(hidden, n_actions)

    def forward(self, x):
        z = self.trunk(x)
        return self.joint_head(z), self.action_head(z)


def split_by_episode(EP, A, val_frac=0.2, seed=0):
    """Stratified (per action) split on whole episodes so validation frames are never seen in training."""
    rng = np.random.default_rng(seed)
    eps = np.unique(EP)
    ep_action = {e: int(A[EP == e][0]) for e in eps}
    val = set()
    for a in sorted(set(ep_action.values())):
        es = [e for e in eps if ep_action[e] == a]
        rng.shuffle(es)
        if len(es) >= 2:
            val.update(es[:max(1, int(round(len(es) * val_frac)))])
    val_mask = np.isin(EP, list(val))
    return ~val_mask, val_mask


def train_model(X, Y, A, EP, epochs=200, lr=2e-3, batch=128, val_frac=0.2, seed=0, hidden=128,
                action_weight=0.2, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, va = split_by_episode(EP, A, val_frac, seed)
    if va.sum() == 0:
        log("WARNING: not enough episodes for a validation split; validating on training data.")
        va = tr
    fm, fs = X[tr].mean(0), np.maximum(X[tr].std(0), 0.05)
    jm, js = Y[tr].mean(0), np.maximum(Y[tr].std(0), 0.05)
    f = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    Xt, Yt, At = f((X[tr] - fm) / fs), f((Y[tr] - jm) / js), torch.tensor(A[tr])
    Xv, Yv, Av = f((X[va] - fm) / fs), f((Y[va] - jm) / js), torch.tensor(A[va])
    model = BCNet(X.shape[1], Y.shape[1], len(ACTIONS), hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse, ce = nn.MSELoss(), nn.CrossEntropyLoss()
    best, best_state, hist = float("inf"), None, []
    n = len(Xt)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            pj, pa = model(Xt[idx])
            loss = mse(pj, Yt[idx]) + action_weight * ce(pa, At[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            pj, pa = model(Xv)
            vloss = float(mse(pj, Yv) + action_weight * ce(pa, Av))
            vrmse = float(torch.sqrt(torch.mean(((pj - Yv) * f(js)) ** 2)))
            vacc = float((pa.argmax(1) == Av).float().mean())
        hist.append((ep, tot / n, vloss, vrmse, vacc))
        if vloss < best:
            best, best_state = vloss, {k: v.clone() for k, v in model.state_dict().items()}
        if ep == 1 or ep % 20 == 0 or ep == epochs:
            log(f"epoch {ep:4d}/{epochs}  train_loss {tot / n:.4f}  val_loss {vloss:.4f}  "
                f"val_joint_RMSE {vrmse:.3f} rad  val_action_acc {vacc * 100:.1f}%")
    model.load_state_dict(best_state)
    meta = dict(feature_mean=fm.tolist(), feature_std=fs.tolist(), joint_mean=jm.tolist(), joint_std=js.tolist(),
                joint_names=list(BC_JOINTS), feature_names=list(FEATURE_NAMES), actions=list(ACTIONS),
                hidden=int(hidden), n_train=int(tr.sum()), n_val=int(va.sum()))
    # final metrics of best model
    model.eval()
    with torch.no_grad():
        pj, pa = model(Xv)
        meta["val_joint_rmse_rad"] = float(torch.sqrt(torch.mean(((pj - Yv) * f(js)) ** 2)))
        meta["val_action_acc"] = float((pa.argmax(1) == Av).float().mean())
    return model, meta, hist


def save_checkpoint(path, model, meta):
    torch.save({"state_dict": model.state_dict(), "meta": meta}, str(path))


def load_checkpoint(path):
    ck = torch.load(str(path), map_location="cpu", weights_only=True)
    meta = ck["meta"]
    model = BCNet(len(meta["feature_mean"]), len(meta["joint_mean"]), len(meta["actions"]), meta["hidden"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, meta


class BehaviorCloningPolicy:
    """Loaded model + normalisation: predict(features) -> (joint targets [rad], action name, probs)."""

    def __init__(self, path):
        self.model, self.meta = load_checkpoint(path)
        self.fm = np.array(self.meta["feature_mean"], dtype=np.float32)
        self.fs = np.array(self.meta["feature_std"], dtype=np.float32)
        self.jm = np.array(self.meta["joint_mean"], dtype=np.float32)
        self.js = np.array(self.meta["joint_std"], dtype=np.float32)
        self.joint_names = self.meta["joint_names"]
        self.actions = self.meta["actions"]

    @torch.no_grad()
    def predict(self, features):
        x = torch.tensor(((np.asarray(features, dtype=np.float32) - self.fm) / self.fs)[None])
        pj, pa = self.model(x)
        joints = pj[0].numpy() * self.js + self.jm
        probs = torch.softmax(pa[0], 0).numpy()
        return joints, self.actions[int(probs.argmax())], probs
