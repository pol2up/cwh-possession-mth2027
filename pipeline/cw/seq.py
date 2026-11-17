"""Sequence model over the tracking features (bidirectional GRU at 5 Hz).

Trained only on other labelled matches.  The gradient-boosted stage A sees one frame at a time;
this model reads 30 s windows, which is where short possessions and hand-overs are decided.
Inference runs the five 5 Hz phases of the 25 fps feed and both team orientations.
"""
import numpy as np
import torch
import torch.nn as nn

from cw import features as F
from cw.masks import runs

STEP = 5
WIN = 150          # 30 s at 5 Hz
DEV = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self, n_in, hidden=128, layers=2, drop=.15):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(drop))
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True, bidirectional=True, dropout=drop)
        self.out = nn.Linear(2 * hidden, 3)

    def forward(self, x):
        h, _ = self.gru(self.inp(x))
        return self.out(h)


def spell_weights(ref):
    """Per-frame weight: short spells count more, long ones less (clip sqrt(5 s / duration))."""
    w = np.ones(len(ref), np.float32)
    for l, s, e in runs(ref):
        if l < 3:
            w[s:e] = np.clip(np.sqrt(5. * 25 / (e - s)), .5, 2.)
    return w


class Scaler:
    def __init__(self, xs):
        x = np.concatenate([a[::7] for a in xs])
        self.mu = np.nanmean(x, axis=0)
        self.sd = np.nanstd(x, axis=0) + 1e-6

    def __call__(self, x):
        return np.nan_to_num(np.clip((x - self.mu) / self.sd, -6, 6), nan=0.).astype(np.float32)


def train(seqs, names, epochs=14, batches=500, bs=32, hidden=128, seed=2027, weighted=True, log=None):
    """seqs: list of (x_5hz raw, y_5hz, w_5hz, [(a, z) 5 Hz blocks])."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sc = Scaler([s[0] for s in seqs])
    data = []
    for x, y, w, blocks in seqs:
        xa, xb = sc(x), sc(F.swap(x, names))
        yb = np.where(y < 2, 1 - y, y)
        for a, z in blocks:
            if z - a > WIN:
                data.append((torch.from_numpy(xa[a:z]), torch.from_numpy(xb[a:z]),
                             torch.from_numpy(y[a:z].astype(np.int64)), torch.from_numpy(yb[a:z].astype(np.int64)),
                             torch.from_numpy(w[a:z])))
    lens = np.array([len(d[2]) for d in data], float)
    pick = lens / lens.sum()
    net = Net(seqs[0][0].shape[1], hidden).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * batches)
    lossf = nn.CrossEntropyLoss(ignore_index=3, reduction='none')
    for ep in range(epochs):
        net.train()
        tot = 0.
        for _ in range(batches):
            xb_, yb_, wb_ = [], [], []
            for i in rng.choice(len(data), bs, p=pick):
                xa, xs, y, ys, w = data[i]
                s = int(rng.integers(0, len(y) - WIN))
                sw = rng.random() < .5
                xb_.append((xs if sw else xa)[s:s + WIN])
                yb_.append((ys if sw else y)[s:s + WIN])
                wb_.append(w[s:s + WIN])
            x = torch.stack(xb_).to(DEV)
            y = torch.stack(yb_).to(DEV)
            w = torch.stack(wb_).to(DEV)
            l = lossf(net(x).reshape(-1, 3), y.reshape(-1))
            if weighted:
                l = l * w.reshape(-1)
            l = l.sum() / (y != 3).sum().clamp(min=1)
            opt.zero_grad()
            l.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.)
            opt.step()
            sched.step()
            tot += float(l)
        if log:
            log(f'  epoch {ep + 1}/{epochs} loss {tot / batches:.4f}')
    return net.eval(), sc


@torch.no_grad()
def predict(net, sc, x, names, blocks):
    """x: full-rate raw features.  Returns full-rate probabilities (uniform outside blocks)."""
    out = np.full((len(x), 3), 1 / 3, np.float32)
    xa, xs = sc(x), sc(F.swap(x, names))
    for a, z in blocks:
        for ph in range(STEP):
            idx = np.arange(a + ph, z, STEP)
            pa = torch.softmax(net(torch.from_numpy(xa[idx])[None].to(DEV)), -1)[0].cpu().numpy()
            pb = torch.softmax(net(torch.from_numpy(xs[idx])[None].to(DEV)), -1)[0].cpu().numpy()
            out[idx] = (pa + pb[:, [1, 0, 2]]) / 2
    return out


def blend(p, q, w=.5):
    """Geometric blend of two probability arrays."""
    l = (1 - w) * np.log(np.clip(p, 1e-5, 1)) + w * np.log(np.clip(q, 1e-5, 1))
    e = np.exp(l - l.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def save(path, net, sc, names):
    torch.save({'state': {k: v.cpu() for k, v in net.state_dict().items()},
                'mu': torch.from_numpy(np.asarray(sc.mu, np.float64)), 'sd': torch.from_numpy(np.asarray(sc.sd, np.float64)),
                'names': list(names)}, path)


def load(path, device=None):
    """Returns (net, scaler, feature names).  Tensors and plain lists only, so weights_only is safe."""
    global DEV
    if device is not None:
        DEV = torch.device(device)
    blob = torch.load(path, weights_only=True, map_location='cpu')
    hidden = blob['state']['inp.0.weight'].shape[0]
    net = Net(blob['state']['inp.0.weight'].shape[1], hidden).to(DEV)
    net.load_state_dict(blob['state'])
    sc = Scaler.__new__(Scaler)
    sc.mu, sc.sd = blob['mu'].numpy(), blob['sd'].numpy()
    return net.eval(), sc, blob['names']
