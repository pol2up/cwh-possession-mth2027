"""Second stage: re-read the first stage's probabilities with several seconds of context."""
import numpy as np

FPS = 25
OFFSETS = (-15, -8, -4, -2, -1, -.4, .4, 1, 2, 4, 8, 15)
WINDOWS = (1, 3, 8, 20)


def _roll(x, seconds):
    k = max(1, int(seconds * FPS / 2))
    c = np.vstack([np.zeros(x.shape[1]), np.cumsum(x, axis=0)])
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - k), np.minimum(len(x), i + k + 1)
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


def context(p, blocks):
    """Columns come in (H, A, D) triples so team swap is a fixed permutation."""
    out = np.full((len(p), 3 * (1 + len(OFFSETS) + len(WINDOWS))), np.nan, np.float32)
    for a, z in blocks:
        q = p[a:z]
        n = len(q)
        cols = [q]
        for off in OFFSETS:
            j = np.clip(np.arange(n) + int(off * FPS), 0, n - 1)
            cols.append(q[j])
        for w in WINDOWS:
            cols.append(_roll(q, w))
        out[a:z] = np.column_stack(cols)
    return out


def swap(x):
    idx = np.arange(x.shape[1]).reshape(-1, 3)[:, [1, 0, 2]].ravel()
    return x[:, idx]


def train(name, harness):
    """Fit the second stage on out-of-fold first-stage probabilities of the labelled matches."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    H = harness
    xs, ys = [], []
    for m in H.MATCHES:
        _, _, _, r, blocks, _ = H.load_match(m)
        for c in H.EVAL_CONDS:
            x = context(np.load(H.PROBS / name / f'{m}_{c}.npy').astype(float), blocks)[::5]
            y = r[::5]
            xs += [x, swap(x)]
            ys += [y, np.where(y < 2, 1 - y, y)]
    model = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, learning_rate=.08,
                                           min_samples_leaf=100, l2_regularization=1., random_state=0)
    return model.fit(np.concatenate(xs), np.concatenate(ys))


def apply(model, p, blocks):
    x = context(np.asarray(p, float), blocks)
    out = np.full((len(p), 3), 1 / 3)
    ok = np.isfinite(x).all(axis=1)
    out[ok] = (model.predict_proba(x[ok]) + model.predict_proba(swap(x[ok]))[:, [1, 0, 2]]) / 2
    return out
