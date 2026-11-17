"""Train on all labelled other-game matches and label the challenge match."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cw import harness as H, features as F  # noqa: E402

BLOCKS = [(11185, 90652), (110808, 190198)]
PITCH = (105.3, 68.0)
N = 199319
OUT = HERE / 'out'


def tracking():
    d = np.load(HERE / 'challenge_tracking.npz')
    return d['players'], d['ball'], d['teams']


def features():
    f = H.CACHE / 'challenge_all.npy'
    if not f.exists():
        p, b, t = tracking()
        x, names = F.build(p, b, t, BLOCKS, PITCH, H.ALL_SETS)
        assert names == H.names()
        np.save(f, x)
    return np.load(f)


def train_full(name):
    from sklearn.ensemble import HistGradientBoostingClassifier
    cfg = H.CONFIGS[name]
    cols = H.select(cfg['sets'])
    nm = [H.names()[i] for i in cols]
    xs, ys = [], []
    for m in H.MATCHES:
        y = H.load_match(m)[3][::H.STEP]
        ok = y < 3
        for c in cfg['conds']:
            x = np.load(H.CACHE / f'{m}_tr_{c}.npy')[:, cols][ok]
            xs += [x, F.swap(x, nm)]
            ys += [y[ok], np.where(y[ok] < 2, 1 - y[ok], y[ok])]
    import metrica_train
    for x, y in (metrica_train.train_rows(g) for g in cfg['metrica']):
        x = x[:, cols]
        xs += [x, F.swap(x, nm)]
        ys += [y, np.where(y < 2, 1 - y, y)]
    model = HistGradientBoostingClassifier(
        max_iter=cfg['iters'], max_leaf_nodes=31, learning_rate=.08, min_samples_leaf=30,
        l2_regularization=1., class_weight='balanced', early_stopping=False, random_state=2027)
    model.fit(np.concatenate(xs), np.concatenate(ys))
    return model, cols, nm


def predict(model, cols, nm, x=None):
    x = features()[:, cols] if x is None else x[:, cols]
    pa = model.predict_proba(x)
    pb = model.predict_proba(F.swap(x, nm))
    return (pa + pb[:, [1, 0, 2]]) / 2


def write(codes, name):
    OUT.mkdir(exist_ok=True)
    path = OUT / f'{name}.csv'
    pd.DataFrame({'frame': np.arange(N), 'label': np.array(list('HADX'))[codes]}).to_csv(path, index=False)
    return path


def describe(codes):
    from cw.masks import runs
    out = {}
    for h, (a, z) in enumerate(BLOCKS):
        x = codes[a:z]
        poss = [(e - s) / 25 for l, s, e in runs(x) if l < 2]
        out[f'half{h + 1}'] = dict(share=[round(float((x == k).mean()), 3) for k in range(3)],
                                   n_poss=len(poss), mean_s=round(float(np.mean(poss)), 1),
                                   median_s=round(float(np.median(poss)), 1))
    return out
