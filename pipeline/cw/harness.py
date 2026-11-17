"""First-stage training data and out-of-fold predictions on the seven labelled Sportec matches.

prepare : cache features for every match and ball-visibility condition
fit     : train on six matches, predict the seventh (all folds); the second stage is trained on these
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cw.features import REPO  # noqa: E402
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from cw import features as F, masks as M  # noqa: E402
from solution.model import decode  # noqa: E402

MATCHES = ['J03WMX', 'J03WN1', 'J03WOH', 'J03WOY', 'J03WPY', 'J03WQQ', 'J03WR9']
CACHE = HERE / 'cache2'
ALL_SETS = ('base', 'gap', 'shape', 'traj')
TRAIN_CONDS = ('vis', 'real1', 'real2', 'noisyreal')
EVAL_CONDS = ('vis', 'real')
STEP = 5
PROBS = CACHE / 'probs'


def load_match(m):
    with np.load(REPO / f'data/external/sportec/{m}_parsed.npz') as d:
        blocks = [tuple(x) for x in d['blocks'].tolist()]
        return (d['players'], d['ball'], d['teams'], d['reference'],
                blocks, tuple(d['pitch'].tolist()))


def observe(cond, i, p, b, r, blocks):
    """Every condition first replaces the frozen dead ball with synthetic dead-ball movement."""
    seeds = {'vis': 100, 'real1': 300, 'real2': 400, 'noisyreal': 500, 'ev_vis': 700, 'ev_real': 900}
    sd = seeds[cond] + i
    b = M.dead_motion(p, b, r, blocks, sd)
    if cond in ('vis', 'ev_vis'):
        return p, b
    if cond in ('real1', 'real2'):
        return p, M.realistic(b, r, blocks, sd + 1000)
    if cond == 'noisyreal':
        pn, bn = M.noisy(p, b, sd + 2000)
        return pn, M.realistic(bn, r, blocks, sd + 1000)
    if cond == 'ev_real':
        return p, M.realistic(b, r, blocks, sd + 1000, q=.8, live_share=.15)
    raise ValueError(cond)


def _prepare_one(task):
    i, m, kind, c = task
    out = CACHE / f'{m}_{kind}_{c}.npy'
    if out.exists():
        return None
    p, b, t, r, blocks, pitch = load_match(m)
    pp, bb = observe(c if kind == 'tr' else 'ev_' + c, i, p, b, r, blocks)
    x, names = F.build(pp, bb, t, blocks, pitch, ALL_SETS)
    np.save(out, x[::STEP] if kind == 'tr' else x)
    return names


def prepare():
    from multiprocessing import Pool
    CACHE.mkdir(exist_ok=True)
    tasks = [(i, m, kind, c) for i, m in enumerate(MATCHES)
             for kind, conds in (('tr', TRAIN_CONDS), ('ev', EVAL_CONDS)) for c in conds]
    with Pool(7) as pool:
        for names in pool.imap_unordered(_prepare_one, tasks):
            if names:
                (CACHE / 'names.json').write_text(json.dumps(names))


def names():
    return json.loads((CACHE / 'names.json').read_text())


def select(sets):
    """Column indices of the requested feature sets inside the cached 'all' matrix."""
    d = np.load(HERE / 'challenge_tracking.npz')
    p, b, t = d['players'][:600], d['ball'][:600], d['teams']
    all_names = names()
    keep = []
    for s in sets:
        _, nm = F.build(p, b, t, [(0, 600)], (105.3, 68.), (s,))
        keep += [all_names.index(n) for n in nm]
    return keep


CONFIGS = {
    # all four feature sets, 300 iterations, the three Metrica sample games added to the Sportec matches
    'all300m3_ni': dict(sets=ALL_SETS, conds=TRAIN_CONDS, iters=300, metrica=(1, 2, 3)),
}


def fit(name, folds=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    cfg = CONFIGS[name]
    cols = select(cfg['sets'])
    nm = [names()[i] for i in cols]
    out = PROBS / name
    out.mkdir(parents=True, exist_ok=True)
    refs = {m: load_match(m)[3] for m in MATCHES}
    for test in (folds or MATCHES):
        if all((out / f'{test}_{c}.npy').exists() for c in EVAL_CONDS):
            continue
        t0 = time.time()
        xs, ys = [], []
        for m in MATCHES:
            if m == test:
                continue
            y = refs[m][::STEP]
            ok = y < 3
            for c in cfg['conds']:
                x = np.load(CACHE / f'{m}_tr_{c}.npy')[:, cols][ok][::cfg.get('sub', 2)]
                yy = y[ok][::cfg.get('sub', 2)]
                xs += [x, F.swap(x, nm)]
                ys += [yy, np.where(yy < 2, 1 - yy, yy)]
        if cfg.get('metrica'):
            import metrica_train
            for g in cfg['metrica']:
                xm, ym = metrica_train.train_rows(g)
                xm = xm[:, cols]
                xs += [xm, F.swap(xm, nm)]
                ys += [ym, np.where(ym < 2, 1 - ym, ym)]
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        model = HistGradientBoostingClassifier(
            max_iter=cfg['iters'], max_leaf_nodes=31, learning_rate=.08, min_samples_leaf=30,
            l2_regularization=1., class_weight='balanced', early_stopping=False, random_state=2027)
        model.fit(x, y)
        del x, xs
        for c in EVAL_CONDS:
            xe = np.load(CACHE / f'{test}_ev_{c}.npy')[:, cols]
            pa = model.predict_proba(xe)
            pb = model.predict_proba(F.swap(xe, nm))
            np.save(out / f'{test}_{c}.npy', ((pa + pb[:, [1, 0, 2]]) / 2).astype(np.float32))
        print(name, test, 'rows', len(y), f'{time.time() - t0:.0f}s', flush=True)


def decode_probs(p, blocks, cost=.6, min_team=.4, min_dead=.6, cost_td=None, prior=None):
    scores = np.log(np.clip(p, 1e-5, 1))
    if prior is not None:
        scores = scores + np.asarray(prior)
    td = cost if cost_td is None else cost_td
    pen = np.array([[0, cost, td], [cost, 0, td], [td, td, 0]], float)
    mins = (max(1, round(min_team * 25)),) * 2 + (max(1, round(min_dead * 25)),)
    out = np.full(len(p), 3, np.int8)
    for a, z in blocks:
        out[a:z] = decode(scores[a:z] / 25, pen, mins)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd')
    ap.add_argument('names', nargs='*')
    ap.add_argument('--folds', nargs='*')
    a = ap.parse_args()
    if a.cmd == 'prepare':
        prepare()
    elif a.cmd == 'fit':
        for n in a.names:
            fit(n, a.folds)
