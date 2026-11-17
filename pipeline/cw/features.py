"""Feature sets for the possession classifier.

base   = the 55 base features of experiments/learned_features.py
gap    = what the ball's visible/missing pattern and its last/next sighting say
shape  = attack-direction-normalised team shape and collective movement
traj   = ball trajectory cues on visible frames (touches, boundary distance)

Naming: 'home_*'/'away_*' columns are exchanged under team swap; 'flip_*' columns are
negated under team swap (they are signed home-minus-away quantities).
"""
import sys
from pathlib import Path

import numpy as np

def _find_repo(start):
    for parent in [start, *start.parents]:
        if (parent / 'evaluation' / 'possession_metrics.py').exists():
            return parent
    return start.parent / 'repo'

REPO = _find_repo(Path(__file__).resolve().parents[1])
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.learned_features import control_features  # noqa: E402
from solution.model import rolling_mean  # noqa: E402

FPS = 25


def swap(features, names):
    other = [n.replace('home_', 'away_', 1) if n.startswith('home_') else
             n.replace('away_', 'home_', 1) if n.startswith('away_') else n for n in names]
    out = np.asarray(features)[:, [names.index(n) for n in other]].copy()
    flip = [i for i, n in enumerate(names) if n.startswith('flip_')]
    out[:, flip] *= -1
    return out


def _ffill(values, valid):
    idx = np.where(valid, np.arange(len(valid)), -1)
    idx = np.maximum.accumulate(idx)
    out = values[np.maximum(idx, 0)].astype(float)
    out[idx < 0] = np.nan
    return out


def _bfill(values, valid):
    return _ffill(values[::-1], valid[::-1])[::-1]


def _roll(x, seconds):
    return rolling_mean(x, max(1, round(seconds * FPS / 2)))


def _edge(b, pitch):
    return np.minimum.reduce([b[:, 0], pitch[0] - b[:, 0], b[:, 1], pitch[1] - b[:, 1]])


def _near(p, b, t):
    d = np.linalg.norm(p[:, :, :2] - b[:, None, :2], axis=2)
    d[~np.isfinite(d)] = np.inf
    return np.column_stack([d[:, t == k].min(axis=1) for k in (0, 1)])


def gap_features(p, b, t, blocks, pitch):
    n = len(b)
    vis = np.isfinite(b[:, :2]).all(axis=1)
    near = _near(p, b, t)
    near[~vis] = np.nan
    edge = np.where(vis, _edge(b, pitch), np.nan)
    v = np.full((n, 2), np.nan)
    v[1:-1] = (b[2:, :2] - b[:-2, :2]) * FPS / 2
    spd = np.linalg.norm(v, axis=1)
    cols = {k: np.full(n, np.nan) for k in (
        'gap_since_visible', 'gap_until_visible', 'gap_length', 'gap_position',
        'vis_run_length', 'vis_since_start', 'vis_until_end',
        'last_edge', 'next_edge', 'last_speed', 'next_speed', 'gap_displacement',
        'flip_last_near_diff', 'flip_next_near_diff', 'last_near_min', 'next_near_min',
        'next_still_2s', 'last_out_speed', 'ball_edge', 'missing_share_10s', 'missing_share_30s')}
    for a, z in blocks:
        s = slice(a, z)
        m = z - a
        vv = vis[s]
        i = np.arange(m)
        last = np.maximum.accumulate(np.where(vv, i, -10 ** 7))
        nxt = np.minimum.accumulate(np.where(vv, i, 10 ** 7)[::-1])[::-1]
        since = np.minimum((i - last) / FPS, 60.)
        until = np.minimum((nxt - i) / FPS, 60.)
        cols['gap_since_visible'][s] = since
        cols['gap_until_visible'][s] = until
        length = np.where(vv, 0, np.minimum(since + until, 120.))
        cols['gap_length'][s] = length
        cols['gap_position'][s] = np.where(vv, np.nan, since / np.maximum(since + until, 1e-6))
        # visible-run geometry: the automatic ball track mostly lives in long runs
        lastm = np.maximum.accumulate(np.where(~vv, i, -10 ** 7))
        nxtm = np.minimum.accumulate(np.where(~vv, i, 10 ** 7)[::-1])[::-1]
        vs = np.minimum((i - lastm) / FPS, 120.)
        vu = np.minimum((nxtm - i) / FPS, 120.)
        cols['vis_since_start'][s] = np.where(vv, vs, np.nan)
        cols['vis_until_end'][s] = np.where(vv, vu, np.nan)
        cols['vis_run_length'][s] = np.where(vv, np.minimum(vs + vu, 240.), 0)
        okv = vv & np.isfinite(spd[s])
        diff = np.clip(near[s, 1], 0, 30) - np.clip(near[s, 0], 0, 30)   # + = home closer
        for name, src, ok in (('edge', edge[s], vv), ('speed', np.minimum(spd[s], 40), okv),
                              ('near_min', np.minimum(near[s].min(axis=1), 30), vv)):
            cols['last_' + name][s] = _ffill(src, ok)
            cols['next_' + name][s] = _bfill(src, ok)
        cols['flip_last_near_diff'][s] = _ffill(diff, vv)
        cols['flip_next_near_diff'][s] = _bfill(diff, vv)
        bx = b[s, :2]
        lastpos = np.column_stack([_ffill(bx[:, k], vv) for k in (0, 1)])
        nextpos = np.column_stack([_bfill(bx[:, k], vv) for k in (0, 1)])
        cols['gap_displacement'][s] = np.where(vv, 0, np.linalg.norm(nextpos - lastpos, axis=1))
        # was the ball heading out when last seen?  outward velocity component at the nearest line
        c = np.array([pitch[0] / 2, pitch[1] / 2])
        rel = (bx - c) / c
        axis_x = np.abs(rel[:, 0]) > np.abs(rel[:, 1])
        outward = np.where(axis_x, np.sign(rel[:, 0]) * v[s, 0], np.sign(rel[:, 1]) * v[s, 1])
        cols['last_out_speed'][s] = _ffill(np.clip(outward, -30, 30), okv)
        # is the ball lying still once it is seen again (set on the spot for a restart)?
        still = _roll(np.where(okv, (spd[s] < .5).astype(float), np.nan), 2)
        fwd = np.full(m, np.nan)
        fwd[:m - 25] = still[25:]
        cols['next_still_2s'][s] = _bfill(fwd, vv & np.isfinite(fwd))
        cols['ball_edge'][s] = edge[s]
        miss = (~vv).astype(float)
        cols['missing_share_10s'][s] = _roll(miss, 10)
        cols['missing_share_30s'][s] = _roll(miss, 30)
    names = list(cols)
    return np.column_stack([cols[k] for k in names]), names


def attack_signs(p, t, blocks, pitch):
    """+1 if home attacks towards +x in the block, else -1.

    Uses each team's deepest player (its goalkeeper) over the whole half; the opening seconds
    are unreliable in automatic tracking.
    """
    import warnings
    signs = []
    for a, z in blocks:
        x = p[a:z:5, :, 0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lo = [np.nanmean(np.nanmin(np.where(np.isfinite(x[:, t == k]), x[:, t == k], np.nan), axis=1)) for k in (0, 1)]
            hi = [np.nanmean(np.nanmax(np.where(np.isfinite(x[:, t == k]), x[:, t == k], np.nan), axis=1)) for k in (0, 1)]
        signs.append(1. if (lo[0] - lo[1]) + (hi[0] - hi[1]) < 0 else -1.)
    return signs


def shape_features(p, b, t, blocks, pitch):
    n = len(p)
    L, W = pitch
    out = {}
    per = {}
    for k, side in ((0, 'home'), (1, 'away')):
        x = np.where(np.isfinite(p[:, t == k, 0]), p[:, t == k, 0], np.nan)
        y = np.where(np.isfinite(p[:, t == k, 1]), p[:, t == k, 1], np.nan)
        with np.errstate(all='ignore'):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                per[side] = dict(cx=np.nanmean(x, axis=1), cy=np.nanmean(y, axis=1),
                                 sx=np.nanstd(x, axis=1), sy=np.nanstd(y, axis=1),
                                 lo=np.nanpercentile(x, 15, axis=1), hi=np.nanpercentile(x, 85, axis=1),
                                 q50=np.nanmedian(x, axis=1))
    signs = attack_signs(p, t, blocks, pitch)
    for name in ('home_adv', 'away_adv', 'home_adv_v', 'away_adv_v', 'home_depth', 'away_depth',
                 'home_width', 'away_width', 'home_rear', 'away_rear', 'home_front', 'away_front',
                 'home_rear_v', 'away_rear_v', 'home_width_v', 'away_width_v',
                 'home_ball_adv', 'away_ball_adv', 'flip_centroid_gap', 'flip_adv_v_diff',
                 'ball_lateral', 'home_ball_ahead', 'away_ball_ahead'):
        out[name] = np.full(n, np.nan)
    for (a, z), sign in zip(blocks, signs):
        s = slice(a, z)
        for side, sg in (('home', sign), ('away', -sign)):
            d = per[side]
            # advancement in [0,1]: 0 = own goal line, 1 = opponent goal line
            adv = (d['cx'][s] / L) if sg > 0 else (1 - d['cx'][s] / L)
            rear = (d['lo'][s] / L) if sg > 0 else (1 - d['hi'][s] / L)
            front = (d['hi'][s] / L) if sg > 0 else (1 - d['lo'][s] / L)
            out[side + '_adv'][s] = adv
            out[side + '_rear'][s] = rear
            out[side + '_front'][s] = front
            out[side + '_depth'][s] = d['sx'][s] / L
            out[side + '_width'][s] = d['sy'][s] / W

            def vel(x, sec=2.):
                k = int(sec * FPS / 2)
                sm = _roll(x, 1.)
                o = np.full(len(x), np.nan)
                o[k:-k] = (sm[2 * k:] - sm[:-2 * k]) / sec
                return o
            out[side + '_adv_v'][s] = vel(adv) * L          # m/s towards the opponent goal
            out[side + '_rear_v'][s] = vel(rear) * L
            out[side + '_width_v'][s] = vel(d['sy'][s])
            bx = (b[s, 0] / L) if sg > 0 else (1 - b[s, 0] / L)
            out[side + '_ball_adv'][s] = bx
            out[side + '_ball_ahead'][s] = bx - adv
        out['flip_centroid_gap'][s] = out['home_adv'][s] - out['away_adv'][s]
        out['flip_adv_v_diff'][s] = out['home_adv_v'][s] - out['away_adv_v'][s]
        out['ball_lateral'][s] = np.abs(b[s, 1] / W - .5)
    names = list(out)
    base = np.column_stack([out[k] for k in names])
    cols = [base]
    allnames = list(names)
    dyn = [i for i, k in enumerate(names) if k.endswith('_v') or k == 'flip_adv_v_diff']
    for seconds in (3, 8):
        w = np.full((n, len(dyn)), np.nan)
        for a, z in blocks:
            for j, i in enumerate(dyn):
                w[a:z, j] = _roll(base[a:z, i], seconds)
        cols.append(w)
        allnames += [f'{names[i]}_mean{seconds}s' for i in dyn]
    return np.column_stack(cols), allnames


def traj_features(p, b, t, blocks, pitch):
    n = len(b)
    vis = np.isfinite(b[:, :2]).all(axis=1)
    near = _near(p, b, t)
    near[~vis] = np.nan
    out = {k: np.full(n, np.nan) for k in (
        'ball_acc', 'ball_turn', 'ball_speed_5', 'ball_speed_std_1s', 'touch_rate_3s',
        'home_touch_3s', 'away_touch_3s', 'home_touch_8s', 'away_touch_8s',
        'home_since_touch', 'away_since_touch', 'home_until_touch', 'away_until_touch',
        'home_close_3s', 'away_close_3s', 'home_close_10s', 'away_close_10s',
        'flip_touch_balance_15s', 'ball_still_3s')}
    for a, z in blocks:
        s = slice(a, z)
        m = z - a
        x = b[s, :2]
        k = 3
        v = np.full((m, 2), np.nan)
        v[k:-k] = (x[2 * k:] - x[:-2 * k]) * FPS / (2 * k)
        sp = np.linalg.norm(v, axis=1)
        acc = np.full(m, np.nan)
        acc[k:-k] = np.linalg.norm(v[2 * k:] - v[:-2 * k], axis=1) * FPS / (2 * k)
        turn = np.full(m, np.nan)
        dot = (v[2 * k:] * v[:-2 * k]).sum(axis=1)
        den = np.linalg.norm(v[2 * k:], axis=1) * np.linalg.norm(v[:-2 * k], axis=1)
        with np.errstate(all='ignore'):
            turn[k:-k] = np.where(den > 1., 1 - dot / den, 0.)
        out['ball_acc'][s] = np.minimum(acc, 150) / 150
        out['ball_turn'][s] = turn / 2
        out['ball_speed_5'][s] = np.minimum(sp, 40) / 40
        sq = _roll(np.minimum(sp, 40) ** 2, 1)
        mu = _roll(np.minimum(sp, 40), 1)
        out['ball_speed_std_1s'][s] = np.sqrt(np.maximum(sq - mu ** 2, 0)) / 10
        out['ball_still_3s'][s] = _roll(np.where(np.isfinite(sp), (sp < .6).astype(float), np.nan), 3)
        # a "touch": the ball changes velocity sharply while a player is within reach.
        # Thresholds are this half's own upper quantiles, so a smoother or noisier ball track
        # yields the same share of candidate touches.
        fa = acc[np.isfinite(acc)]
        ft = turn[np.isfinite(turn)]
        acc_thr = np.quantile(fa, .85) if len(fa) else np.inf
        turn_thr = max(np.quantile(ft, .85), 1e-3) if len(ft) else np.inf
        kick = (acc > acc_thr) | (turn > turn_thr)
        out['ball_acc'][s] = np.minimum(acc / max(acc_thr, 1e-6), 4) / 4
        idx = np.arange(m)
        for side, col in (('home', 0), ('away', 1)):
            mine = near[s, col]
            other = near[s, 1 - col]
            touch = kick & (mine < 2.5) & (mine + .5 < other)
            close = (mine < 1.5) & (mine + .75 < other)
            tf = np.where(vis[s], touch.astype(float), np.nan)
            cf = np.where(vis[s], close.astype(float), np.nan)
            out[f'{side}_touch_3s'][s] = _roll(tf, 3)
            out[f'{side}_touch_8s'][s] = _roll(tf, 8)
            out[f'{side}_close_3s'][s] = _roll(cf, 3)
            out[f'{side}_close_10s'][s] = _roll(cf, 10)
            last = np.maximum.accumulate(np.where(touch, idx, -10 ** 7))
            nxt = np.minimum.accumulate(np.where(touch, idx, 10 ** 7)[::-1])[::-1]
            out[f'{side}_since_touch'][s] = np.minimum((idx - last) / FPS, 20.)
            out[f'{side}_until_touch'][s] = np.minimum((nxt - idx) / FPS, 20.)
        out['touch_rate_3s'][s] = _roll(np.where(vis[s], kick.astype(float), np.nan), 3)
        h = _roll(np.where(vis[s], ((near[s, 0] < 2.5) & (near[s, 0] + .5 < near[s, 1])).astype(float), np.nan), 15)
        w = _roll(np.where(vis[s], ((near[s, 1] < 2.5) & (near[s, 1] + .5 < near[s, 0])).astype(float), np.nan), 15)
        out['flip_touch_balance_15s'][s] = h - w
    names = list(out)
    return np.column_stack([out[k] for k in names]), names


SETS = {'gap': gap_features, 'shape': shape_features, 'traj': traj_features}


def build(p, b, t, blocks, pitch, sets=('base',)):
    cols, names = [], []
    for name in sets:
        if name == 'base':
            x, nm = control_features(p, b, t, blocks, FPS, pitch)
        else:
            x, nm = SETS[name](p, b, t, blocks, pitch)
        cols.append(x.astype(np.float32))
        names += nm
    if len(set(names)) != len(names):
        raise ValueError('duplicate feature names')
    return np.column_stack(cols), names
