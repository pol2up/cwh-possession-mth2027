"""Ball-availability simulation on fully observed other-game tracking.

realistic : state-dependent - the automatic tracker mostly loses the ball around stoppages (long
            runs) and sometimes for several seconds in live play.  Parameters are randomised so
            a model cannot lean on one assumed rate.
"""
import numpy as np

FPS = 25


def runs(a):
    idx = np.flatnonzero(np.r_[True, a[1:] != a[:-1]])
    ends = np.r_[idx[1:], len(a)]
    return [(a[s], int(s), int(e)) for s, e in zip(idx, ends)]


def realistic(ball, ref, blocks, seed, q=None, live_share=None):
    rng = np.random.default_rng(seed)
    out = ball.copy()
    for a, z in blocks:
        qq = rng.uniform(.5, .95) if q is None else q
        ls = rng.uniform(.04, .30) if live_share is None else live_share
        hide = np.zeros(z - a, bool)
        r = ref[a:z]
        for label, s, e in runs(r):
            if label != 2 or rng.random() > qq:
                continue
            s2 = s + int(rng.uniform(-1., 4.) * FPS)
            e2 = e + int(rng.uniform(-3., 1.5) * FPS)
            if e2 - s2 > FPS:
                hide[max(0, s2):max(0, min(z - a, e2))] = True
        live = np.flatnonzero(r < 2)
        target = ls * len(live)
        hidden_live = int((hide & (r < 2)).sum())
        guard = 0
        while hidden_live < target and guard < 500:
            guard += 1
            dur = int(np.clip(rng.lognormal(np.log(7.), .7), 1.5, 45.) * FPS)
            s = int(live[rng.integers(len(live))])
            e = min(z - a, s + dur)
            hidden_live += int(((r[s:e] < 2) & ~hide[s:e]).sum())
            hide[s:e] = True
        out[a:z][hide] = np.nan
    return out


def noisy(players, ball, seed):
    rng = np.random.default_rng(seed)
    p = players.copy()
    b = ball.copy()
    p[:, :, :2] += rng.normal(0, .2, p[:, :, :2].shape)
    b[:, :2] += rng.normal(0, .35, b[:, :2].shape)
    drop = rng.random(((len(p) + 9) // 10, p.shape[1])) < .05
    p[np.repeat(drop, 10, axis=0)[:len(p)]] = np.nan
    return p, b


def dead_motion(players, ball, ref, blocks, seed):
    """Replace the ball during dead periods with plausible movement.

    Sportec parks the ball at one coordinate for every dead frame (speed exactly 0 on 100% of
    them), which lets a classifier learn 'frozen ball = dead'.  Real automatic tracking shows a
    dead ball rolling on, being returned and carried to the restart.  Here the ball rolls out,
    travels to the player who takes the restart, stays with him, and is sometimes set down.
    """
    rng = np.random.default_rng(seed)
    out = ball.copy()
    n = len(ball)
    for a, z in blocks:
        for label, s, e in runs(ref[a:z]):
            if label != 2:
                continue
            s += a
            e += a
            m = e - s
            if m < 3:
                continue
            if s - 2 >= a and np.isfinite(ball[s - 1, 0]) and np.isfinite(ball[s - 2, 0]):
                o = ball[s - 1, :2].copy()
                v0 = (ball[s - 1, :2] - ball[s - 2, :2]) * FPS
            else:
                o = ball[s, :2].copy()
                v0 = np.zeros(2)
            if not np.isfinite(o).all():
                continue
            if e < z and np.isfinite(ball[e, 0]):
                target = ball[e, :2]
                d = np.linalg.norm(players[e, :, :2] - target, axis=1)
                taker = int(np.nanargmin(d)) if np.isfinite(d).any() else None
            else:
                taker = None
            path = np.empty((m, 2))
            # phase 1: roll on, decaying
            t1 = int(min(rng.uniform(.5, 4.) * FPS, .4 * m))
            tau = rng.uniform(.5, 1.5)
            tt = np.arange(1, m + 1) / FPS
            roll = o + v0[None] * (tau * (1 - np.exp(-tt / tau)))[:, None]
            path[:] = roll[min(t1, m - 1)]
            path[:t1] = roll[:t1]
            if taker is not None:
                tp = players[s:e, taker, :2].copy()
                bad = ~np.isfinite(tp).all(axis=1)
                if bad.all():
                    tp[:] = ball[e, :2]
                elif bad.any():
                    idx = np.arange(m)
                    for k in (0, 1):
                        tp[bad, k] = np.interp(idx[bad], idx[~bad], tp[~bad, k])
                tp = tp + rng.normal(0, .25, 2) + np.cumsum(rng.normal(0, .02, (m, 2)), axis=0)
                # phase 2: travel to the taker
                speed = rng.uniform(3., 12.)
                start = path[min(t1, m - 1)].copy()
                wait = int(rng.uniform(0, .3) * (m - t1))
                k = t1 + wait
                pos = start
                while k < m:
                    gap = tp[k] - pos
                    dist = np.linalg.norm(gap)
                    step = speed / FPS
                    if dist <= step:
                        break
                    pos = pos + gap / dist * step
                    path[k] = pos
                    k += 1
                path[t1:t1 + wait] = start
                # phase 3: with the taker
                path[k:] = tp[k:]
                # sometimes set down on the restart spot shortly before play resumes
                if rng.random() < .5:
                    hold = int(min(rng.uniform(.5, 4.) * FPS, m - k))
                    if hold > 0:
                        path[m - hold:] = ball[e, :2]
            out[s:e, :2] = path + rng.normal(0, .02, (m, 2))
    return out
