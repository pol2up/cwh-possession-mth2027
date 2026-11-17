"""First-stage training rows from the Metrica sample games: native ball, ball hidden mainly around
stoppages, and the latter with positional noise and player drop-outs."""
import sys, numpy as np, warnings
warnings.simplefilter('ignore')
sys.path.insert(0, '.')
from cw import harness as H, features as F, masks as M, metrica


def train_rows(g):
    f = H.CACHE / f'metrica{g}_train.npz'
    if not f.exists():
        p, b, t, r, blocks, pitch = metrica.load(g)
        xs, ys = [], []
        pn, bn = M.noisy(p, b, 70 + g)
        for pp, ball in ((p, b),
                         (p, M.realistic(M.dead_motion(p, b, r, blocks, 90 + g), r, blocks, 95 + g)),
                         (pn, M.realistic(bn, r, blocks, 97 + g, q=1., live_share=None))):
            x, names = F.build(pp, ball, t, blocks, pitch, H.ALL_SETS)
            ok = r[::5] < 3
            xs.append(x[::5][ok]); ys.append(r[::5][ok])
        np.savez_compressed(f, x=np.concatenate(xs), y=np.concatenate(ys))
    d = np.load(f)
    return d['x'], d['y']
