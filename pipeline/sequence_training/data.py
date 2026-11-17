"""Training sequences of the Metrica sample games for the sequence model (5 Hz), three ball variants."""
import sys, warnings
from pathlib import Path
import numpy as np
warnings.simplefilter('ignore')
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cw import harness as H, features as F, masks as M, metrica, seq as S  # noqa: E402

CACHE = Path(__file__).resolve().parent / 'cache'
CACHE.mkdir(exist_ok=True)
VARIANTS = ('native', 'real', 'noisyreal')


def metrica_seqs(g):
    """[(x_5hz, y_5hz, w_5hz, blocks_5hz)] for one game; same recipes and seeds as metrica_train.train_rows."""
    f = CACHE / f'metrica{g}_seqs.npz'
    if not f.exists():
        p, b, t, r, blocks, pitch = metrica.load(g)
        pn, bn = M.noisy(p, b, 70 + g)
        xs = []
        for pp, ball in ((p, b), (p, M.realistic(M.dead_motion(p, b, r, blocks, 90 + g), r, blocks, 95 + g)),
                         (pn, M.realistic(bn, r, blocks, 97 + g, q=1., live_share=None))):
            x, names = F.build(pp, ball, t, blocks, pitch, H.ALL_SETS)
            assert names == H.names()
            xs.append(x[::H.STEP].astype(np.float32))
        np.savez_compressed(f, x=np.stack(xs), y=r[::H.STEP], w=S.spell_weights(r)[::H.STEP],
                            blocks=np.array([((s + H.STEP - 1) // H.STEP, e // H.STEP) for s, e in blocks]))
    d = np.load(f)
    b5 = [tuple(x) for x in d['blocks'].tolist()]
    return [(d['x'][k], d['y'], d['w'], b5) for k in range(len(VARIANTS))]


def sportec_seqs(m):
    _, _, _, r, blocks, _ = H.load_match(m)
    y, w = r[::H.STEP], S.spell_weights(r)[::H.STEP]
    b5 = [((s + H.STEP - 1) // H.STEP, e // H.STEP) for s, e in blocks]
    return [(np.load(H.CACHE / f'{m}_tr_{c}.npy'), y, w, b5) for c in ('vis', 'real1', 'real2', 'noisyreal')]


if __name__ == '__main__':
    for g in (1, 2, 3):
        s = metrica_seqs(g)
        print(g, [x[0].shape for x in s], np.bincount(s[0][1], minlength=4), flush=True)
