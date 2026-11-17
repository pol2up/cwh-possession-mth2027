"""Camera-view channel on top of the base file: same models, blend and decoder; the blended
probabilities get the externally fitted camera log-ratio on the dead-ball class before decoding.

Usage: python camera/build.py          out/v6.csv -> out/v6_cam.csv
Needs make_final.py (it caches the stage probabilities) and prepare_challenge.py to have been run.
"""
import hashlib, sys, warnings
import numpy as np, pandas as pd
warnings.simplefilter('ignore'); sys.path.insert(0, '.')
from cw import harness as H, challenge as C, seq as S
from camera import camera as K

BASE, OUT, COST = 'v6', 'v6_cam', .3
decode = lambda p: H.decode_probs(p, C.BLOCKS, cost=COST)

p = S.blend(np.load(H.CACHE / f'final_{BASE}_stack.npy').astype(float), np.load(H.CACHE / f'final_{BASE}_seq.npy').astype(float), .5)
base = decode(p)
frozen = pd.read_csv(C.OUT / f'{BASE}.csv')['label'].map(dict(H=0, A=1, D=2, X=3)).values
assert (base == frozen).all(), f'cached probabilities do not reproduce the {BASE} file'
codes = decode(K.fuse(p, K.features(np.load('broadcast_valid.npy'), C.BLOCKS)))
path = C.write(codes, OUT)
g = np.r_[C.BLOCKS[0][0]:C.BLOCKS[0][1], C.BLOCKS[1][0]:C.BLOCKS[1][1]]
print(path, hashlib.sha256(path.read_bytes()).hexdigest())
print('possessions per half', [v['n_poss'] for v in C.describe(codes).values()], '| frames changed by the camera channel', round(float((codes[g] != base[g]).mean()), 4))
