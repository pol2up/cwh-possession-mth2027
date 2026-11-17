"""Second-half pre-kick-off rule.

The organisers' FAQ ("Which frames are actually graded?") states that the second half is graded from
video frame 110,808 but does not kick off until 113,742, and that those 2,934 frames are all `D`.
This step writes `D` on exactly those frames of a finished file.  Nothing is fitted; the two frame
numbers are the published ones.

Usage: python camera/prekickoff.py     out/v6_cam.csv -> out/possession_final.csv
"""
import hashlib, sys
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from cw import challenge as C

GRADED_FROM, KICK_OFF = 110808, 113742
src, dst = 'v6_cam', 'possession_final'
codes = pd.read_csv(C.OUT / f'{src}.csv')['label'].map(dict(H=0, A=1, D=2, X=3)).values.astype(np.int8)
before = codes.copy()
assert C.BLOCKS[1][0] == GRADED_FROM
codes[GRADED_FROM:KICK_OFF] = 2
path = C.write(codes, dst)
w = before[GRADED_FROM:KICK_OFF]
runs = np.flatnonzero(np.diff(np.r_[0, (w < 2).astype(int), 0]) == 1)
print(path, hashlib.sha256(path.read_bytes()).hexdigest())
print(f'frames changed {int((before != codes).sum())}; possessions removed or shortened before kick-off: {len(runs)}; '
      f'possessions {C.describe(before)["half2"]["n_poss"]} -> {C.describe(codes)["half2"]["n_poss"]} in half 2; '
      f'first label at kick-off: {"HAD"[codes[KICK_OFF]]}')
