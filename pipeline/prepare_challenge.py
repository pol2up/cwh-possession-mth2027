"""Cache the challenge tracking and the time-aligned broadcast validity flag (label-free inputs)."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
from cw import harness as H
from games import GAME, GAMES, asset_paths
from solution.io import load_tracking

# broadcast frame + offset = tactical frame, per half (derived by camera/clock_offsets.py)
OFFSETS = ((4014, (11185, 90652)), (26920, (110808, 190198)))

paths = asset_paths()
game = GAMES[GAME]
p, b, t = load_tracking(paths['atd_meta'], paths['atd_raw'], game['n_video_frames'], game['pitch'])
np.savez_compressed(H.HERE / 'challenge_tracking.npz', players=p, ball=b, teams=t)
print('tracking', p.shape)
h = pd.read_csv(asset_paths(source='broadcast')['homography'], usecols=['frame', 'valid'])
bv = np.full(len(b), -1)
for off, (a, z) in OFFSETS:
    tt = h['frame'].values + off
    m = (tt >= a) & (tt < z)
    bv[tt[m]] = h['valid'].values[m]
np.save(H.HERE / 'broadcast_valid.npy', bv)
print('broadcast validity known on', float((bv >= 0).mean()))
