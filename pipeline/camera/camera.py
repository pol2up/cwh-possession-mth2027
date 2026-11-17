"""Camera-view channel on the challenge match.

The table (log-likelihood ratio dead / live per camera state and run length) and its weight were
fitted on 15 SkillCorner Open Data matches and confirmed on 5 others (see the technical document).
Nothing is fitted here: the broadcast's homography `valid` flag, time-aligned to the tracking
(prepare_challenge.py --with-broadcast), is turned into the same feature and looked up in that table.
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CH = json.loads((HERE / 'camera_channel.json').read_text())
TABLE, WEIGHT, EDGES = np.array(CH['log_ratio']), float(CH['weight']), CH['bins_seconds']


def features(view, blocks, fps=25):
    """view: 1 pitch in view, 0 not, anything else unknown.  Returns (state, run-length bucket), -1 unknown."""
    out = np.full((len(view), 2), -1, np.int8)
    for a, z in blocks:
        x = view[a:z]
        edges = np.r_[0, np.flatnonzero(x[1:] != x[:-1]) + 1, len(x)]
        for l, r in zip(edges[:-1], edges[1:]):
            if x[l] in (0, 1):
                out[a + l:a + r] = (x[l], np.searchsorted(EDGES, (r - l) / fps, side='left'))
    return out


def fuse(p, feats, weight=WEIGHT):
    ok = feats[:, 0] >= 0
    ratio = np.zeros(len(p))
    ratio[ok] = TABLE[feats[ok, 0], feats[ok, 1]]
    s = np.log(np.clip(p, 1e-12, 1)).astype(float)
    s[:, 2] += weight * ratio
    s -= s.max(axis=1, keepdims=True)
    q = np.exp(s)
    return q / q.sum(axis=1, keepdims=True)
