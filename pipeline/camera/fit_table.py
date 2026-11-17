"""Fit the camera-view table of camera_channel.json on SkillCorner Open Data (no provided-match data).

For each of the 15 development matches (commit 4340d27 of github.com/SkillCorner/opendata):
  camera state : 1 if the provider gives a pitch projection of the image corners for the frame, 0 if
                 that projection is empty (replay, close-up), unknown if the frame is missing
  reference    : live while the provider assigns possession to a team; dead in gaps between the
                 provider's phases of play that carry a game-interruption marker; otherwise unknown
Both are moved from 10 Hz to 25 Hz by nearest sample.  Frames are counted per class (live / dead),
camera state and length of the run of that state (<=1, 1-3, 3-8, 8-20, >20 s).  The table is
    log((dead_cell + 1) / (dead_total + 10)) - log((live_cell + 1) / (live_total + 10)).
Its weight in the build (1.0) was the best of 0, 0.25, 0.5 and 1 on the same 15 matches and was then
confirmed on the 5 other matches; that comparison needs a possession model for SkillCorner tracking
and is not part of this repository.

Usage: python camera/fit_table.py      expects data/external/skillcorner/<match>/<match>_match.json,
       _tracking_extrapolated.jsonl, _dynamic_events.csv and _phases_of_play.csv
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from cw.features import REPO  # noqa: E402

MATCHES = ('1874553', '1886347', '1899585', '1925299', '1927964', '1953632', '1959846', '1986691',
           '1996435', '1996436', '2006229', '2006363', '2007448', '2007721', '2010085')
EDGES = [1., 3., 8., 20.]


def read_match(folder, match):
    """Native 10 Hz camera state (0 / 1 / 3 unknown), reference (0 live / 2 dead / 3 unknown), periods."""
    meta = json.loads((folder / f'{match}_match.json').read_text())
    periods = sorted(meta['match_periods'], key=lambda p: p['period'])
    blocks = [(int(p['start_frame']), int(p['end_frame']) + 1) for p in periods]
    by_period = {int(p['period']): b for p, b in zip(periods, blocks)}
    n = blocks[-1][1]
    view = np.full(n, 3, np.int8)
    team = np.zeros(n, bool)
    covered = np.zeros(n, bool)
    with (folder / f'{match}_tracking_extrapolated.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['period'] is None:
                continue
            f = row['frame']
            corners = row['image_corners_projection']
            view[f] = int(corners is not None and any(v is not None for v in corners.values()))
            team[f] = row['possession']['group'] is not None
            covered[f] = True
    with (folder / f'{match}_phases_of_play.csv').open() as stream:
        phases = list(csv.DictReader(stream))
    with (folder / f'{match}_dynamic_events.csv').open() as stream:
        events = list(csv.DictReader(stream))
    after = {(int(e['period']), e.get('phase_index', '')) for e in events if e.get('game_interruption_after')}
    before = {(int(e['period']), e.get('phase_index', '')) for e in events if e.get('game_interruption_before')}
    marker, in_phase = np.zeros(n, bool), np.zeros(n, bool)
    for e in events:
        if e['event_type'] == 'game_interruption' and int(e['frame_end']) > int(e['frame_start']):
            marker[max(0, int(e['frame_start'])):min(n, int(e['frame_end']))] = True
    for period, (a, z) in by_period.items():
        ph = sorted((p for p in phases if int(p['period']) == period), key=lambda p: (int(p['frame_start']), int(p['frame_end'])))
        for p in ph:
            in_phase[max(a, int(p['frame_start'])):min(z, int(p['frame_end']))] = True
        for prev, nxt in zip(ph, ph[1:]):
            start, end = max(a, int(prev['frame_end'])), min(z, int(nxt['frame_start']))
            if end > start and ((period, prev['index']) in after or (period, nxt['index']) in before):
                marker[start:end] = True
    ref = np.where(team, 0, 3).astype(np.int8)
    ref[~team & marker & ~in_phase & covered] = 2
    ref[~covered] = 3
    return view, ref, blocks


def to_25hz(x, blocks):
    out = np.full((len(x) * 5 + 1) // 2, 3, np.int8)
    for a, z in blocks:
        dest = np.arange((a * 5 + 1) // 2, (z * 5 + 1) // 2)
        nearest = (dest * 4 + 5) // 10
        ok = (nearest >= a) & (nearest < z)
        out[dest[ok]] = x[nearest[ok]]
    return out


def run_buckets(view, blocks, fps=25):
    """Per frame: bucket of the length of the run of the same known camera state; 255 if unknown."""
    out = np.full(len(view), 255, np.uint8)
    for a, z in blocks:
        x = view[a:z]
        edges = np.r_[0, np.flatnonzero(x[1:] != x[:-1]) + 1, len(x)]
        for l, r in zip(edges[:-1], edges[1:]):
            if x[l] != 3:
                out[a + l:a + r] = np.searchsorted(EDGES, (r - l) / fps, side='left')
    return out


def counts(match, root):
    view, ref, blocks = read_match(root / match, match)
    blocks25 = [((a * 5 + 1) // 2, (z * 5 + 1) // 2) for a, z in blocks]
    view, ref = to_25hz(view, blocks), to_25hz(ref, blocks)
    bucket = run_buckets(view, blocks25)
    ok = (view < 2) & (bucket < 5) & (ref < 3)
    c = np.zeros((2, 2, 5), np.int64)          # class (live, dead) x camera state x run-length bucket
    np.add.at(c, ((ref[ok] == 2).astype(int), view[ok], bucket[ok]), 1)
    return c


def fit_table(c):
    p = (c + 1.) / (c + 1.).sum(axis=(1, 2), keepdims=True)
    return np.log(p[1]) - np.log(p[0])


if __name__ == '__main__':
    root = REPO / 'data/external/skillcorner'
    total = sum(counts(m, root) for m in MATCHES)
    table = fit_table(total)
    committed = np.array(json.loads((HERE / 'camera_channel.json').read_text())['log_ratio'])
    print(np.round(table, 6))
    print('largest difference from camera_channel.json:', float(np.abs(table - committed).max()))
