"""H/A/D labels for Metrica's public sample games, derived from their event files.

Possession = team of the latest on-ball event; dead from a ball-out / foul / goal / offside until
the next restart event.  An approximation of Metrica's own possession semantics.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from cw.features import REPO
ROOT = REPO / 'data/external/metrica-sample'
ONBALL = {'PASS', 'SHOT', 'RECOVERY', 'BALL LOST', 'SET PIECE', 'CARRY'}


def labels(game):
    if game == 3:
        e, period = _events3()
        return _labels(e, period)
    d = ROOT / f'Sample_Game_{game}'
    e = pd.read_csv(d / f'Sample_Game_{game}_RawEventsData.csv')
    home = pd.read_csv(d / f'Sample_Game_{game}_RawTrackingData_Home_Team.csv', skiprows=2)
    return _labels(e, home.iloc[:, 0].values)


G3 = ROOT / 'Sample_Game_3'
G3_PERIODS = ((1, 69661), (69662, 143761))


def _events3():
    import json
    rows = []
    for x in json.loads((G3 / 'Sample_Game_3_events.json').read_text())['data']:
        st = x.get('subtypes')
        st = [] if not st else st if isinstance(st, list) else [st]
        rows.append({'Team': 'Home' if x['team']['id'] == 'FIFATMA' else 'Away', 'Type': x['type']['name'],
                     'Subtype': '-'.join(s['name'] for s in st) if st else 'nan', 'Period': x['period'],
                     'Start Frame': x['start']['frame'], 'End Frame': x['end']['frame']})
    period = np.zeros(G3_PERIODS[1][1], int)
    for k, (a, z) in enumerate(G3_PERIODS):
        period[a - 1:z] = k + 1
    return pd.DataFrame(rows), period


def _load3(pitch):
    import re
    meta = (G3 / 'Sample_Game_3_metadata.xml').read_text()
    team_of = dict(re.findall(r'<Player id="(\w+)" teamId="(\w+)">', meta))
    chan = dict(re.findall(r'<PlayerChannel channelId="x" id="(\w+)_x" playerId="(\w+)"', meta))
    ids = sorted(team_of, key=lambda q: (team_of[q], q))
    slot = {q: i for i, q in enumerate(ids)}
    specs = []
    for a, z, body in re.findall(r'<DataFormatSpecification endFrame="(\d+)" separator=":" startFrame="(\d+)">(.*?)</DataFormatSpecification>', meta, re.S):
        specs.append((int(z), int(a), [slot[chan[c]] for c in re.findall(r'playerChannelId="(\w+)_x"', body)]))
    n = G3_PERIODS[1][1]
    xy = np.full((n, len(ids), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    with open(G3 / 'Sample_Game_3_tracking.txt') as f:
        for line in f:
            fr, pl, bl = line.strip().split(':')
            fr = int(fr)
            cols = next(c for a, z, c in specs if a <= fr <= z)
            v = np.array([q.split(',') for q in pl.split(';')], float)
            xy[fr - 1, cols] = v
            ball[fr - 1] = np.array(bl.split(','), float)
    teams = np.array([0 if team_of[q] == 'FIFATMA' else 1 for q in ids])
    return xy * np.array(pitch), ball * np.array(pitch), teams


def _labels(e, period):
    n = len(period)
    lab = np.full(n, 3, np.int8)
    e = e.sort_values(['Start Frame', 'End Frame']).reset_index(drop=True)
    owner = None
    dead_from = None
    cur = 0
    last_period = None
    for _, ev in e.iterrows():
        s = int(ev['Start Frame']) - 1
        en = max(s, int(ev['End Frame']) - 1)
        typ = ev['Type']
        sub = str(ev['Subtype'])
        team = 0 if ev['Team'] == 'Home' else 1
        if ev['Period'] != last_period:
            owner = None
            dead_from = None
            cur = s
            last_period = ev['Period']
        if typ in ONBALL:
            if dead_from is not None:
                lab[dead_from:s] = 2
                if owner is not None:
                    lab[cur:dead_from] = owner
                dead_from = None
                cur = s
                owner = team
            elif owner is None:
                owner = team
                cur = s
            elif team != owner:
                lab[cur:s] = owner
                cur = s
                owner = team
        stop = None
        if typ == 'BALL OUT' or (typ == 'SHOT' and ('GOAL' in sub.split('-')[-1] or sub.endswith('OUT'))):
            stop = en + 1
        elif typ == 'FAULT RECEIVED' or 'OFFSIDE' in sub or 'HAND BALL' in sub:
            stop = s if typ == 'FAULT RECEIVED' else en + 1
        if stop is not None and dead_from is None:
            dead_from = max(stop, cur)
        end_of_period = np.flatnonzero(period == ev['Period'])[-1] + 1
        if owner is not None and dead_from is None:
            lab[cur:end_of_period] = owner   # provisional fill, overwritten by later events
        elif dead_from is not None:
            if owner is not None:
                lab[cur:dead_from] = owner
            lab[dead_from:end_of_period] = 2
    blocks = [(int(np.flatnonzero(period == k)[0]), int(np.flatnonzero(period == k)[-1]) + 1) for k in (1, 2)]
    return lab, blocks


if __name__ == '__main__':
    from cw.masks import runs
    for g in (1, 2, 3):
        lab, blocks = labels(g)
        poss, dead = [], []
        for a, z in blocks:
            for l, s, e in runs(lab[a:z]):
                (poss if l < 2 else dead if l == 2 else []).append((e - s) / 25)
        live = sum(poss) / 60
        print(f'game {g}: shares {np.bincount(lab, minlength=4) / len(lab)} possessions {len(poss)} '
              f'mean {np.mean(poss):.1f}s median {np.median(poss):.1f}s per live minute {len(poss) / live:.2f} '
              f'dead n {len(dead)} mean {np.mean(dead):.1f}s')


def blank_dead(lab, ball_x):
    """Provider's own dead-ball flag: Metrica blanks the ball while play is stopped.  Runs of
    >= 1 s without a ball are dead; elsewhere the event-derived label stands."""
    from cw.masks import runs
    out = lab.copy()
    for v, s, e in runs(~np.isfinite(ball_x)):
        if v and e - s >= 25:
            out[s:e] = np.where(lab[s:e] < 3, 2, 3)
    return out


def load(game, pitch=(105., 68.)):
    """(players, ball, teams, reference, blocks, pitch) in the same layout as the Sportec cache.
    reference = event-derived H/A with dead ball from events or the provider's ball blanking."""
    if game == 3:
        xy, ball, teams = _load3(pitch)
        lab, blocks = labels(3)
        speed = np.full(xy.shape[:2], np.nan)
        for a, z in blocks:
            speed[a + 2:z - 2] = np.linalg.norm(xy[a + 4:z] - xy[a:z - 4], axis=2) * 25 / 4
        players = np.dstack([xy, np.minimum(speed, 12.)])
        return players, np.column_stack([ball, np.full(len(xy), np.nan)]), teams, blank_dead(lab, ball[:, 0]), blocks, pitch
    d = ROOT / f'Sample_Game_{game}'
    arrays, teams = [], []
    ball = None
    for k, side in enumerate(('Home', 'Away')):
        df = pd.read_csv(d / f'Sample_Game_{game}_RawTrackingData_{side}_Team.csv', skiprows=2)
        v = df.iloc[:, 3:].to_numpy(float)
        xy = v.reshape(len(v), -1, 2)
        ball = xy[:, -1]
        arrays.append(xy[:, :-1])
        teams += [k] * (xy.shape[1] - 1)
    xy = np.concatenate(arrays, axis=1) * np.array(pitch)
    lab, blocks = labels(game)
    n = len(xy)
    speed = np.full(xy.shape[:2], np.nan)
    for a, z in blocks:
        speed[a + 2:z - 2] = np.linalg.norm(xy[a + 4:z] - xy[a:z - 4], axis=2) * 25 / 4
    speed = np.minimum(speed, 12.)
    players = np.dstack([xy, speed])
    b = np.column_stack([ball * np.array(pitch), np.full(n, np.nan)])
    return players, b, np.array(teams), blank_dead(lab, b[:, 0]), blocks, pitch
