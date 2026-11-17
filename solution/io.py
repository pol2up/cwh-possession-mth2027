"""Readers for the tracking files."""

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def load_tracking(meta, raw, n_frames, pitch):
    root = ET.parse(meta).getroot()
    team_ids = [node.get('id') for node in root.iter('Team')]
    if len(team_ids) != 2 or len(set(team_ids)) != 2:
        raise ValueError('metadata must define exactly two distinct teams')
    slots = [node.get('teamId') for node in root.iter('Player')]
    if not slots or any(team not in team_ids for team in slots):
        raise ValueError('missing player slots or unknown slot team')
    teams = np.array([team_ids.index(team) for team in slots], dtype=np.int8)
    players = np.full((n_frames, len(slots), 3), np.nan, dtype=np.float64)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float64)
    count = 0
    with Path(raw).open() as handle:
        for row, line in enumerate(handle):
            if row >= n_frames:
                raise ValueError('too many tracking frames')
            parts = line.strip().split(':')
            if len(parts) != 3:
                raise ValueError(f'frame row {row}: expected three colon-separated fields')
            try:
                frame_id = int(parts[0])
            except ValueError as error:
                raise ValueError(f'frame row {row}: invalid frame id') from error
            if frame_id != row + 1:
                raise ValueError(f'frame row {row}: expected id {row+1}, got {frame_id}')
            groups = parts[1].split(';')
            # Some FIFA exports terminate the slot list with a semicolon.
            if groups and groups[-1] == '':
                groups.pop()
            if len(groups) != len(slots):
                raise ValueError(f'frame row {row}: slot count differs from metadata')
            try:
                values = [[float(x) for x in group.split(',')] for group in groups]
                b = [float(x) for x in parts[2].rstrip(';').split(',')]
                if any(len(v) != 3 for v in values) or len(b) != 3:
                    raise ValueError('expected x,y,speed triples')
                players[row] = values
                ball[row] = b
            except ValueError as error:
                raise ValueError(f'frame row {row}: malformed coordinate triple') from error
            if np.isinf(players[row]).any() or np.isinf(ball[row]).any():
                raise ValueError(f'frame row {row}: infinite coordinate')
            count += 1
    if count != n_frames:
        raise ValueError(f'tracking frame count {count} != {n_frames}')
    players[:, :, :2] *= np.asarray(pitch)
    ball[:, :2] *= np.asarray(pitch)
    return players, ball, teams


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
