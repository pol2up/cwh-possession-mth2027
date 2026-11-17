"""Derive the two clock offsets (broadcast frame + offset = tactical frame, one per half) that
prepare_challenge.py uses to move the broadcast `valid` flag onto the tactical timeline.

Label-free: both automatic tracking files of the provided match and the broadcast homography are
read; no possession labels, no video.

  1. anchors  : every second of the broadcast tracking is matched to the tactical frame with the
                most similar team formation (quartiles of each team's player positions); matches
                that are not consistent in time with their neighbours are dropped.
  2. segments : around each run of anchors the offset is refined at full frame rate on the same
                formation signature; a segment is kept if the ball positions of the two trackers
                agree under it.
  3. halves   : segments are grouped by offset (one group per half), the weighted median offset
                of each group is refined once more by one-to-one player assignment.

Usage: python camera/clock_offsets.py        prints the two offsets (4014 and 26920)
"""
import sys
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

sys.path.insert(0, '.')
from cw.features import REPO  # noqa: E402  (puts the repository root on the path)
from games import GAME, GAMES, asset_paths, half_video_blocks, source_frames  # noqa: E402
from solution.io import load_tracking  # noqa: E402


def cloud_features(players, teams, pitch):
    """Per team: quartiles of x and of y of its players (12 numbers per frame)."""
    p = np.asarray(players, dtype=float)
    valid = np.isfinite(p[:, :, :2]).all(axis=2)
    out = np.full((len(p), 2, 2, 3), np.nan)
    enough = np.ones(len(p), bool)
    for team in (0, 1):
        members = teams == team
        okay = valid[:, members].sum(axis=1) >= 6
        enough &= okay
        if okay.any():
            xy = np.where(valid[okay][:, members, None], p[okay][:, members, :2], np.nan) / np.asarray(pitch)
            out[okay, team] = np.nanpercentile(xy, [25, 50, 75], axis=1).transpose(1, 2, 0)
    out[~enough] = np.nan
    return out.reshape(len(p), -1)


def coherent_anchor_mask(broadcast, tactical, min_anchors=4, max_gap=75, tolerance=25):
    keep = np.zeros(len(broadcast), bool)
    if not len(broadcast):
        return keep
    db, dt = np.diff(broadcast), np.diff(tactical)
    joined = (db > 0) & (db <= max_gap) & (np.abs(dt - db) <= tolerance)
    breaks = np.r_[0, np.flatnonzero(~joined) + 1, len(broadcast)]
    for start, stop in zip(breaks[:-1], breaks[1:]):
        if stop - start >= min_anchors:
            keep[start:stop] = True
    return keep


def retrieve_anchors(tf, bf, t_indices, b_indices, ratio_limit=.85):
    mean = np.mean(tf, axis=0)
    scale = np.maximum(np.std(tf, axis=0), .025)
    distance, index = cKDTree((tf - mean) / scale).query((bf - mean) / scale, k=min(128, len(tf)))
    best = index[:, 0]
    distinct = np.abs(t_indices[index] - t_indices[best, None]) > 250     # adjacent frames are near-duplicates
    alternatives = np.where(distinct, distance, np.inf).min(axis=1)
    ratio = np.divide(distance[:, 0], alternatives, out=np.ones(len(bf)), where=np.isfinite(alternatives) & (alternatives > 1e-8))
    trusted = (ratio < ratio_limit) & (distance[:, 0] < 2.5)
    b, t = b_indices[trusted], t_indices[best[trusted]]
    coherent = coherent_anchor_mask(b, t)
    return b[coherent], t[coherent]


def flip(features, flip_x, flip_y):
    f = features.reshape(-1, 2, 2, 3).copy()
    if flip_x:
        f[:, :, 0, :] = 1 - f[:, :, 0, ::-1]
    if flip_y:
        f[:, :, 1, :] = 1 - f[:, :, 1, ::-1]
    return f.reshape(-1, 12)


def refine_offsets(tf, bf, b, t):
    scale = np.maximum(np.std(tf[np.isfinite(tf).all(axis=1)], axis=0), .025)
    joined = (np.diff(b) <= 75) & (np.abs(np.diff(t) - np.diff(b)) <= 25)
    breaks = np.r_[0, np.flatnonzero(~joined) + 1, len(b)]
    result = []
    for a, z in zip(breaks[:-1], breaks[1:]):
        if z - a < 4:
            continue
        bi = b[a:z]
        offsets = t[a:z] - bi
        median = int(round(np.median(offsets)))
        if np.percentile(np.abs(offsets - median), 90) > 25:
            continue
        candidates = np.arange(median - 30, median + 31)
        errors = []
        for shift in candidates:
            ti = bi + shift
            valid = (ti >= 0) & (ti < len(tf)) & (bi >= 0) & (bi < len(bf))
            if valid.sum() < 4:
                errors.append(np.inf)
                continue
            distance = np.linalg.norm((tf[ti[valid]] - bf[bi[valid]]) / scale, axis=1)
            finite = np.isfinite(distance)
            errors.append(float(np.median(distance[finite])) if finite.sum() >= 4 else np.inf)
        best = int(np.argmin(errors))
        error = errors[best]
        if not np.isfinite(error) or error > .6:
            continue
        comparable = candidates[np.asarray(errors) <= error + .01]
        result.append(dict(start_b=int(bi[0]), stop_b=int(bi[-1]) + 1, offset=int(candidates[best]),
                           uncertainty_frames=int(comparable[-1] - comparable[0]), anchors=int(z - a)))
    return result


def partial_cloud_error(target, source, target_teams, source_teams):
    distances = []
    for team in (0, 1):
        t = target[(target_teams == team) & np.isfinite(target[:, :2]).all(axis=1), :2]
        b = source[(source_teams == team) & np.isfinite(source[:, :2]).all(axis=1), :2]
        if len(b) < 3 or len(b) > len(t):
            return np.inf, np.inf, 0
        costs = np.linalg.norm(b[:, None] - t[None, :], axis=2)
        rows, columns = linear_sum_assignment(costs)
        distances.extend(costs[rows, columns].tolist())
    return float(np.median(distances)), float(np.percentile(distances, 90)), len(distances)


def stable_clock_blocks(start, stop, offset, weight):
    order = np.argsort(offset)
    result = []
    for group in np.split(order, np.flatnonzero(np.diff(offset[order]) > 25) + 1):
        if len(group) < 3 or weight[group].sum() < 20:
            continue
        ranking = group[np.argsort(offset[group])]
        index = np.searchsorted(np.cumsum(weight[ranking]), weight[ranking].sum() / 2)
        result.append(dict(start_b=int(start[group].min()), stop_b=int(stop[group].max()), offset=int(offset[ranking[index]])))
    return sorted(result, key=lambda x: x['start_b'])


def main():
    pitch = GAMES[GAME]['pitch']
    tp, tb, tt = load_tracking(*(asset_paths(source='tactical')[k] for k in ('atd_meta', 'atd_raw')), source_frames('tactical'), pitch)
    bpaths = asset_paths(source='broadcast')
    bp, bb, bt = load_tracking(bpaths['atd_meta'], bpaths['atd_raw'], source_frames('broadcast'), pitch)
    h = pd.read_csv(bpaths['homography'], usecols=['valid', 'reprojection_error'])
    valid, reprojection = h['valid'].to_numpy().astype(bool), h['reprojection_error'].to_numpy()
    graded = np.zeros(len(tb), bool)
    for _, a, z in half_video_blocks(GAME):
        graded[a:z + 1] = True
    tcount = np.isfinite(tp[:, :, :2]).all(axis=2).sum(axis=1)
    bcount = np.isfinite(bp[:, :, :2]).all(axis=2).sum(axis=1)
    tf, bf = cloud_features(tp, tt, pitch), cloud_features(bp, bt, pitch)

    # 1. anchors, on a coarse grid; the pitch orientation of the broadcast tracking is the one with most anchors
    ti, bi = np.arange(0, len(tb), 10), np.arange(0, len(bb), 25)
    te = np.isfinite(tf[ti]).all(axis=1) & (tcount[ti] >= 18) & (tcount[ti] <= 24) & graded[ti]
    be = np.isfinite(bf[bi]).all(axis=1) & (bcount[bi] >= 18) & (bcount[bi] <= 24) & valid[bi] & (reprojection[bi] < 2)
    candidates = []
    for fx, fy in ((False, False), (True, False), (False, True), (True, True)):
        b, t = retrieve_anchors(tf[ti][te], flip(bf[bi][be], fx, fy), ti[te], bi[be])
        candidates.append((len(b), fx, fy, b, t))
    _, fx, fy, b, t = candidates[int(np.argmax([c[0] for c in candidates]))]
    bf = flip(bf, fx, fy)
    other_ball = bb[:, :2].copy()
    if fx:
        other_ball[:, 0] = pitch[0] - other_ball[:, 0]
    if fy:
        other_ball[:, 1] = pitch[1] - other_ball[:, 1]

    # 2. segments, screened by agreement of the two trackers' ball positions
    tokay = (tcount >= 18) & (tcount <= 24) & graded
    bokay = (bcount >= 18) & (bcount <= 24) & valid & (reprojection < 1)
    scale = np.maximum(np.std(tf[np.isfinite(tf).all(axis=1)], axis=0), .025)
    accepted = []
    for segment in refine_offsets(tf, bf, b, t):
        sb = np.arange(segment['start_b'], segment['stop_b'])
        st = sb + segment['offset']
        inrange = (st >= 0) & (st < len(tb))
        sb, st = sb[inrange], st[inrange]
        distance = np.linalg.norm((tf[st] - bf[sb]) / scale, axis=1)
        okay = tokay[st] & bokay[sb] & np.isfinite(distance) & (distance < .6)
        both = okay & np.isfinite(other_ball[sb]).all(axis=1) & np.isfinite(tb[st, :2]).all(axis=1)
        errors = np.linalg.norm(tb[st, :2] - other_ball[sb], axis=1)
        screen = both & ((sb // 50) % 2 == 0)
        if (screen.sum() >= 10 and segment['uncertainty_frames'] <= 12
                and np.median(errors[screen]) < 1.5 and np.percentile(errors[screen], 90) < 4):
            accepted.append(segment)
    blocks = stable_clock_blocks(*(np.asarray([s[k] for s in accepted], int) for k in ('start_b', 'stop_b', 'offset', 'anchors')))

    # 3. one offset per half, refined by one-to-one player assignment (original, unflipped coordinates)
    eligible = valid & (reprojection < 1) & (bcount >= 8) & (bcount <= 22)
    offsets = []
    for block in blocks:
        sample = np.arange(block['start_b'], block['stop_b'], 250)
        sample = sample[eligible[sample]]
        shifts = np.arange(block['offset'] - 15, block['offset'] + 16)
        errors = []
        for shift in shifts:
            observations = []
            for i in sample:
                j = i + shift
                if not 0 <= j < len(tp) or not graded[j]:
                    continue
                median, p90, count = partial_cloud_error(tp[j], bp[i], tt, bt)
                if count and p90 < 8:
                    observations.append(median)
            errors.append(float(np.median(observations)) if len(observations) >= 10 else np.inf)
        best = int(np.argmin(errors))
        if np.isfinite(errors[best]) and errors[best] <= 2:
            offsets.append(int(shifts[best]))
            print(f'broadcast frames {block["start_b"]}-{block["stop_b"]}: offset {offsets[-1]} (formation error {errors[best]:.2f} m)')
    print('offsets', offsets)


if __name__ == '__main__':
    main()
