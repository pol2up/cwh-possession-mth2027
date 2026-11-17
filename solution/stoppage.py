"""Collective-motion stoppage features; contains no ball-missing indicator."""

import numpy as np

from solution.model import rolling_mean


def motion_features(players,blocks,fps):
    p=np.asarray(players,dtype=float)
    if p.ndim!=3 or p.shape[2]!=3 or not p.shape[1]:
        raise ValueError('expected frame x slot x (x,y,speed) players')
    speed=p[:,:,2]
    known=np.isfinite(p[:,:,:2]).all(axis=2) & np.isfinite(speed) & (speed>=0)
    data=np.where(known,speed,np.nan)
    valid=known.any(axis=1)
    base=np.full((len(p),8),np.nan)
    if valid.any():
        base[valid,:5]=np.nanpercentile(data[valid],[10,25,50,75,90],axis=1).T
        counts=known[valid].sum(axis=1)
        base[valid,5]=((speed[valid]<.5) & known[valid]).sum(axis=1)/counts
        base[valid,6]=((speed[valid]>2) & known[valid]).sum(axis=1)/counts
        base[valid,7]=((speed[valid]>4) & known[valid]).sum(axis=1)/counts
    names=['speed_q10','speed_q25','speed_q50','speed_q75','speed_q90',
           'fraction_below_0p5','fraction_above_2','fraction_above_4']
    features=np.full((len(p),24),np.nan)
    all_names=names+[f'{name}_mean{seconds}s' for seconds in (1,3) for name in names]
    for start,stop in blocks:
        features[start:stop,:8]=base[start:stop]
        for window,seconds in enumerate((1,3),start=1):
            radius=max(1,round(seconds*fps/2))
            for column in range(8):
                features[start:stop,window*8+column]=rolling_mean(base[start:stop,column],radius)
    # Do not infer current player motion solely from neighbouring observations.
    features[~valid]=np.nan
    return features,all_names
