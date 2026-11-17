"""Anonymous team-control features, with offline context confined to each half."""

import numpy as np

from solution.ball_motion import ball_speed
from solution.model import rolling_mean
from solution.stoppage import motion_features


def control_features(players,ball,teams,blocks,fps,pitch):
    p,b,t=np.asarray(players),np.asarray(ball),np.asarray(teams)
    visible=np.isfinite(b[:,:2]).all(axis=1)
    distance=np.linalg.norm(p[:,:,:2]-b[:,None,:2],axis=2)
    distance[~np.isfinite(distance)]=np.inf
    near=np.full((len(b),2),np.inf);speed=np.full((len(b),2),np.nan)
    for team in (0,1):
        if not (t==team).any():continue
        near[:,team]=distance[:,t==team].min(axis=1)
        known=np.isfinite(p[:,t==team,:2]).all(axis=2)&np.isfinite(p[:,t==team,2])&(p[:,t==team,2]>=0)
        values=np.where(known,p[:,t==team,2],np.nan);valid=known.any(axis=1)
        speed[valid,team]=np.nanmedian(values[valid],axis=1)
    contact=np.exp(-.5*(near/2.)**2)
    nearest=np.where(np.isfinite(near),np.minimum(near,30)/30,np.nan)
    bs=np.full(len(b),np.nan)
    for start,stop in blocks:bs[start:stop]=ball_speed(b[start:stop],fps)
    outside=visible&((b[:,0]<-.5)|(b[:,0]>pitch[0]+.5)|(b[:,1]<-.5)|(b[:,1]>pitch[1]+.5))
    motion,motion_names=motion_features(p,blocks,fps)
    names=['home_nearest','away_nearest','home_contact','away_contact','home_motion','away_motion',
           'ball_visible','ball_speed','ball_outside']+motion_names
    base=np.column_stack((nearest,contact,speed/6,visible.astype(float),np.minimum(bs,30)/30,outside.astype(float),motion))
    columns=[base];all_names=list(names)
    for seconds in (1,3,5):
        window=np.full((len(b),6),np.nan)
        for start,stop in blocks:
            for j in range(6):window[start:stop,j]=rolling_mean(base[start:stop,j],max(1,round(seconds*fps/2)))
        columns.append(window);all_names.extend(f'{name}_mean{seconds}s' for name in names[:6])
    history=np.full((len(b),4),10.)
    for start,stop in blocks:
        index=np.arange(stop-start)
        for side in (0,1):
            control=(near[start:stop,side]<2)&(near[start:stop,side]+.75<near[start:stop,1-side])
            last=np.maximum.accumulate(np.where(control,index,-1000000))
            next_=np.minimum.accumulate(np.where(control,index,1000000)[::-1])[::-1]
            history[start:stop,side*2]=np.minimum((index-last)/fps,10)
            history[start:stop,side*2+1]=np.minimum((next_-index)/fps,10)
    columns.append(history);all_names.extend(('home_since_control','home_until_control','away_since_control','away_until_control'))
    return np.column_stack(columns),all_names
