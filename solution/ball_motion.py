"""Short-horizon ball kinematics from observed pitch positions only."""

import numpy as np


def ball_speed(ball,fps):
    """Metres/second, using a 0.16 s centered baseline where fully observed.

    Near an occlusion boundary use only consecutive observations. Missing
    positions never become zero speed and no gap is interpolated. The third
    export field is retained by the reader but is not needed here.
    """
    b=np.asarray(ball,dtype=float)
    if b.ndim!=2 or b.shape[1]<2 or fps<=0:
        raise ValueError('ball must contain x/y positions and fps must be positive')
    n=len(b)
    out=np.full(n,np.nan)
    if n<2:
        return out
    known=np.isfinite(b[:,:2]).all(axis=1)
    pairs=known[1:] & known[:-1]
    displacement=np.linalg.norm(b[1:,:2]-b[:-1,:2],axis=1)*fps
    out[1:]=np.where(pairs,displacement,np.nan)
    if n>=5:
        complete=np.convolve(known.astype(int),np.ones(5,dtype=int),'valid')==5
        centered=np.linalg.norm(b[4:,:2]-b[:-4,:2],axis=1)*fps/4
        out[2:-2]=np.where(complete,centered,out[2:-2])
    return out
