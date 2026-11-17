"""Duration-constrained sequence decoding and a rolling mean."""

import numpy as np


def decode(scores, penalties, minimum_frames, allowed_entry=None):
    """Exact maximum-score path with minimum run lengths, including edge runs.

    dp[t,k] ends at exclusive endpoint t in state k. A state either extends
    one frame or starts a complete minimum-duration run after another state.
    This costs O(T*K^2), independent of the longest possession's duration.
    """
    scores = np.asarray(scores,dtype=float)
    penalties = np.asarray(penalties,dtype=float)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError('scores must be a finite two-dimensional array')
    n,k = scores.shape
    if penalties.shape != (k,k) or not np.isfinite(penalties).all():
        raise ValueError('penalties must be a finite square state matrix')
    durations = np.asarray(minimum_frames)
    if durations.shape != (k,) or not np.issubdtype(durations.dtype,np.integer) or (durations<1).any():
        raise ValueError('minimum durations must be positive integers, one per state')
    if not n:
        return np.empty(0,dtype=np.int8)
    if allowed_entry is None:
        allowed_entry = np.ones((n,k),dtype=bool)
    allowed_entry = np.asarray(allowed_entry,dtype=bool)
    if allowed_entry.shape!=(n,k):
        raise ValueError('allowed_entry must have one flag per frame and state')
    # A short input block cannot contain a longer run than itself.
    durations = np.minimum(durations,n)
    cumulative = np.vstack((np.zeros(k), np.cumsum(scores,axis=0)))
    dp = np.full((n+1,k),-np.inf)
    previous = np.full((n+1,k),-1,dtype=np.int16)
    step = np.ones((n+1,k),dtype=np.int32)
    for end in range(1,n+1):
        for state in range(k):
            d = int(durations[state])
            if end<d:
                continue
            if end==d:
                dp[end,state] = cumulative[end,state]
                step[end,state] = d
            else:
                dp[end,state] = dp[end-1,state]+scores[end-1,state]
                previous[end,state] = state
            start = end-d
            if start and allowed_entry[start,state]:
                alternatives = dp[start]-penalties[:,state]
                alternatives[state] = -np.inf
                other = int(np.argmax(alternatives))
                value = alternatives[other]+cumulative[end,state]-cumulative[start,state]
                if value>dp[end,state]:
                    dp[end,state] = value
                    previous[end,state] = other
                    step[end,state] = d
    state = int(np.argmax(dp[n]))
    out = np.empty(n,dtype=np.int8)
    end = n
    while end:
        start = end-int(step[end,state])
        out[start:end] = state
        state = int(previous[end,state])
        end = start
    return out


def rolling_mean(values, radius):
    """Centered finite-observation mean with clipped boundaries."""
    values = np.asarray(values,dtype=float)
    n = len(values)
    index = np.arange(n)
    left,right = np.maximum(0,index-radius),np.minimum(n,index+radius+1)
    known = np.isfinite(values)
    sums = np.r_[0,np.cumsum(np.where(known,values,0))]
    counts = np.r_[0,np.cumsum(known)]
    denominator = counts[right]-counts[left]
    return np.divide(sums[right]-sums[left], denominator,
                     out=np.full(n,np.nan), where=denominator>0)
