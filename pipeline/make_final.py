"""Build the base file.

  first stage  : gradient-boosted classifier on tracking features, trained on other labelled matches
  second stage : second classifier re-reading the first stage's probabilities with temporal context
  sequence     : GRU (weights in weights/, trained by sequence_training/train.py; inference on CPU)
  decode       : geometric 50:50 blend of second stage and sequence model, duration-constrained decoding

Usage: python make_final.py            -> out/v6.csv
Needs prepare_challenge.py, `cw.harness prepare` and `cw.harness fit all300m3_ni` to have been run.
"""
import hashlib, sys, warnings
import numpy as np
warnings.simplefilter('ignore')
sys.path.insert(0, '.')
from cw import harness as H, challenge as C, stack as K, seq as S

NAME, OUT_NAME, COST, BLEND = 'all300m3_ni', 'v6', .3, .5
model, cols, nm = C.train_full(NAME)
pa = C.predict(model, cols, nm)
ps = K.apply(K.train(NAME, H), pa, C.BLOCKS)
net, sc, names = S.load(H.HERE / 'weights' / 'seq_nim3.pt', device='cpu')
assert names == H.names()
q = S.predict(net, sc, C.features(), names, C.BLOCKS)
np.save(H.CACHE / f'final_{OUT_NAME}_stack.npy', ps.astype(np.float32))
np.save(H.CACHE / f'final_{OUT_NAME}_seq.npy', q.astype(np.float32))
codes = H.decode_probs(S.blend(ps, q.astype(float), BLEND), C.BLOCKS, cost=COST)
path = C.write(codes, OUT_NAME)
print(path, hashlib.sha256(path.read_bytes()).hexdigest())
print({h: (v['n_poss'], [round(x, 3) for x in v['share']]) for h, v in C.describe(codes).items()})
