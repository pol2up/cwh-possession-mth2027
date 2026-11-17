"""Train the sequence model on the seven Sportec matches and the three Metrica sample games and
write weights/seq_nim3.pt.  The committed weights were produced by this script on an Apple GPU;
GPU training is not bit-reproducible, so the submitted file is built from the committed weights.

Usage: python sequence_training/train.py        (refuses to overwrite existing weights)
Needs `python -m cw.harness prepare` to have been run.
"""
import sys, warnings
warnings.simplefilter('ignore'); sys.path.insert(0, '.')
from cw import harness as H, seq as S
from sequence_training import data as D

wf = H.HERE / 'weights' / 'seq_nim3.pt'
assert not wf.exists(), 'weights already exist'
seqs = [s for m in H.MATCHES for s in D.sportec_seqs(m)] + [s for g in (1, 2, 3) for s in D.metrica_seqs(g)]
net, sc = S.train(seqs, H.names(), seed=2027, log=lambda s: print(s, flush=True))
S.save(wf, net, sc, H.names())
print('saved', wf)
