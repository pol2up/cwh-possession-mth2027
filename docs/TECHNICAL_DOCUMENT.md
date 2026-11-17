# Technical document: possession labels for FC Barcelona vs Athletic Club

## 1. Approach

The file is produced from the tactical automatic tracking (player positions and speeds, team
membership, the ball where the tracker reports one) and from one signal of the broadcast bundle:
the per-frame flag saying whether the broadcast homography is valid, i.e. whether the broadcast
shows the pitch. No video pixels, audio or Smart Tagging are read.

Per frame we compute 132 tracking features. *Base* (55): per team the nearest player's distance to
the ball, a contact measure and median player speed; ball visibility, speed and position relative to
the pitch; rolling means; time since and until each team was clearly closest to the ball. *Gap*
(21): whether the ball is reported, for how long it has been and will be missing, and what its last
and next sightings looked like. *Shape* (37): each team's advancement, depth, width and collective
movement, normalised by attacking direction. *Trajectory* (19): touches and direction changes of the
visible ball, with thresholds set per half from that match's own ball motion, without labels, in the
same way for every match. Features look backwards and forwards in time; the method is offline.

Three models read these features.

1. A gradient-boosted tree classifier (scikit-learn `HistGradientBoostingClassifier`, 300
   iterations) predicts home / away / dead per frame. Every training row is also presented with the
   teams exchanged, and predictions are averaged over both orientations, so no side is preferred.
2. A second classifier of the same family re-reads the first one's probabilities at twelve time
   offsets up to 15 s before and after and as rolling means over 1 to 20 s. It is trained on predictions made for each
   Sportec match by a model that had not seen that match.
3. A sequence model: a two-layer bidirectional GRU (hidden size 128) over 30-second windows at
   5 Hz, trained with cross-entropy in which short possessions weigh more. At inference it runs on
   all five 5 Hz phases and both team orientations.

The probabilities of models 2 and 3 are combined as an equal-weight geometric mean and decoded per
half (section 4). Blend weight and decoding cost were fixed on leave-one-match-out results on the
Sportec matches.

*Camera-view channel.* A broadcast director leaves the wide view (replay, close-up) mostly while the
ball is dead. Before decoding, the dead-ball probability of each frame is multiplied by a likelihood
ratio that depends on whether the broadcast shows the pitch and on how long that state lasts (five
duration classes). The table and its weight (1.0) were fitted on 15 SkillCorner matches and confirmed
on 5 others; nothing in it comes from the provided match. The broadcast flag is moved onto the
tactical timeline with one fixed frame offset per half (+4,014 and +26,920, from matching player
formations between the two tracking files, without labels). The channel changes 3.0% of graded frames.

*Pre-kick-off rule.* The organisers' FAQ states that the second half is graded from frame 110,808 but
kicks off at 113,742 and that those frames are all `D`; the file is written accordingly.

*Hidden ball in training.* The training matches have a complete ball track; the provided match does
not. Each training match is used several times: ball fully visible; hidden mainly around stoppages
and for some seconds of live play (two random draws); and the latter with positional noise and
player drop-outs. None of these takes data from the provided match. One provider parks the ball at a
fixed coordinate during every stoppage; that frozen ball is replaced by simulated movement so that
"ball not moving" cannot be learned as the definition of a stoppage.

## 2. Assets used and dropped

- **Tactical tracking: used.** The input of all three models.
- **Broadcast homography validity: used** as the camera-view channel. An earlier step that learned
  the relation on the provided match itself was removed after the organisers' clarification.
- **Broadcast/tactical clock: used as a fixed offset per half.** Formation matching and whistle-band
  audio agree to within a few frames; a frame-exact mapping could not be certified. The channel works
  on stretches of seconds, so this matters little.
- **Video, ball recovery: dropped.** A pretrained ball detector recovered 73% of annotated ball
  positions on external multi-camera footage; attempts to raise that failed their preset gates.
- **Video, player continuity through ball occlusions: dropped.** It reached too few gaps to change
  possession decisions, and its team estimate was fitted on images of the provided match.
- **Other broadcast shot-state and whistle signals: dropped.** Their only use would have been a
  model fitted on the provided match.
- **Smart Tagging: not an input.** Its effect could not be validated on another match. It was read
  once as an automatic check: its team tag matches our restart team in 23 of 24 tagged set pieces.
- **Broadcast tracking: not a model input.** Another timeline, partial pitch coverage. Its player
  positions serve only to derive the two clock offsets.

## 3. Training data

Nothing from the provided match (tracking, video, audio, Smart Tagging or the public ground truth)
is a training example, label or tuning target.

| Dataset | Source | Licence | Matches | Labels |
|---|---|---|---|---|
| An integrated dataset of spatiotemporal and event data in elite soccer (DFL / Sportec Solutions); Bassek et al., *Scientific Data* 12, 195 (2025) | figshare, doi 10.6084/m9.figshare.28196177.v1 | CC BY 4.0 | 7; all three models | The provider's per-frame `BallStatus` and `BallPossession`; never features. |
| Metrica Sports sample data | github.com/metrica-sports/sample-data, commit `e706dd5` | none stated; source acknowledged as requested | 3; first and sequence models | Our derivation from the event files: the team of each on-ball event (pass, shot, recovery, ball lost, set piece, carry) holds possession until an on-ball event of the other team; dead ball from stoppage events or wherever the provider's tracking has no ball for at least 1 s. |
| SkillCorner Open Data | github.com/SkillCorner/opendata, commit `4340d27` | MIT | 20: 15 to fit the camera table, 5 to confirm it; none trains the three models | Dead / live from the provider's match phases (provider inference, not human labels); camera state = whether the provider gives a pitch projection. |

The Metrica games were added to both models because the hidden labels come from the same provider.
Tested by leaving one Metrica game out at a time (rule written beforehand): spellF1 0.833 / 0.711 /
0.805 to 0.859 / 0.719 / 0.819 (ball visible / the provided match's on-off pattern / gaps mostly in
stoppages), all nine game-by-condition cells improved; on two Sportec folds the mean changed by -0.001.

## 4. Possession rules implemented

- **H / A**: possession as the training labels define it, changing at every touch that the provider
  credits: a loose ball, a contested ball or a ball in flight stays with the team that last played it. No separate rule
  for *controlled* possession is applied; a short spell of the other team is a possession of its own
  whenever the evidence for it outweighs the decoder's cost of two changes.
- **D**: the ball is out of play, as the providers mark it; plus the pre-kick-off frames of the
  second half.
- **X**: every frame outside the two halves. Inside them the pipeline never abstains.
- **Decoding.** Each half is decoded exactly by dynamic programming. A sequence scores the sum of
  log-probabilities (per second) minus 0.3 for every change of label, the same for a turnover and
  for entering or leaving a stoppage. A possession lasts at least 0.4 s, a stoppage at least 0.6 s.
- **Restarts.** No dead-ball ownership is output; the first possession after a stoppage comes from
  the models.
- **Missing ball.** A frame without a ball is neither dead nor unknown by rule; the gap and shape
  features decide, and training includes many hidden-ball examples.
- **Symmetry.** No rule or learned weight distinguishes home from away.

## 5. Limitations

- The hidden-match score is unknown. No setting was fitted or tuned on the public five minutes and
  no file was chosen by its score there. Three disclosures. (a) That window was looked at once early
  on to diagnose a missed-stoppage failure, which led to the dead-ball fix of section 1. (b) On
  16 November the public labels were inspected once for the possession definition, as the organisers'
  clarification allows, with the question and its consequence written beforehand: does any possession
  shorter than 3 s lie directly between two possessions of the other team? Three of 17 do (2.8 s,
  0.8 s, 1.3 s). We had two finished decoders, this one and one requiring 3 s of control for a change
  of team; that answer decided for this one. (c) During development the build printed a score on that window as a sanity
  check; no choice was made on it.
- The training providers' possession definitions are not the challenge's. The file follows the
  providers' (348 possessions). If the reference is closer to controlled possession, it predicts too
  many.
- Held out, against the providers' own labels and without the camera channel: spellF1 0.765 / 0.670
  / 0.734 on Sportec (leave one match out, before the Metrica games were added to the sequence
  model) and the Metrica figures of section 3. After the final training no Metrica game is held out,
  and the Sportec check of the final recipe covers two folds only.
- The camera channel cannot be evaluated on our labelled tracking matches, which have no broadcast
  view. Its measured support is on SkillCorner (possession-spell F1 0.374 to 0.416, dead-ball F1 0.756
  to 0.920 on the 5 confirmation matches), with a weaker base model, another decoder and a
  provider-derived reference. "The broadcast shows the pitch" is measured differently there and here;
  that the two behave alike was checked only through label-free statistics (no pitch view on 33.2% of
  frames here, 30.5 to 43.2% there).
- In the leave-one-match-out benchmark the second model for a test match was trained on predictions
  from first models that had seen that match; refitting everything match-disjoint moved the estimate
  by less than 0.001.
- The Metrica labels are our derivation, not an official possession track. The hidden-ball
  conditions are simulations of an automatic tracker, not measurements of one.
- Very short possessions (under about 1.5 s) are mostly missed; turnover boundaries are within 0.5 s
  of the reference about 80% of the time on held-out matches.
- Stoppages for fouls are placed from movement only; the whistle is not observed.
- The sequence model is trained on a GPU, which is not bit-reproducible; its weights are published
  and the build runs inference only. Identical regeneration was shown on macOS / Apple silicon only.
- With more time: an event-level model for whether a touch established possession, and broadcast
  tracking to fill the ball during live play.

## 6. Reproduction

Python 3.12; NumPy 2.2.6, SciPy 1.16.3, pandas 2.3.3, scikit-learn 1.7.2, PyTorch 2.9.1
(`requirements.txt`). Inputs and commands are in the repository's `README.md`; the build
takes about 45 minutes on a laptop CPU and writes `pipeline/out/possession_final.csv`, SHA256
`332b9e8c5a39b4ab335e5e0f57ea4ae66ed89243c9c3d25f034ff974deaff965`. Three things enter the build as
fixed inputs, each with the script that produced it: the sequence-model weights
(`sequence_training/train.py`), the camera table (`camera/fit_table.py`; the comparison that chose
its weight is not included) and the two clock offsets (`camera/clock_offsets.py`).
