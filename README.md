# Possession labels for FC Barcelona vs Athletic Club

More Than A Hack 2027, Initial Challenge.

This repository regenerates our submitted possession file (`H` / `A` / `D` / `X` for each of the
199,319 frames of the tactical video) from the provided data and public external datasets. The
method, the training data, the possession rules and the limitations are described in
[docs/TECHNICAL_DOCUMENT.md](docs/TECHNICAL_DOCUMENT.md).

No match data is in this repository.

## Data you need

```
data/
  tactical/     BAR-ATH_tactical_atd.xml, BAR-ATH_tactical_atd.txt     (provided)
  broadcast/    BAR-ATH_broadcast_homography.csv                       (provided)
                BAR-ATH_broadcast_atd.xml, BAR-ATH_broadcast_atd.txt   (provided; clock offsets only)
  external/
    sportec/          DFL_02_01_matchinformation_*.xml and DFL_04_03_positions_raw_observed_*.xml
                      of the seven matches of "An integrated dataset of spatiotemporal and event
                      data in elite soccer" (Bassek et al., 2025; CC BY 4.0;
                      doi 10.6084/m9.figshare.28196177.v1)
    metrica-sample/   Sample_Game_1, Sample_Game_2, Sample_Game_3 of
                      github.com/metrica-sports/sample-data
    skillcorner/      the match folders of github.com/SkillCorner/opendata, commit 4340d27
                      (camera table only)
```

## Install

Python 3.12.

```bash
pip install -r requirements.txt
```

## Reproduce the submitted file

```bash
python -m experiments.sportec          # parse the Sportec matches
cd pipeline
python prepare_challenge.py            # tracking and time-aligned broadcast flag of the provided match
python -m cw.harness prepare           # features of the seven Sportec matches (about 10 min)
python -m cw.harness fit all300m3_ni   # out-of-fold first-stage probabilities (about 20 min)
python make_final.py                   # first stage, second stage, sequence model, decode
python camera/build.py                 # camera-view channel
python camera/prekickoff.py            # pre-kick-off rule, writes out/possession_final.csv
python ../evaluation/validate_submission.py out/possession_final.csv
```

`pipeline/out/possession_final.csv` is the submitted file, SHA256
`332b9e8c5a39b4ab335e5e0f57ea4ae66ed89243c9c3d25f034ff974deaff965`.
The build takes about 45 minutes on a 10-core laptop, CPU only.

The sequence-model weights (`pipeline/weights/seq_nim3.pt`) were trained with
`python sequence_training/train.py`. GPU training is not bit-reproducible, so the weights are
committed and the build runs inference only.

## Fixed inputs of the build

The build reads two fixed inputs. Each can be re-derived, in `pipeline/`:

```bash
python camera/clock_offsets.py         # the two broadcast-to-tactical clock offsets in prepare_challenge.py (4014, 26920)
python camera/fit_table.py             # the camera table in camera/camera_channel.json, from SkillCorner Open Data
```

## Layout

| path | content |
|---|---|
| `pipeline/cw/` | features, ball-visibility simulation for training, the three models, decoding |
| `pipeline/camera/` | camera-view channel, its fitted table and clock offsets, and the pre-kick-off rule |
| `pipeline/sequence_training/` | training of the sequence model |
| `solution/`, `experiments/` | tracking loader, base features, decoder, reader for the Sportec matches |
| `games.py`, `evaluation/` | from the organisers' repository (asset paths, validator), under its MIT licence (`evaluation/LICENSE`) |
