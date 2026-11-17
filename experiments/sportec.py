"""Strict local reader for OTHER-game Sportec Open DFL v1 position references.

BallStatus and BallPossession are labels, never model input features. This
module is not used to read or infer labels for the challenge match.
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from solution.ball_motion import ball_speed
from solution.io import sha256


PERIODS={'firstHalf':0,'secondHalf':1}


def load_match(metadata,tracking):
    root=ET.parse(metadata).getroot()
    environment=root.find('.//Environment')
    if environment is None:raise ValueError('missing pitch dimensions')
    pitch=(float(environment.get('PitchX')),float(environment.get('PitchY')))
    if not np.isfinite(pitch).all() or min(pitch)<=0:raise ValueError('invalid pitch dimensions')
    sides={team.get('Role'):team for team in root.iter('Team')}
    if not {'home','guest'}<=sides.keys():raise ValueError('home and guest teams required')
    ids=[];teams=[]
    for index,side in enumerate(('home','guest')):
        for player in sides[side].iter('Player'):
            ids.append(player.get('PersonId'));teams.append(index)
    if not ids or None in ids or len(set(ids))!=len(ids):raise ValueError('invalid or duplicate player IDs')
    slots={identity:i for i,identity in enumerate(ids)}
    chunks=[];ball_rows=[];current=None;rows=[]
    for event,node in ET.iterparse(tracking,events=('start','end')):
        if event=='start' and node.tag=='FrameSet':
            period=node.get('GameSection')
            if period not in PERIODS:raise ValueError('unsupported game section')
            identity=node.get('PersonId')
            is_ball=str(node.get('TeamId')).upper()=='BALL'
            current=(period,identity,is_ball);rows=[]
        elif event=='end' and node.tag=='Frame':
            if current is not None:
                period,identity,is_ball=current
                if is_ball or identity in slots:
                    frame=int(node.get('N'));x=float(node.get('X'));y=float(node.get('Y'))
                    if is_ball:
                        status=node.get('BallStatus');owner=node.get('BallPossession')
                        if status not in ('0','1'):raise ValueError('unknown ball status')
                        label=2 if status=='0' else {'1':0,'2':1}.get(owner,3)
                        ball_rows.append((period,frame,x,y,label))
                    else:rows.append((frame,x,y,float(node.get('S'))/3.6))
            node.clear()
        elif event=='end' and node.tag=='FrameSet':
            if rows:chunks.append((current[0],slots[current[1]],np.asarray(rows,float)))
            current=None;rows=[];node.clear()
    ball_rows.sort(key=lambda row:(PERIODS[row[0]],row[1]))
    keys=[(row[0],row[1]) for row in ball_rows]
    if not keys or len(set(keys))!=len(keys):raise ValueError('missing or duplicate ball frames')
    lookup={key:i for i,key in enumerate(keys)}
    players=np.full((len(keys),len(ids),3),np.nan)
    for period,slot,values in chunks:
        matched=[(j,lookup[(period,int(row[0]))]) for j,row in enumerate(values) if (period,int(row[0])) in lookup]
        if not matched:continue
        source,target=np.asarray(matched,int).T
        if len(np.unique(target))!=len(target):raise ValueError('duplicate player frames')
        players[target,slot]=values[source,1:]
    ball=np.full((len(keys),3),np.nan)
    ball[:,:2]=np.asarray([row[2:4] for row in ball_rows])
    reference=np.asarray([row[4] for row in ball_rows],np.int8)
    shift=np.asarray(pitch)/2
    players[:,:,:2]+=shift;ball[:,:2]+=shift
    if np.isinf(players).any() or np.isinf(ball).any():raise ValueError('infinite positions')
    boundaries=[0]+[i for i in range(1,len(keys)) if keys[i][0]!=keys[i-1][0]]+[len(keys)]
    blocks=list(zip(boundaries[:-1],boundaries[1:]))
    for start,stop in blocks:
        frame_ids=np.asarray([key[1] for key in keys[start:stop]])
        if not np.array_equal(frame_ids,np.arange(frame_ids[0],frame_ids[0]+len(frame_ids))):
            raise ValueError('missing internal ball-frame IDs')
        ball[start:stop,2]=ball_speed(ball[start:stop],25)
    return players,ball,np.asarray(teams,np.int8),blocks,reference,pitch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',default='data/external/sportec')
    args=parser.parse_args();directory=Path(args.directory)
    for metadata in sorted(directory.glob('*matchinformation*.xml')):
        identity=metadata.stem.split('DFL-MAT-')[-1]
        matches=list(directory.glob(f'*positions_raw_observed*{identity}.xml'))
        if len(matches)!=1:continue
        tracking=matches[0];out=directory/f'{identity}_parsed.npz'
        players,ball,teams,blocks,reference,pitch=load_match(metadata,tracking)
        np.savez_compressed(out,players=players,ball=ball,teams=teams,blocks=np.asarray(blocks),reference=reference,pitch=np.asarray(pitch))
        report={'match':identity,'frames':len(ball),'slots':len(teams),'blocks':blocks,
                'reference_counts':{name:int((reference==i).sum()) for i,name in enumerate('HADX')},
                'inputs':{metadata.name:sha256(metadata),tracking.name:sha256(tracking)},
                'parsed_sha256':sha256(out),'parser_sha256':sha256(__file__),
                'reference_source':'Sportec Open DFL v1 BallStatus/BallPossession, other game; never feature inputs'}
        out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
