"""Score a lunchbox's contents, lid and released latches from observed poses."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


LID = '/Container/Geometry/Lid'
SIDES = ('front','back','left','right')
STEP_LIMIT = 2400
INSTRUCTION = ('Stand the gray divider in the center guides of the lunchbox. Put the orange packet '
               'in the farther compartment and the green packet in the nearer compartment. '
               'Place the lid on the box, press all four latches closed, and release it.')


def audit(rows):
    if len(rows)<20:
        raise ValueError('Expected at least 20 observed controls')
    tail = rows[-20:]
    box = np.asarray([r['poses']['tub'] for r in tail])
    inverse = Rotation.from_quat(box[:,3:]).inv()

    def relative(name):
        pose = np.asarray([r['poses'][name] for r in tail])
        pos = inverse.apply(pose[:,:3]-box[:,:3])
        axis = (inverse*Rotation.from_quat(pose[:,3:])).as_matrix()[:,2,2]
        tilt = float(np.rad2deg(np.arccos(np.clip(axis,-1.,1.))).max())
        released = max(sum(r['contacts_n'][name]) for r in tail)<=.1
        return pos,tilt,released

    divider,divider_tilt,divider_released = relative('divider')
    separator = bool(np.all(np.abs(divider)<=np.array([.010,.009,.012]))
                     and divider_tilt<=5. and divider_released)
    food = {}
    for name,direction in (('wrapped_item_left',1.),('wrapped_item_right',-1.)):
        pos,tilt,released = relative(name)
        side = direction*pos[:,1]
        food[name] = bool(np.all((side>=.023)&(side<=.057)) and np.abs(pos[:,0]).max()<=.031
                          and np.all((pos[:,2]>=0.)&(pos[:,2]<=.016)) and tilt<=22. and released)

    lid,tilt,lid_released = relative('lid')
    lid_seated = bool(np.linalg.norm(lid[:,:2],axis=1).max()<=.010
                      and np.abs(lid[:,2]-.0669).max()<=.009 and tilt<=5. and lid_released)
    latches = {}
    for side in SIDES:
        maximum = max(abs(r['joints'][LID]['positions'][r['joints'][LID]['names'].index('latch_'+side)])
                      for r in tail)
        angle = float(np.rad2deg(maximum))
        released = max(sum(r['contacts_n'][side]) for r in tail)<=.1
        latches[side] = {'max_angle_deg':angle,'passed':bool(lid_seated and released and angle<=4.)}
    complete = separator and all(food.values()) and lid_seated and all(v['passed'] for v in latches.values())
    score = round(.15*separator+.15*sum(food.values())+.15*lid_seated+.10*sum(v['passed'] for v in latches.values()),6)
    return {'score':score,'steps':len(rows),'complete':complete,'divider':separator,'packets':food,
            'lid_seated':lid_seated,'latches':latches}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trial',type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.trial/'trace.jsonl').open()]
    result = audit(rows)
    (args.trial/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
