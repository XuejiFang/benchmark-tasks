"""Score selective watering and a released pitcher returned to its coaster."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ORIGIN = np.array([.24, 0., .64])
POTS = {'target':(.175,0.), 'neighbor':(.175,.153)}
DROP_COUNT = 196
STEP_LIMIT = 2400
INSTRUCTION = ('Pour about half the water from the teal pitcher into the terracotta planter. '
               'Save the rest in the pitcher for later, keep the green planter dry, and return the '
               'pitcher upright to its coaster. Release the pitcher.')


def classify(points, pose):
    local = points-ORIGIN
    pitch = Rotation.from_quat(pose[3:]).inv().apply(points-pose[:3])
    in_pitcher = (np.abs(pitch[:,0])<.0295)&(np.abs(pitch[:,1])<.0295)&(pitch[:,2]>.002)&(pitch[:,2]<.067)
    pots = {name:(np.abs(local[:,0]-x)<.049)&(np.abs(local[:,1]-y)<.048)
                 &(local[:,2]>.002)&(local[:,2]<.073) for name,(x,y) in POTS.items()}
    return {name:int(mask.sum()) for name,mask in pots.items()} | {
        'pitcher':int(in_pitcher.sum()),'spill':int((~(in_pitcher|pots['target']|pots['neighbor'])).sum())}


def audit(rows):
    if len(rows)<20:
        raise ValueError('Watering requires at least 20 observed controls')
    final = rows[-20:]
    minimum = min(r['counts']['target'] for r in final)
    neighbor = max(r['counts']['neighbor'] for r in rows)
    spill = max(r['counts']['spill'] for r in final)
    retained = max(r['counts']['pitcher'] for r in final)
    poses = np.array([r['pitcher_pose_xyzw'] for r in final])
    offsets = poses[:,:3]-ORIGIN
    z_axis = Rotation.from_quat(poses[:,3:]).as_matrix()[:,2,2]
    returned = bool(np.linalg.norm(offsets[:,:2],axis=1).max()<=.018 and np.abs(offsets[:,2]).max()<=.009
                    and z_axis.min()>=np.cos(np.deg2rad(8.))
                    and max(sum(r['contact_n']) for r in final)<=.1)
    maximum = max(r['counts']['target'] for r in final)
    reserve_min = min(r['counts']['pitcher'] for r in final)
    clean = neighbor==0 and spill<=8
    partial = clean and minimum>=45 and maximum<=134 and reserve_min>=50
    full = clean and minimum>=80 and maximum<=116 and reserve_min>=76
    score = .5*partial+.3*full+.2*(full and returned)
    return {'score':round(score,3),'steps':len(rows),'target_water_min':minimum,'target_water_max':maximum,
            'neighbor_water_max':neighbor,'spill_max':spill,'pitcher_water_min':reserve_min,
            'pitcher_water_max':retained,'pitcher_returned':returned,'complete':score==1.}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trial',type=Path)
    args = parser.parse_args()
    report = audit([json.loads(line) for line in (args.trial/'trace.jsonl').open()])
    (args.trial/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
