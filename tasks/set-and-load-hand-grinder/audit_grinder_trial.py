"""Score the tracked dial, bean transfer and crank joint from a trial."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


INTAKE = np.array([.31,.132,.75])
CUP_HOME = np.array([.230,.061,.75])
RUN_LIMIT = 2400
INSTRUCTION = (
    'Align the wooden grind setting pointer with the blue mark on the bench-mounted grinder. '
    'Pour all the beans from the blue dosing cup into the side intake. Return the empty '
    'cup upright to its starting spot, then turn the wooden crank handle counterclockwise '
    'through one full turn as seen from above and release it.'
)


def audit(rows):
    final = rows[-20:]
    if len(final)<20:
        raise ValueError('Expected at least 20 observations')
    positions = np.asarray([r['bean_xyz_m'] for r in rows],dtype=float)-INTAKE
    if positions.shape!=(len(rows),12,3) or not np.isfinite(positions).all():
        raise ValueError('Expected continuous observations for 12 independent beans')
    counts = ((np.linalg.norm(positions[:,:,:2],axis=2)<.034)&
              (positions[:,:,2]>.004)&(positions[:,:,2]<.090)).sum(axis=1)
    joints = [next(j for j in row['articulations'].values() if 'CrankRotation' in j['names'])
              for row in rows]
    names = joints[0]['names']
    motion = np.rad2deg(np.unwrap([j['positions'] for j in joints],axis=0))
    dial = motion[-20:,names.index('GrindSize')]
    crank = motion[:,names.index('CrankRotation')]
    whole = next((i for i in range(len(rows)-4) if np.all(counts[i:i+5]==12)),None)
    after_load = float(crank[-1]-crank[whole]) if whole is not None else 0.
    dial_ok = np.all((dial>=26.)&(dial<=34.)) and np.ptp(dial)<=2.
    released = max(sum(row['contacts_n']['knob']) for row in final)<=.10
    turns = np.clip(after_load/350.,0.,1.) if released else 0.
    pose = np.array([r['cup_pose_xyzw'] for r in final])
    parked = (np.max(np.linalg.norm(pose[:,:3]-CUP_HOME,axis=1))<=.020 and
              Rotation.from_quat(pose[:,3:]).as_matrix()[:,2,2].min()>=np.cos(np.deg2rad(8.)) and
              max(sum(r['cup_contact_n']) for r in final)<=.10)
    retained = int(counts[-20:].min())
    score = round(.25*dial_ok+.35*retained/12.+.30*turns+(.10*parked if retained else 0.),6)
    return {'score':float(score),'steps':len(rows),'dial_set':bool(dial_ok),
            'dial_final_deg':float(dial[-1]),'beans_retained':retained,
            'crank_after_load_deg':after_load,'crank_released':bool(released),'cup_parked':bool(parked)}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trial',type=Path)
    args = parser.parse_args()
    rows = [json.loads(l) for l in (args.trial/'trace.jsonl').open()]
    result = audit(rows)
    (args.trial/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
