"""Score released cups and the upper rack from an observed dishwasher trial."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


RACK_BODY = '/Machine/Links/Upper_Rack_Assembly'
RACK_SITES = {'tall_cup': (.154,-.099,.0773), 'short_cup': (.167,-.190,.078)}
WOOD_START = (.294,-.442,.590)
RUN_LIMIT = 2400
INSTRUCTION = (
    'Pull the upper rack out by its orange tab. Load the tall blue cup and the '
    'short light-tan cup into the upper rack with their openings facing down. '
    'Leave the dark wooden cup on its side worktop. Push the loaded upper rack '
    'back to its starting position and release it.'
)


def audit(rows):
    tail = rows[-20:]
    if len(tail)<20:
        raise ValueError('Expected at least 20 observed controls')
    rail = np.array([r['rail_m'] for r in rows])
    starting = rail[0]
    extended = starting - rail >= .225
    opened = np.lib.stride_tricks.sliding_window_view(extended, 5).all(axis=1).any()
    end_error = float(np.abs(rail[-20:]-starting).max())
    released = max(sum(r['contacts_n']['rack']) for r in tail)<=.1
    rack = np.array([r['bodies'][RACK_BODY]['pose_xyzw'] for r in tail])
    inverse = Rotation.from_quat(rack[:,3:]).inv()
    cups = {}
    for name,site in RACK_SITES.items():
        pose = np.array([r['loads'][name] for r in tail])
        up = Rotation.from_quat(pose[:,3:]).as_matrix()[:,2,2]
        tilt = float(np.rad2deg(np.arccos(np.clip(-up,-1.,1.))).max())
        error = float(np.linalg.norm(inverse.apply(pose[:,:3]-rack[:,:3])-site,axis=1).max())
        contact = float(max(sum(r['cup_contact_n'][name]) for r in tail))
        cups[name] = {'site_error_m':error,'tilt_deg':tilt,'max_finger_force_n':contact,
                      'passed':error<=.032 and tilt<=20. and contact<=.10}
    wood = np.array([r['wood_pose_xyzw'] for r in tail])
    wood_error = float(np.linalg.norm(wood[:,:3]-WOOD_START,axis=1).max())
    wood_up = Rotation.from_quat(wood[:,3:]).as_matrix()[:,2,2]
    wood_stayed = bool(wood_error<=.018 and wood_up.min()>=np.cos(np.deg2rad(12.)))
    count = sum(c['passed'] for c in cups.values())
    returned = end_error<=.010 and released and wood_stayed and opened and count==2
    score = round(.20*opened+.30*count+.20*returned,6)
    return {'score':float(score),'steps':len(rows),'opened':bool(opened),'cups':cups,
            'wood_cup_max_displacement_m':wood_error,'wood_cup_unchanged':wood_stayed,
            'final_rail_error_m':end_error,'rail_released':released,'complete':bool(returned)}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trial',type=Path)
    args = parser.parse_args()
    rows = [json.loads(l) for l in (args.trial/'trace.jsonl').open()]
    result = audit(rows)
    (args.trial/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
