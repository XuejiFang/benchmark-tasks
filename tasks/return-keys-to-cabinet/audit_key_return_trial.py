"""Grade observed shaft intersections, releases and closed-door state."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from key_return_score import HOOK_Y, blade_in_guide, settled_hangs, threaded_on_shaft


def audit(rows, cabinet_x, cabinet_y):
    final = rows[-20:]
    if len(final)<20:
        raise ValueError('Scoring needs 20 physically observed control frames')
    stable = settled_hangs(rows,cabinet_x,cabinet_y=cabinet_y)
    per_key = {}
    for name,hook in HOOK_Y.items():
        frames = [row['keys'][name] for row in final]
        poses = np.array([row['pose_xyzw'] for row in frames])
        attitude = Rotation.from_quat(poses[:,3:])
        forces = [row.get('gripper_force_n') for row in frames]
        peak = max(map(sum,forces)) if all(f is not None for f in forces) else None
        seated_frames = sum(blade_in_guide(pose, cabinet_x, hook + cabinet_y) for pose in poses)
        per_key[name] = {'scored':bool(stable[name] and peak is not None and seated_frames == len(final)),
                         'threaded_frames':sum(threaded_on_shaft(pose,cabinet_x,hook+cabinet_y) for pose in poses),
                         'blade_seated_frames':seated_frames,
                         'gripper_contact_max_n':peak,
                         'position_drift_max_m':float(np.max(np.linalg.norm(poses[:,:3]-poses[-1,:3],axis=1))),
                         'rotation_drift_max_deg':float(np.rad2deg((attitude[-1].inv()*attitude).magnitude().max()))}
    open_error = max(abs(row['hinge_deg']) for row in final)
    door_released = max(sum(row['contacts_n']['door']) for row in final) <= .1
    count = sum(row['scored'] for row in per_key.values())
    complete = count == 2 and door_released and open_error <= 3.
    return {'score':1. if complete else .4*count,'steps':len(rows),'keys':per_key,
            'door_released':door_released,'door_max_error_deg':open_error,'complete':complete}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trial',type=Path)
    parser.add_argument('--cabinet-x',type=float,default=.55)
    parser.add_argument('--cabinet-y',type=float,default=-.06)
    parser.add_argument('--report', type=Path, help='Write the audit here instead of in the trial directory.')
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.trial/'trace.jsonl').read_text().splitlines()]
    result = audit(rows,args.cabinet_x,args.cabinet_y)
    (args.report or args.trial/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
