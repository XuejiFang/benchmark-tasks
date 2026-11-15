"""Score the final second from measured grain, lid and carton states."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ORIGIN = np.array([.24,0.,.64])
LID_SEAT = ORIGIN+[.205,0.,.0825]
INVENTORY = 80
INSTRUCTION = (
    'Remove the green lid from the green storage jar and set it flat on the worktop. '
    'Grasp both side tabs of the open cardboard carton and pour the dry grains into '
    'the green jar. Keep grains out of the brown jar and off the worktop. Return '
    'the carton upright to where it started, then reseat the green lid and release it.'
)
RUN_LIMIT = 2400


def audit(rows):
    tail = rows[-20:]
    if len(tail)<20:
        raise ValueError('The audit requires 20 physically observed final controls')
    points = np.array([row['grain_xyz_m'] for row in tail])-ORIGIN
    if points.shape!=(20,INVENTORY,3) or not np.isfinite(points).all():
        raise ValueError('Expected all 80 observed grain positions in every scoring frame')
    height = points[:,:,2]
    target = ((np.linalg.norm(points[:,:,:2]-[.205,0.],axis=2)<.050)&(height>.003)&(height<.088))
    other = ((np.linalg.norm(points[:,:,:2]-[.205,.150],axis=2)<.044)&(height>.003)&(height<.082))
    carton_pose = np.array([row['carton_pose_xyzw'] for row in tail])
    carton_orientation = Rotation.from_quat(carton_pose[:,3:])
    relative = carton_orientation.inv().as_matrix()
    local = np.einsum('fij,fnj->fni',relative,points+ORIGIN-carton_pose[:,None,:3])
    inside = ((np.abs(local[:,:,0])<.042)&(np.abs(local[:,:,1])<.027)&
              (local[:,:,2]>.003)&(local[:,:,2]<.126))
    outside = ~(target|other|inside)
    contained = int(target.sum(axis=1).min())
    clean = contained==INVENTORY and not (other|outside).any()
    lid_pose = np.array([row['lid_pose_xyzw'] for row in tail])
    lid_attitude = Rotation.from_quat(lid_pose[:,3:]).as_matrix()[:,2,2]
    lid_error = float(np.linalg.norm(lid_pose[:,:3]-LID_SEAT,axis=1).max())
    lid_force = max(max(row['lid_fingertip_force_n']) for row in tail)
    lid_seated = lid_error<=.004 and lid_attitude.min()>=np.cos(np.deg2rad(2.)) and lid_force<.1
    carton_error = float(np.linalg.norm(carton_pose[:,:3]-ORIGIN,axis=1).max())
    carton_force = max(max(row['fingertip_force_n']) for row in tail)
    carton_upright = carton_orientation.as_matrix()[:,2,2].min()>=np.cos(np.deg2rad(5.))
    carton_parked = carton_error<=.018 and carton_upright and carton_force<.1
    score = round(.55*contained/INVENTORY+.20*clean+
                  (.15*lid_seated+.10*carton_parked if contained else 0.),6)
    return {'score':score,'steps':len(rows),'target_min':contained,
            'other_max':int(other.sum(axis=1).max()),'outside_max':int(outside.sum(axis=1).max()),
            'max_lid_seat_error_m':lid_error,'max_carton_return_error_m':carton_error,
            'lid_seated':bool(lid_seated),'carton_returned':bool(carton_parked)}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trial',type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.trial/'trace.jsonl').read_text().splitlines()]
    result = audit(rows)
    (args.trial/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
