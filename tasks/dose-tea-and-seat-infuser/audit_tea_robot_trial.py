"""Independently grade a dry-tea trial from observed object and leaf poses."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


TIN = np.array([.27,.20,.75])
CUP = np.array([.18,.32,.75])
CAP_AREA = np.array([.34,.12,.75])
STATION_YAW = Rotation.from_euler('z',90,degrees=True)


def audit(rows):
    window = rows[-20:]
    if len(window) != 20 or not all(set(row.get('parts',{})) == {'cap','filter'} for row in window):
        raise ValueError('Expected at least one second of observed cap and filter poses')

    cap = np.array([row['parts']['cap'] for row in window])
    basket = np.array([row['parts']['filter'] for row in window])
    tin_counts, basket_counts, outside_counts = [],[],[]
    for row,pose in zip(window,basket):
        leaves = np.asarray(row['leaf_positions_station_m'])
        if leaves.shape != (160,3) or not np.isfinite(leaves).all():
            raise ValueError('Expected 160 finite leaf centers')
        in_tin = (np.linalg.norm(leaves[:,:2],axis=1)<.037)&(leaves[:,2]>.004)&(leaves[:,2]<.055)
        local = Rotation.from_quat(pose[3:]).inv().apply(STATION_YAW.apply(leaves)+TIN-pose[:3])
        in_basket = (np.linalg.norm(local[:,:2],axis=1)<.028)&(local[:,2]>.004)&(local[:,2]<.033)
        tin_counts.append(int(in_tin.sum()))
        basket_counts.append(int(in_basket.sum()))
        outside_counts.append(int((~(in_tin|in_basket)).sum()))
    cap_ok = (np.max(np.linalg.norm(cap[:,:2]-CAP_AREA[:2],axis=1))<=.022
              and np.max(np.abs(cap[:,2]-CAP_AREA[2]-.003))<=.006
              and np.max(np.linalg.norm(cap[:,:3]-cap[-1,:3],axis=1))<=.002)
    dose_ok = min(basket_counts)>=24 and max(basket_counts)<=33
    basket_error = float(np.max(np.linalg.norm(basket[:,:2]-CUP[:2],axis=1)))
    basket_seated = (basket_error<=.0045 and np.max(np.abs(basket[:,2]-CUP[2]-.0165))<=.004
                     and np.max(np.linalg.norm(basket[:,:3]-basket[-1,:3],axis=1))<=.0015
                     and np.min(Rotation.from_quat(basket[:,3:]).apply([0.,0.,1.])[:,2])>=.998)
    spill_free = max(outside_counts)==0
    score = .2*bool(cap_ok)+.35*bool(dose_ok)+.35*bool(basket_seated)+.1*bool(dose_ok and spill_free)
    return {'score':round(score,2),
            'steps':len(rows),'cap_parked':bool(cap_ok),'dose_in_range':bool(dose_ok),
            'filter_seated':bool(basket_seated),'no_spill':bool(spill_free),
            'filter_leaf_count_range':[min(basket_counts),max(basket_counts)],
            'tin_leaf_count_range':[min(tin_counts),max(tin_counts)],
            'max_leaves_outside_containers':max(outside_counts),'filter_max_planar_error_m':basket_error}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trial',type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.trial/'trace.jsonl').read_text().splitlines()]
    score = audit(rows)
    (args.trial/'audit.json').write_text(json.dumps(score,indent=2)+'\n')
    print(json.dumps(score))
