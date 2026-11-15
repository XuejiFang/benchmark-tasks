"""Score the released rail height and all three released louver angles."""

import argparse
import json
from pathlib import Path

import numpy as np


ORIGIN = np.array([.36,.035,.63])
HEIGHT = .085
TILT = -28.
STEP_LIMIT = 2400
INSTRUCTION = ('Raise the blue lower rail handle to the stripe on the window frame. '
               'Press the orange lever until all the slats angle 28 degrees down toward you. '
               'Release both handles and leave the blind in this position.')


def audit(rows):
    if len(rows)<20:
        raise ValueError('Blind scoring requires at least 20 observed controls')
    final = rows[-20:]
    heights = np.array([r['lift_m'] for r in final])
    angles = np.array([r['slat_deg'] for r in final])
    release_rail = max(sum(r['rail_contact_n']) for r in final)<=.1
    release_slats = max(sum(r['slat_contact_n']) for r in final)<=.1
    height_error = float(np.max(np.abs(heights-HEIGHT)))
    angle_error = float(np.max(np.abs(angles-TILT)))
    lift = release_rail and height_error<=.008
    tilt = release_slats and angle_error<=4.
    return {'score':round(.45*lift+.55*tilt,3),'steps':len(rows),'rail_height_m':float(heights[-1]),
            'height_error_max_m':height_error,'slat_angles_deg':angles[-1].tolist(),
            'tilt_error_max_deg':angle_error,'rail_released':release_rail,'slats_released':release_slats,
            'complete':lift and tilt}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trial',type=Path)
    args = parser.parse_args()
    report = audit([json.loads(line) for line in (args.trial/'trace.jsonl').open()])
    (args.trial/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
