"""Score setting, verification lift and reseating for a physical selector dumbbell."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


INSTRUCTION = ('Move the orange selector into the lower slot. Lift the handle with both weight pairs attached, '
               'then set it down on the cradle and release it.')
STEP_LIMIT = 2400
SLOT = np.array([-.087,0.,.042])
PARTS = ('Handle','Light','Heavy')


def slot_error(row):
    handle = np.asarray(row['poses']['Handle'])
    pin = np.asarray(row['poses']['Selector'])
    local = Rotation.from_quat(handle[3:]).inv().apply(pin[:3]-handle[:3])
    return float(np.linalg.norm(local-SLOT))


def force(row,keys):
    return sum(sum(row['contacts_n'][key]) for key in keys)


def audit(rows):
    if len(rows)<20:
        raise ValueError('Dumbbell scoring requires at least 20 observed controls')
    home = np.array([rows[0]['poses'][name][:3] for name in PARTS])
    final = rows[-20:]
    pin_error = max(slot_error(row) for row in final)
    pin_present = pin_error<=.013 and max(force(row,['Selector']) for row in final)<=.12
    max_height = 0.
    current_span = longest_span = 0
    for row in rows:
        positions = np.array([row['poses'][name][:3] for name in PARTS])
        lowest_rise = float(np.min(positions[:,2]-home[:,2]))
        max_height = max(max_height,lowest_rise)
        if (lowest_rise>=.045 and slot_error(row)<=.015
                and force(row,['Light','Heavy'])<=.20):
            current_span += 1
            longest_span = max(longest_span,current_span)
        else:
            current_span = 0
    lifted = longest_span>=12
    return_error = max(float(np.max(np.linalg.norm(
        np.array([row['poses'][name][:3] for name in PARTS])-home,axis=1))) for row in final)
    released = max(force(row,[*PARTS,'Selector']) for row in final)<=.12
    returned = return_error<=.015 and released
    return {'score':round(.45*pin_present+.35*(pin_present and lifted)+.20*(pin_present and lifted and returned),2),
            'steps':len(rows),'selector_retained':pin_present,'selector_error_m':pin_error,'lifted':lifted,
            'lift_controls':longest_span,'common_lift_m':max_height,'returned':returned,'released':released,
            'return_error_m':return_error}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trial',type=Path)
    args = parser.parse_args()
    report = audit([json.loads(line) for line in (args.trial/'trace.jsonl').open()])
    (args.trial/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
