"""Measure stand height, pin engagement, and tilt after the grippers leave."""

import argparse
import json
import math
from pathlib import Path


INSTRUCTION = ('Support the monitor and press the orange paddle to unlock the height. '
               'Lower the monitor to the middle slot, let the paddle return, '
               'and tilt the screen back 10 degrees from vertical. Release the monitor.')
STEP_LIMIT = 2400
MIDDLE_HEIGHT = -.056439999490976334


def audit(rows, midpoint=MIDDLE_HEIGHT):
    if len(rows)<20:
        raise ValueError('Monitor scoring requires 20 observed controls')
    final = rows[-20:]
    height_error = max(abs(row['joint']['height_slide']-midpoint) for row in final)
    pin_error = max(abs(row['joint']['height_pin']) for row in final)
    tilt_error = max(abs(math.degrees(row['joint']['tilt_hinge'])+10.) for row in final)
    contact = max(sum(row['contacts_n']['pin']+row['contacts_n']['bezel']) for row in final)
    released = contact<=.1
    height = released and height_error<=.004 and pin_error<=.002
    tilt = released and tilt_error<=2.
    return {
        'score':round(.60*height+.40*tilt,2),
        'steps':len(rows),
        'retraction_mm':1000*max(0.,-min(row['joint']['height_pin'] for row in rows)),
        'final_height_mm':1000*final[-1]['joint']['height_slide'],
        'final_height_error_mm':1000*height_error,
        'final_pin_error_mm':1000*pin_error,
        'final_contact_n':contact,
        'tilt_deg':math.degrees(final[-1]['joint']['tilt_hinge']),
        'tilt_error_max_deg':tilt_error,
        'complete':height and tilt,
    }


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trial',type=Path)
    args = parser.parse_args()
    report = audit([json.loads(line) for line in (args.trial/'trace.jsonl').open()])
    (args.trial/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
