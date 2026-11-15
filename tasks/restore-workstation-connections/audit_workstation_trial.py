"""Audit both released connections and the loose cable collar."""

import argparse
import json
from math import cos, radians
from pathlib import Path


INSTRUCTION = ('Plug the two blue cable ends into the green-marked ports on the two devices. '
               'Place the gold collar in the open cable guide and let go of both ends and the collar.')
PORT_X = .390
PORT_Z = .803
GUIDE_X = .232
MIN_ALIGNMENT = cos(radians(6.))


def audit(rows):
    if len(rows)<20:
        raise ValueError('Expected at least 20 controls to verify the released assembly')

    def connector(key,sensor,y):
        observations = []
        for row in rows:
            x, lateral, height, qx, qy, qz, _ = row[key]
            axis_toward_port = 1-2*(qy*qy+qz*qz)
            shell_right_side_up_or_inverted = abs(1-2*(qx*qx+qy*qy))
            aligned = (x-PORT_X<=.0105 and abs(lateral-y)<=.001 and abs(height-PORT_Z)<=.001
                       and min(axis_toward_port,shell_right_side_up_or_inverted)>=MIN_ALIGNMENT)
            observations.append((x-PORT_X,aligned,sum(row['contacts_by_object_n'][sensor]),
                                 abs(lateral-y),abs(height-PORT_Z)))
        tail = observations[-20:]
        stable = 0
        established = None
        for i,(depth,aligned,contact,*_) in enumerate(observations):
            stable = stable+1 if depth>=.0075 and aligned and contact<=.1 else 0
            if stable==20:
                established = i-19
                break
        maintained = established is not None and all(depth>=.0075 and aligned
                                                     for depth,aligned,*_ in observations[established:])
        return {'passed':maintained and all(depth>=.0075 and aligned and contact<=.1
                                           for depth,aligned,contact,*_ in tail),
                'final_min_depth_mm':1000*min(v[0] for v in tail),
                'final_max_lateral_mm':1000*max(v[3] for v in tail),
                'final_max_vertical_mm':1000*max(v[4] for v in tail),
                'retained_after_release':maintained}

    first = connector('first_pose_xyzw','first',-.220)
    second = connector('second_pose_xyzw','plug',.220)
    order_ok = first['passed'] and second['passed']
    collar = all(abs(row['collar_pose_xyzw'][0]-GUIDE_X)<=.018 and abs(row['collar_pose_xyzw'][1])<=.045
                 and .752<=row['collar_pose_xyzw'][2]<=.766 and sum(row['contacts_by_object_n']['collar'])<=.1
                 for row in rows[-20:])
    score = round(.35*first['passed']+.35*second['passed']+.30*(order_ok and collar),6)
    return {'score':score,'steps':len(rows),'first':first,'second':second,
            'collar_in_guide':collar,'complete':order_ok and collar}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trial',type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.trial/'trace.jsonl').open()]
    result = audit(rows)
    (args.trial/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
