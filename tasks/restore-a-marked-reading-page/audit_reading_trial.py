"""Score a released book's placement, open cover, and turned page."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ORIGIN = np.array([.25,.055,.71])
INSTRUCTION = ('Pick the green book up with both arms and set it on the wooden reading stand. '
               'Open its cover, turn the blue-marked page, and release the book.')
STEP_LIMIT = 2400


def audit(rows):
    if len(rows)<20:
        raise ValueError('Reading task scoring needs at least 20 observed controls')
    final = rows[-20:]
    pose = np.asarray([r['poses']['Spine'] for r in final])
    offsets = pose[:,:3]-ORIGIN
    tilt_deg = np.rad2deg(np.arccos(np.clip(Rotation.from_quat(pose[:,3:]).as_matrix()[:,2,2],-1.,1.)))
    seated = bool(np.linalg.norm(offsets[:,:2],axis=1).max()<=.016 and np.abs(offsets[:,2]).max()<=.010
                  and tilt_deg.max()<=9. and max(sum(r['contacts_n']['Spine']) for r in final)<=.1)
    angles = {name:max(r['angles_deg'][name] for r in final) for name in ('CoverFold','LeafFold')}
    opened = seated and angles['CoverFold']<=-150. and max(sum(r['contacts_n']['Cover']) for r in final)<=.1
    turned = opened and angles['LeafFold']<=-150. and max(sum(r['contacts_n']['Page']) for r in final)<=.1
    return {'score':round(.35*seated+.3*opened+.35*turned,4),'steps':len(rows),
            'book_on_stand':seated,'cover_open':opened,'marked_page_turned':turned,
            'final_angles_deg':angles,'complete':turned}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trial',type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.trial/'trace.jsonl').open()]
    result = audit(rows)
    (args.trial/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
