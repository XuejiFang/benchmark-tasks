"""Score the released textile in the fixed-size laundry compartment."""

import argparse
import json
from pathlib import Path

import numpy as np


INSTRUCTION = ('Use the folding board to fold the tan towel. Reopen the board, '
               'place the folded towel fully inside the wooden compartment, and release it.')
STEP_LIMIT = 2400
CUBBY_CENTER = np.array([.025,0.,.692])
CUBBY_SIZE = np.array([.208,.268,.014])


def audit(rows):
    if len(rows)<20:
        raise ValueError('Towel scoring requires at least 20 observed controls')
    final = rows[-20:]
    low = np.min([row['cloth_min_xyz'] for row in final],axis=0)
    high = np.max([row['cloth_max_xyz'] for row in final],axis=0)
    width = max(row['towel_width_m'] for row in final)
    edge_spacing = max(abs(row['edge_x_m'][0]-row['edge_x_m'][1]) for row in final)
    board_error = max(abs(row['board_deg']) for row in final)
    contacts = max(sum(row['force_n']) for row in final)
    hands_clear = all(np.linalg.norm(np.maximum(
        np.maximum(np.asarray(row['cloth_min_xyz'])-row[key],
                   np.asarray(row[key])-row['cloth_max_xyz']),0.))>=.022
        for row in final for key in ('tcp_xyz_m','left_tcp_xyz_m'))
    board_used = any(row['board_deg']>=140. and row['towel_width_m']<=.225 for row in rows)
    folded = bool(board_used and width<=.210 and edge_spacing<=.092 and board_error<=4.
                  and contacts<=.1 and hands_clear)
    limits = np.array([.102,.127])
    floor = CUBBY_CENTER[2]+CUBBY_SIZE[2]/2
    supported = all(abs(row['cloth_z_p10_m']-floor)<=.006 for row in final)
    contained = bool(folded and supported and
                     np.all(low[:2]>=CUBBY_CENTER[:2]-limits) and
                     np.all(high[:2]<=CUBBY_CENTER[:2]+limits) and
                     low[2]>=floor-.003 and high[2]<=floor+.038)
    return {'score':round(.55*folded+.45*contained,2),'steps':len(rows),'folded':folded,
            'contained':contained,'hands_clear':hands_clear,'board_used':board_used,'terminal_width_m':width,
            'opposite_ends_space_m':edge_spacing,'board_error_deg':board_error,
            'cloth_min_xyz_m':low.tolist(),'cloth_max_xyz_m':high.tolist()}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trial',type=Path)
    args = parser.parse_args()
    score = audit([json.loads(line) for line in (args.trial/'trace.jsonl').open()])
    (args.trial/'audit.json').write_text(json.dumps(score,indent=2)+'\n')
    print(json.dumps(score))
