"""Score released, retained stowage of the folding shower seat."""

import argparse
import json
from pathlib import Path

STEP_LIMIT = 2400
INSTRUCTION = (
    'Hold the orange retaining lever clear. Fold the shower seat up to the wall, '
    'let the lever click back in front of it, and release the seat.'
)


def audit(rows):
    if len(rows) < 20:
        raise ValueError('Scoring requires 20 controls')
    window = rows[-20:]
    opened = any(row['catch_deg'] >= -18. for row in rows)
    raised = any(row['fold_deg'] >= 85. for row in rows)
    closed = max(row['catch_deg'] for row in window) <= -40.
    settled = min(row['fold_deg'] for row in window) >= 85.
    clear = max(sum(sum(force) for force in row['contacts_n'].values()) for row in window) <= .1
    success = opened and raised and closed and settled and clear
    return {
        'score': 1.0 if success else .5 if opened and raised else .2 if opened else 0.0,
        'steps': len(rows),
        'keeper_opened': opened,
        'seat_raised': raised,
        'keeper_closed': closed,
        'seat_retained': settled,
        'released': clear,
        'final_minimum_fold_deg': min(row['fold_deg'] for row in window),
        'final_maximum_keeper_deg': max(row['catch_deg'] for row in window),
        'final_maximum_contact_n': max(sum(sum(force) for force in row['contacts_n'].values()) for row in window),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trial', type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.trial/'trace.jsonl').open()]
    result = audit(rows)
    (args.trial/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
