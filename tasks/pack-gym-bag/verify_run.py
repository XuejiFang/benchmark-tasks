"""Independently recompute the expert's packing score from its recorded trial."""

import argparse
import json
from pathlib import Path

import numpy as np

from audit_dream_run import recompute


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    run = parser.parse_args().run
    if (run / 'failure.json').exists():
        raise SystemExit('The recorded trial failed before completion.')
    result = json.loads((run / 'result.json').read_text())
    rows = [json.loads(line) for line in (run / 'trace.jsonl').read_text().splitlines()]
    with np.load(run / 'scoring_vertices.npz', allow_pickle=False) as vertices:
        scores, geometry, finite = recompute(rows, vertices)

    actions_ok = all(
        row['step'] == index
        and len(row['actions']) == 14
        and np.isfinite(row['actions']).all()
        for index, row in enumerate(rows)
    )
    score_ok = all(row['score'] == score for row, score in zip(rows, scores))
    report = {
        'steps': len(rows),
        'duration_s': len(rows) / 20,
        'score': scores[-1] if scores else None,
        'valid': bool(
            40 <= len(rows) == result['steps'] <= 3000
            and actions_ok and score_ok and geometry and finite
            and scores[-1] == result['score']
            and (scores[-1] == 1.0) == result['success']
        ),
    }
    (run / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    return 0 if report['valid'] and report['score'] == 1.0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
