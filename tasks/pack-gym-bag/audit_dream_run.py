"""Recompute a Dream rollout's packing score from saved geometry and measured state.

Works locally without Isaac/USD. A completed zero-score rollout is valid evidence;
an interrupted API/simulator run is not assigned a fabricated official score.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from layout import ITEMS, BAG_POS, BAG_R, WORLD_ROT, task_pos, corners


def recompute(rows, vertices):
    held = dict.fromkeys(ITEMS, 0)
    geometry_agrees = True
    scores = []
    finite = True
    for row in rows:
        score = 0.
        for name in ITEMS:
            state = row['objects'][name]
            values = np.r_[state['pos'], state['quat_xyzw'], state['velocity'],
                           state['angular_velocity'], state['bag_force'], state['gripper_normal_force_N']]
            finite &= bool(np.isfinite(values).all())
            rotation = Rotation.from_quat(state['quat_xyzw']).as_matrix()
            cloud = (vertices[name] @ rotation.T + state['pos'] - task_pos(BAG_POS)) @ (WORLD_ROT @ BAG_R)
            low, high = cloud.min(0), cloud.max(0)
            geometry_agrees &= bool(np.allclose(low, state['bag_min'], atol=1e-6) and
                                    np.allclose(high, state['bag_max'], atol=1e-6))
            center = (corners(name) @ rotation.T + state['pos']).mean(0)
            qualifies = (
                np.all(low >= [-.30, -.18645, .0435]) and np.all(high <= [.30, .18645, .245])
                and state['gripper_normal_force_N'] < .02
                and all(np.linalg.norm(np.asarray(robot['tcp']) - center) > .16 for robot in row['robots'].values())
                and np.linalg.norm(state['velocity']) < .01
                and np.linalg.norm(state['angular_velocity']) < .1
            )
            held[name] = held[name] + 1 if qualifies else 0
            if held[name] >= 40:
                score += .125 if name.startswith('shoe') else .25
        scores.append(score)
    return scores, geometry_agrees, finite


def audit(path):
    meta = json.loads((path / 'metadata.json').read_text())
    result = json.loads((path / 'result.json').read_text())
    transcript = json.loads((path / 'dream_transcript.json').read_text())
    rows = [json.loads(line) for line in (path / 'trace.jsonl').read_text().splitlines()]
    vertices = np.load(path / 'scoring_vertices.npz', allow_pickle=False)
    scores, geometry_agrees, finite = recompute(rows, vertices)
    checks = {
        'complete_run': bool(rows) and len(rows) == result['steps'] and result['status'] == 'completed',
        'no_failure_artifacts': not any((path / file).exists() for file in ('failure.json', 'video_failure.json')),
        'correct_step_budget': len(rows) <= meta['max_steps'] and (len(rows) == meta['max_steps'] or result['success']),
        'sequential_finite_actions': all(row['step'] == i and len(row['actions']) == 14 and np.isfinite(row['actions']).all()
                                         for i, row in enumerate(rows)),
        'finite_object_states': finite,
        'mesh_bounds_recomputed': geometry_agrees,
        'all_step_scores_recomputed': all(abs(score - row['score']) < 1e-9 for score, row in zip(scores, rows)),
        'final_score_matches': bool(scores) and scores[-1] == result['score'] and result['success'] == (scores[-1] == 1.),
        'actual_api_requests': bool(transcript['requests']) and not transcript['dry_run'],
        'source_snapshots_match': all(hashlib.sha256((path / name).read_bytes()).hexdigest() == digest
                                      for name, digest in meta['script_sha256'].items()),
        'three_camera_input_hashes_recorded': all(set(row['policy_input_image_sha256']) == {'top_cam', 'left_cam', 'right_cam'} for row in rows),
    }
    report = {'kind': 'independent_dream_packing_audit', 'valid_run': all(checks.values()),
              'smoke_only': result['smoke_only'], 'official_baseline': not result['smoke_only'] and all(checks.values()),
              'score': scores[-1] if scores else None, 'steps': len(rows), 'checks': {k: bool(v) for k, v in checks.items()},
              'api_requests': len(transcript['requests']),
              'trace_sha256': hashlib.sha256((path / 'trace.jsonl').read_bytes()).hexdigest(),
              'audit_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (path / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    report = audit(parser.parse_args().run)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['valid_run'] else 1)
